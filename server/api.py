"""FastAPI server with endpoints for voice interaction processing.

Endpoints:
  POST /api/process_interaction  — main voice pipeline (STT→LLM→TTS)
  GET  /audio/beep/{type}        — serve pre-generated beep WAVs for Sonos
  POST /api/sonos_beep           — trigger Sonos beep (called by Pi client)
  GET  /audio/tts_latest         — serve most recent TTS audio for Sonos pull
  GET  /api/health               — liveness + service status
  GET  /api/conversation/history — inspect current LLM history
  POST /api/conversation/clear   — reset LLM history
  GET  /                         — service info

Security:
  - IP allowlist: _require_allowed_ip() blocks all IPs not in ALLOWED_CLIENT_IPS,
    except /api/health which is open for monitoring tools.
  - Rate limiting: 10 voice requests/minute per IP (generous for a single Pi;
    blocks runaway request loops).
  - Audio and wake_word inputs validated by Pydantic before reaching orchestrator.

Sonos range request support:
  /audio/tts_latest and /audio/beep/* support HTTP Range requests because Sonos
  sends "Range: bytes=0-" to seek before playing.  Without range support, Sonos
  silently fails (returns error, never plays audio).
"""
import logging
import time
from collections import deque
from threading import Lock
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, ValidationError

from shared.models import (
    ProcessInteractionRequest,
    ProcessInteractionResponse,
    HealthCheckResponse,
    Status,
    HealthStatus
)
from shared.utils import decode_audio_base64
from server.orchestrator import Orchestrator
from server.config import ALLOWED_CLIENT_IPS

logger = logging.getLogger(__name__)


def _require_allowed_ip(req: Request):
    """Raise 403 if request is not from an allowed client IP.

    ALLOWED_CLIENT_IPS is populated from PI_HOST in config.py.  Add more
    IPs there when supporting multiple Pi clients.
    """
    client_ip = req.client.host
    if client_ip not in ALLOWED_CLIENT_IPS:
        logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden")


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter (no Redis required).

    Maintains a deque of request timestamps per IP.  On each check, expired
    timestamps (older than `window_seconds`) are pruned before counting.
    IPs idle for a full window are evicted to prevent unbounded dict growth
    (capped at 1000 tracked IPs — more than enough for a home network).

    Not suitable for distributed deployments (no shared state), but fine for
    a single-server home assistant.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: dict[str, deque] = {}
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Return True if the key is within its rate limit for this window.

        Side effect: records this request timestamp if allowed.
        """
        now = time.time()
        with self._lock:
            if key not in self._timestamps:
                self._timestamps[key] = deque()
            ts = self._timestamps[key]
            # Evict timestamps older than the window
            while ts and ts[0] < now - self.window:
                ts.popleft()
            if len(ts) >= self.max_requests:
                return False
            ts.append(now)
            # Evict idle IPs to prevent unbounded dict growth
            if len(self._timestamps) > 1000:
                self._timestamps = {
                    k: v for k, v in self._timestamps.items() if v
                }
            return True


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Create and configure the FastAPI application.

    The orchestrator is passed in (not constructed here) so tests can inject
    a mock orchestrator without starting real hardware.

    Args:
        orchestrator: Fully initialised Orchestrator instance.

    Returns:
        Configured FastAPI application ready for uvicorn.
    """

    app = FastAPI(
        title="Igor Voice Assistant Server",
        description="PC backend for processing voice interactions",
        version="1.0.0"
    )

    from server.config import PI_HOST
    from server.beeps import _DEFS as _BEEP_DEFS

    # CORS: only allow requests from the Pi's Flask server port
    allowed_origins = [
        f"http://{PI_HOST}:8080",
        f"http://{PI_HOST}",
        "http://localhost:8080",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.state.orchestrator = orchestrator
    app.state.start_time = time.time()

    # 10 voice requests per minute per IP — generous for a single Pi, blocks runaway loops.
    # Adjust max_requests/window_seconds here if adding multiple Pi clients.
    _rate_limiter = _RateLimiter(max_requests=10, window_seconds=60)

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logger.error(f"Validation error: {exc}")
        return JSONResponse(status_code=422, content={"error": "Validation failed"})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            raise exc
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    @app.post("/api/process_interaction", response_model=ProcessInteractionResponse)
    def process_interaction(request: ProcessInteractionRequest, req: Request):
        """Main voice pipeline endpoint.

        Validates IP, checks rate limit, decodes audio, hands off to orchestrator.
        The orchestrator runs STT → LLM → TTS and returns a response dict.
        Response is wrapped in ProcessInteractionResponse Pydantic model for
        type safety and automatic JSON serialisation.

        On error: 403 (IP blocked), 429 (rate limited), 400 (bad audio), 500 (pipeline).
        """
        _require_allowed_ip(req)
        client_ip = req.client.host

        if not _rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        try:
            # Sanitize wake_word before logging to prevent log injection
            safe_wake_word = request.wake_word.replace('\n', '').replace('\r', '')
            logger.info(f"Received interaction request (wake word: {safe_wake_word})")

            try:
                audio_bytes = decode_audio_base64(request.audio_base64)
            except Exception as e:
                logger.error(f"Failed to decode audio: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio encoding")

            result = orchestrator.process_interaction(
                audio_bytes, request.wake_word,
                prefer_sonos=request.prefer_sonos_output,
            )

            response = ProcessInteractionResponse(
                transcription=result['transcription'],
                response_text=result['response_text'],
                audio_base64=result['audio_base64'],
                commands_executed=result['commands_executed'],
                timings=result['timings'],
                await_followup=result.get('await_followup', False),
                tts_routed=result.get('tts_routed', False),
                tts_duration_seconds=result.get('tts_duration_seconds'),
                speaker=result.get('speaker'),
                error=result.get('error')
            )

            if result.get('error'):
                logger.warning(f"Interaction completed with error: {result['error']}")
            else:
                logger.info("Interaction completed successfully")

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing interaction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Processing failed")

    @app.get("/audio/beep/{beep_type}")
    async def get_beep_audio(beep_type: str):
        """Serve pre-generated beep WAV files (fetched by Sonos).

        Uses FileResponse (not in-memory Response) because Sonos sends
        "Range: bytes=0-" range requests before playing.  FileResponse
        handles range seeking automatically; an in-memory Response would
        silently reject the range header and Sonos would fail to play.

        Beep WAVs are written to disk by write_beep_files() at server startup.
        If a file is missing, it's regenerated on demand.
        """
        from server.config import DATA_DIR
        from server.beeps import get_beep_wav, _DEFS
        if beep_type not in _DEFS:
            raise HTTPException(status_code=404, detail="Unknown beep type")
        beep_path = DATA_DIR / f"beep_{beep_type}.wav"
        if not beep_path.exists():
            # Write on demand if missing (e.g. data/ was cleared)
            beep_path.write_bytes(get_beep_wav(beep_type))
        return FileResponse(str(beep_path), media_type="audio/wav")

    class _SonosBeepRequest(BaseModel):
        beep_type: str
        indicator_light: str | None = Field(None, max_length=100)

    # Valid beep types derived from _BEEP_DEFS at import time — stays in sync
    # automatically when new beep types are added to beeps.py.
    _VALID_BEEP_TYPES = set(_BEEP_DEFS.keys())

    @app.post("/api/sonos_beep")
    async def sonos_beep(request: _SonosBeepRequest, req: Request):
        """Trigger a beep on Sonos (called by Pi client when USE_SONOS_OUTPUT=True).

        The Pi client fires-and-forgets this endpoint for each start/end/error beep.
        We run play_sonos_beep() in the executor (non-blocking) so the FastAPI event
        loop is never stalled by soco's blocking play_uri() call.

        If TV is playing AND indicator_light is set in the request, the server
        flashes the LIFX light instead of playing audio through Sonos.
        """
        _require_allowed_ip(req)
        if request.beep_type not in _VALID_BEEP_TYPES:
            raise HTTPException(status_code=400, detail="Invalid beep type")
        import asyncio
        # run_in_executor dispatches to the default ThreadPoolExecutor so the
        # blocking soco call doesn't stall the async event loop
        asyncio.get_event_loop().run_in_executor(
            None, lambda: app.state.orchestrator.play_sonos_beep(
                request.beep_type, request.indicator_light
            )
        )
        return {"status": "ok"}

    @app.get("/audio/tts_latest")
    async def get_tts_audio(req: Request):
        """Serve the most recent TTS audio from memory (Sonos pull endpoint).

        After routing TTS to Sonos, the server stores the 44100 Hz WAV in memory
        at orchestrator._tts_audio.  Sonos then GETs this endpoint to fetch the
        audio.  No disk I/O — the WAV lives only in RAM.

        Range request handling:
          Sonos always sends "Range: bytes=0-" before playing.  Without proper
          206 Partial Content responses, Sonos silently fails and never plays.
          We parse the Range header and return the correct byte slice + headers.
        """
        audio = app.state.orchestrator.tts_audio
        if not audio:
            raise HTTPException(status_code=404, detail="No TTS audio available")

        size = len(audio)
        range_header = req.headers.get("range")
        if range_header and range_header.startswith("bytes="):
            # Parse "bytes=start-end" (end may be absent → serve to EOF)
            parts = range_header[6:].split("-")
            try:
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1
                if start < 0 or end < 0 or start > end:
                    raise ValueError("Invalid range bounds")
            except (ValueError, IndexError):
                start, end = 0, size - 1
            end = min(end, size - 1)
            chunk = audio[start:end + 1]
            from fastapi.responses import Response
            return Response(
                content=chunk,
                status_code=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(chunk)),
                    "Content-Type": "audio/wav",
                },
            )

        # No range header — serve full file
        from fastapi.responses import Response
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    @app.get("/api/health", response_model=HealthCheckResponse)
    async def health_check():
        """Liveness + service status check.

        Checks: Whisper model loaded, Pi reachable over HTTP.
        Claude and TTS are always listed as 'connected'/'ready' — there's no
        cheap pre-flight for them without making an actual API call.
        """
        try:
            uptime = time.time() - app.state.start_time
            services = {
                'whisper': 'loaded' if orchestrator.transcriber.model else 'not_loaded',
                'claude': 'connected',
                'kokoro': 'ready',
                'pi_client': 'ready'
            }
            if orchestrator.pi_client.check_health():
                services['pi'] = 'reachable'
            else:
                services['pi'] = 'unreachable'

            return HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services=services,
                uptime_seconds=uptime,
                additional_info={'conversation_messages': len(orchestrator.get_conversation_history())}
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status=HealthStatus.UNHEALTHY, services={}, uptime_seconds=0,
                additional_info={'error': 'Health check failed'}
            )

    @app.get("/api/conversation/history")
    async def get_conversation_history(req: Request):
        _require_allowed_ip(req)
        return {"history": orchestrator.get_conversation_history()}

    @app.post("/api/conversation/clear")
    async def clear_conversation_history(req: Request):
        _require_allowed_ip(req)
        orchestrator.clear_conversation_history()
        return {"status": "success", "message": "Conversation history cleared"}

    @app.get("/")
    async def root():
        return {
            "service": "Igor Voice Assistant Server",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "process": "/api/process_interaction",
                "history": "/api/conversation/history"
            }
        }

    return app
