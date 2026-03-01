"""FastAPI server with endpoints for voice interaction processing."""
import logging
import time
from collections import deque
from threading import Lock
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, ValidationError

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
    """Raise 403 if request is not from an allowed client IP."""
    client_ip = req.client.host
    if client_ip not in ALLOWED_CLIENT_IPS:
        logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden")


class _RateLimiter:
    """Simple in-memory rate limiter (no extra dependencies)."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: dict[str, deque] = {}
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            if key not in self._timestamps:
                self._timestamps[key] = deque()
            ts = self._timestamps[key]
            while ts and ts[0] < now - self.window:
                ts.popleft()
            if len(ts) >= self.max_requests:
                return False
            ts.append(now)
            return True


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Dr. Butts Voice Assistant Server",
        description="PC backend for processing voice interactions",
        version="1.0.0"
    )

    from server.config import PI_HOST
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

    # 10 voice requests per minute per IP — generous for a single Pi, blocks runaway loops
    _rate_limiter = _RateLimiter(max_requests=10, window_seconds=60)

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logger.error(f"Validation error: {exc}")
        return JSONResponse(status_code=422, content={"error": "Validation failed"})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    @app.post("/api/process_interaction", response_model=ProcessInteractionResponse)
    async def process_interaction(request: ProcessInteractionRequest, req: Request):
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

        Uses FileResponse so Sonos range requests (Range: bytes=0-) are
        handled correctly — in-memory Response silently rejects range seeks.
        """
        from server.config import DATA_DIR
        from server.beeps import get_beep_wav, _DEFS
        if beep_type not in _DEFS:
            raise HTTPException(status_code=404, detail="Unknown beep type")
        beep_path = DATA_DIR / f"beep_{beep_type}.wav"
        if not beep_path.exists():
            # Write on demand if missing
            beep_path.write_bytes(get_beep_wav(beep_type))
        return FileResponse(str(beep_path), media_type="audio/wav")

    class _SonosBeepRequest(BaseModel):
        beep_type: str

    _VALID_BEEP_TYPES = {"start", "end", "done", "error", "alert"}

    @app.post("/api/sonos_beep")
    async def sonos_beep(request: _SonosBeepRequest, req: Request):
        """Trigger a beep on Sonos (called by Pi client when USE_SONOS_OUTPUT=True).

        Runs play_sonos_beep in a background thread so the blocking soco
        play_uri call never stalls the FastAPI event loop.
        """
        _require_allowed_ip(req)
        if request.beep_type not in _VALID_BEEP_TYPES:
            raise HTTPException(status_code=400, detail="Invalid beep type")
        import asyncio
        asyncio.get_event_loop().run_in_executor(
            None, app.state.orchestrator.play_sonos_beep, request.beep_type
        )
        return {"status": "ok"}

    @app.get("/audio/tts_latest")
    async def get_tts_audio(req: Request):
        """Serve the most recent TTS audio from memory (Sonos pull endpoint).

        Handles Range requests so Sonos can seek/buffer correctly.
        No audio is written to disk.
        """
        audio = app.state.orchestrator.tts_audio
        if not audio:
            raise HTTPException(status_code=404, detail="No TTS audio available")

        size = len(audio)
        range_header = req.headers.get("range")
        if range_header and range_header.startswith("bytes="):
            parts = range_header[6:].split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1
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

        from fastapi.responses import Response
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    @app.get("/api/health", response_model=HealthCheckResponse)
    async def health_check():
        try:
            uptime = time.time() - app.state.start_time
            services = {
                'whisper': 'loaded' if orchestrator.transcriber.model else 'not_loaded',
                'claude': 'connected',
                'piper': 'ready',
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
            "service": "Dr. Butts Voice Assistant Server",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "process": "/api/process_interaction",
                "history": "/api/conversation/history"
            }
        }

    return app
