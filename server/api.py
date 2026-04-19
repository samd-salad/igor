"""FastAPI server with endpoints for voice interaction processing.

Endpoints:
  POST /api/process_interaction  — main voice pipeline (STT→LLM→TTS)
  POST /api/text_interaction     — text-only pipeline (no STT/TTS)
  POST /api/register             — dynamic client registration
  GET  /audio/beep/{type}        — serve pre-generated beep WAVs for Sonos
  POST /api/sonos_beep           — trigger Sonos beep (called by Pi client)
  GET  /audio/tts/{room_id}      — serve TTS audio for a specific room
  GET  /audio/tts_latest         — serve TTS audio (legacy, uses default room)
  GET  /api/health               — liveness + service status
  GET  /api/conversation/history — inspect current LLM history
  POST /api/conversation/clear   — reset LLM history
  GET  /                         — service info

Security:
  - IP allowlist: _require_allowed_ip() checks ClientRegistry + legacy ALLOWED_CLIENT_IPS.
  - Rate limiting: 10 voice requests/minute per IP.
  - Audio and wake_word inputs validated by Pydantic before reaching orchestrator.

Sonos range request support:
  /audio/tts/* and /audio/beep/* support HTTP Range requests because Sonos
  sends "Range: bytes=0-" to seek before playing.
"""
import logging
import time
from collections import deque
from threading import Lock
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from shared.models import (
    ProcessInteractionRequest,
    ProcessInteractionResponse,
    TextInteractionRequest,
    TextInteractionResponse,
    ClientRegistrationRequest,
    HealthCheckResponse,
    Status,
    HealthStatus
)
from shared.utils import decode_audio_base64
from server.orchestrator import Orchestrator
from server.context import InteractionContext
from server.client_registry import ClientRegistry
from server.room_state import RoomStateManager
from server.rooms import RoomConfig
from server.config import ALLOWED_CLIENT_IPS, AUDIO_TOKEN

logger = logging.getLogger(__name__)


def _require_audio_token(req: Request):
    """Validate audio endpoint access via token query parameter."""
    token = req.query_params.get("token")
    if token != AUDIO_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid audio token")


def _require_allowed_ip(req: Request, registry: ClientRegistry):
    """Raise 403 if request is not from an allowed client IP.

    Checks both the dynamic ClientRegistry and legacy ALLOWED_CLIENT_IPS.
    """
    client_ip = req.client.host
    if client_ip in ALLOWED_CLIENT_IPS:
        return
    if registry.is_allowed_ip(client_ip):
        return
    logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
    raise HTTPException(status_code=403, detail="Forbidden")


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter (no Redis required)."""

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
            if len(self._timestamps) > 1000:
                self._timestamps = {
                    k: v for k, v in self._timestamps.items() if v
                }
            return True


def _build_context(
    client_id: str,
    room_id: str,
    rooms: dict[str, RoomConfig],
    room_state_mgr: RoomStateManager,
    registry: ClientRegistry,
    prefer_sonos: bool = False,
    client_type: str = "audio",
) -> InteractionContext:
    """Build an InteractionContext from request fields."""
    # Look up room config, fall back to first room
    room = rooms.get(room_id)
    if room is None:
        room = next(iter(rooms.values())) if rooms else None
    if room is None:
        from server.rooms import make_default_room
        room = make_default_room()

    # Snapshot TV state from room state
    rs = room_state_mgr.get(room.room_id if room else "default")
    tv_state = rs.tv_state if rs else "unknown"

    # Look up callback URL from registry
    client = registry.get(client_id)
    callback_url = client.callback_url if client else None

    return InteractionContext(
        client_id=client_id,
        room=room,
        client_type=client_type,
        callback_url=callback_url,
        prefer_sonos=prefer_sonos,
        tv_state=tv_state,
    )


def _serve_tts_audio(audio: bytes, req: Request) -> Response:
    """Serve TTS audio with Range request support for Sonos."""
    if not audio:
        raise HTTPException(status_code=404, detail="No TTS audio available")

    size = len(audio)
    range_header = req.headers.get("range")
    if range_header and range_header.startswith("bytes="):
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

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
    )


def create_app(
    orchestrator: Orchestrator = None,
    registry: ClientRegistry = None,
    room_state_mgr: RoomStateManager = None,
    rooms: dict[str, RoomConfig] = None,
    conversation_service=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        orchestrator: Legacy Orchestrator (audio pipeline). Optional during
            migration — being deleted in favor of conversation_service.
        registry: ClientRegistry for dynamic client management.
        room_state_mgr: RoomStateManager for per-room state.
        rooms: Dict of room configs keyed by room_id.
        conversation_service: ConversationService for HA-driven text-only
            conversation (the new entry point at POST /conversation/process).
    """
    # Defaults for backward compat (single-client mode)
    if registry is None:
        from server.config import TRUSTED_IPS
        registry = ClientRegistry(trusted_ips=TRUSTED_IPS)
    if rooms is None:
        from server.rooms import make_default_room
        default = make_default_room()
        rooms = {default.room_id: default}
    if room_state_mgr is None:
        room_state_mgr = RoomStateManager(rooms)

    app = FastAPI(
        title="Igor Voice Assistant Server",
        description="PC backend for processing voice interactions",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    from server.config import PI_HOST
    from server.beeps import _DEFS as _BEEP_DEFS

    allowed_origins = [
        f"http://{PI_HOST}:8080",
        f"http://{PI_HOST}",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )

    app.state.orchestrator = orchestrator
    app.state.registry = registry
    app.state.room_state_mgr = room_state_mgr
    app.state.rooms = rooms
    app.state.conversation_service = conversation_service
    app.state.start_time = time.time()

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

    # ---- Client Registration ----

    @app.post("/api/register")
    def register_client(request: ClientRegistrationRequest, req: Request):
        """Register a client with the server for dynamic routing.

        Restricted to ALLOWED_CLIENT_IPS and TRUSTED_IPS to prevent
        unauthenticated self-enrollment into the IP allowlist.
        """
        _require_allowed_ip(req, registry)
        client_ip = req.client.host
        registry.register(
            client_id=request.client_id,
            room_id=request.room_id,
            client_type=request.client_type,
            callback_url=request.callback_url,
            ip=client_ip,
        )
        return {"status": "ok", "client_id": request.client_id}

    # ---- Voice Interaction ----

    @app.post("/api/process_interaction", response_model=ProcessInteractionResponse)
    def process_interaction(request: ProcessInteractionRequest, req: Request):
        """Main voice pipeline endpoint."""
        _require_allowed_ip(req, registry)
        client_ip = req.client.host

        if not _rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        try:
            safe_wake_word = request.wake_word.replace('\n', '').replace('\r', '')
            logger.info(f"Received interaction request (wake word: {safe_wake_word})")

            try:
                audio_bytes = decode_audio_base64(request.audio_base64)
            except Exception as e:
                logger.error(f"Failed to decode audio: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio encoding")

            ctx = _build_context(
                client_id=request.client_id,
                room_id=request.room_id,
                rooms=rooms,
                room_state_mgr=room_state_mgr,
                registry=registry,
                prefer_sonos=request.prefer_sonos_output,
                client_type="audio",
            )

            result = orchestrator.process_interaction(audio_bytes, request.wake_word, ctx=ctx)

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

    # ---- Text Interaction ----

    @app.post("/api/text_interaction", response_model=TextInteractionResponse)
    def text_interaction(request: TextInteractionRequest, req: Request):
        """Text-only pipeline endpoint (no STT/TTS)."""
        _require_allowed_ip(req, registry)

        if not _rate_limiter.is_allowed(req.client.host):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        try:
            ctx = _build_context(
                client_id=request.client_id,
                room_id=request.room_id,
                rooms=rooms,
                room_state_mgr=room_state_mgr,
                registry=registry,
                prefer_sonos=False,
                client_type="text",
            )

            result = orchestrator.process_text_interaction(request.text, ctx=ctx)

            return TextInteractionResponse(
                response_text=result['response_text'],
                commands_executed=result.get('commands_executed', []),
                await_followup=result.get('await_followup', False),
                error=result.get('error'),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing text interaction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Processing failed")

    # ---- HA Custom Conversation Agent ----

    class _ConversationProcessRequest(BaseModel):
        """Payload from HA's conversation pipeline. Field names follow HA's
        async_process(conversation.ConversationInput) shape."""
        text: str = Field(..., min_length=1, max_length=2000)
        conversation_id: Optional[str] = Field(None, max_length=100)
        device_id: Optional[str] = Field(None, max_length=100)
        language: Optional[str] = Field(None, max_length=20)

    def _build_ctx_from_device(device_id: Optional[str]) -> InteractionContext:
        """Resolve a HA device_id to an InteractionContext.

        device_id → HA area → matching RoomConfig (by ha_area), or a synthesized
        room with just ha_area set. Returns the default room if nothing maps.
        """
        from server.rooms import RoomConfig
        ha_area = ""
        if device_id:
            try:
                from server.ha_client import get_client
                ha_area = get_client().area_of_device(device_id)
            except Exception as e:
                logger.warning(f"Failed to resolve device_id {device_id}: {e}")

        if ha_area:
            for room in rooms.values():
                if (room.ha_area or "").lower() == ha_area.lower():
                    return InteractionContext(
                        client_id=device_id or "ha", room=room, client_type="text",
                        callback_url=None, prefer_sonos=False, tv_state="unknown",
                    )
            synthesized = RoomConfig(
                room_id=ha_area.lower().replace(" ", "_"),
                display_name=ha_area, ha_area=ha_area,
            )
            return InteractionContext(
                client_id=device_id or "ha", room=synthesized, client_type="text",
                callback_url=None, prefer_sonos=False, tv_state="unknown",
            )

        default = next(iter(rooms.values()))
        return InteractionContext(
            client_id=device_id or "ha", room=default, client_type="text",
            callback_url=None, prefer_sonos=False, tv_state="unknown",
        )

    @app.post("/conversation/process")
    async def conversation_process(request: _ConversationProcessRequest, req: Request):
        """HA Custom Conversation Agent endpoint.

        HA's voice pipeline POSTs transcribed user speech here; we run it
        through Igor's brain and return response text + a flag for whether
        the satellite should keep listening (open the mic for follow-up).

        Auth: when IGOR_API_TOKEN env var is set, requests must include
        a matching `X-Igor-Token` header. Unset = no auth (dev mode).
        """
        import os as _os
        expected_token = _os.environ.get("IGOR_API_TOKEN", "")
        if expected_token:
            provided = req.headers.get("X-Igor-Token", "")
            if provided != expected_token:
                logger.warning(f"Conversation request from {req.client.host} with bad/missing token")
                raise HTTPException(status_code=401, detail="Invalid or missing X-Igor-Token")
        if conversation_service is None:
            raise HTTPException(status_code=503, detail="Conversation service not configured")
        if not _rate_limiter.is_allowed(req.client.host):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        try:
            ctx = _build_ctx_from_device(request.device_id)
            result = conversation_service.process(request.text, ctx=ctx)
            return {
                "response": result["response"],
                "conversation_id": request.conversation_id or f"igor-{int(time.time() * 1000)}",
                "end_conversation": result.get("end_conversation", True),
                "commands_executed": result.get("commands_executed", []),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Conversation processing failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Conversation processing failed")

    # ---- Audio Serving ----

    @app.get("/audio/beep/{beep_type}")
    async def get_beep_audio(beep_type: str, req: Request):
        """Serve pre-generated beep WAV files (fetched by Sonos)."""
        _require_audio_token(req)
        from server.config import DATA_DIR
        from server.beeps import get_beep_wav, _DEFS
        if beep_type not in _DEFS:
            raise HTTPException(status_code=404, detail="Unknown beep type")
        beep_path = DATA_DIR / f"beep_{beep_type}.wav"
        if not beep_path.exists():
            beep_path.write_bytes(get_beep_wav(beep_type))
        return FileResponse(str(beep_path), media_type="audio/wav")

    class _SonosBeepRequest(BaseModel):
        beep_type: str
        indicator_light: str | None = Field(None, max_length=100)

    _VALID_BEEP_TYPES = set(_BEEP_DEFS.keys())

    @app.post("/api/sonos_beep")
    async def sonos_beep(request: _SonosBeepRequest, req: Request):
        """Trigger a beep on Sonos (called by Pi client when USE_SONOS_OUTPUT=True)."""
        _require_allowed_ip(req, registry)
        if request.beep_type not in _VALID_BEEP_TYPES:
            raise HTTPException(status_code=400, detail="Invalid beep type")
        import asyncio
        asyncio.get_event_loop().run_in_executor(
            None, lambda: app.state.orchestrator.play_sonos_beep(
                request.beep_type, request.indicator_light
            )
        )
        return {"status": "ok"}

    @app.get("/audio/tts/{room_id}")
    async def get_tts_audio_by_room(room_id: str, req: Request):
        """Serve TTS audio for a specific room (Sonos pull endpoint).

        Token-protected: Sonos gets the token embedded in the play_uri URL.
        Audio is ephemeral (overwritten each interaction) and only available briefly.
        """
        _require_audio_token(req)
        rs = room_state_mgr.get(room_id)
        audio = rs.tts_audio if rs else b""
        return _serve_tts_audio(audio, req)

    @app.get("/audio/tts_latest")
    async def get_tts_audio(req: Request):
        """Serve the most recent TTS audio (legacy endpoint, uses orchestrator buffer)."""
        _require_audio_token(req)
        audio = app.state.orchestrator.tts_audio
        return _serve_tts_audio(audio, req)

    # ---- Admin / Health ----

    @app.get("/api/health", response_model=HealthCheckResponse)
    async def health_check():
        """Liveness + service status check."""
        try:
            uptime = time.time() - app.state.start_time
            services: dict[str, str] = {}
            # Audio path (legacy) — only present when orchestrator is configured
            if orchestrator is not None:
                services['whisper'] = 'loaded' if orchestrator.transcriber.model else 'not_loaded'
                services['kokoro'] = 'ready'
                if orchestrator.pi_client and not registry.list_all():
                    services['pi'] = 'reachable' if orchestrator.pi_client.check_health() else 'unreachable'
            # Conversation path (new)
            if conversation_service is not None:
                services['conversation'] = 'ready'
            services['claude'] = 'connected'
            # Per-client Pi health (audio clients only)
            for client in registry.list_all():
                if client.client_type == "audio" and client.callback_url:
                    try:
                        import requests as http_requests
                        resp = http_requests.get(f"{client.callback_url}/api/health", timeout=2.0)
                        services[f'pi_{client.client_id}'] = 'reachable' if resp.ok else 'unreachable'
                    except Exception:
                        services[f'pi_{client.client_id}'] = 'unreachable'

            return HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services=services,
                uptime_seconds=uptime,
                additional_info={
                    'rooms': list(rooms.keys()),
                    'clients': len(registry.list_all()),
                }
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status=HealthStatus.UNHEALTHY, services={}, uptime_seconds=0,
                additional_info={'error': 'Health check failed'}
            )

    @app.get("/api/conversation/history")
    async def get_conversation_history(req: Request):
        _require_allowed_ip(req, registry)
        return {"history": orchestrator.get_conversation_history()}

    @app.post("/api/conversation/clear")
    async def clear_conversation_history(req: Request):
        _require_allowed_ip(req, registry)
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
                "text": "/api/text_interaction",
                "register": "/api/register",
                "history": "/api/conversation/history"
            }
        }

    return app
