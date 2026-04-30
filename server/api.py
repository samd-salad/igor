"""FastAPI server for Igor text-in/text-out conversation agent.

Endpoints:
  POST /conversation/process — HA Custom Conversation Agent entry point
  GET  /api/health           — liveness + service status
  GET  /                     — service info

Auth / security:
  - /conversation/process requires `X-Igor-Token` header when IGOR_API_TOKEN env set.
  - Rate limiting: 10 conversation requests/minute per IP.
  - All inputs validated by Pydantic.
"""
import logging
import os
import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from server.context import InteractionContext
from server.rooms import RoomConfig

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseModel):
    status: HealthStatus
    services: dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float = Field(..., ge=0)
    additional_info: dict = Field(default_factory=dict)

    @field_validator('services', 'additional_info')
    @classmethod
    def _validate_dict_size(cls, v: dict) -> dict:
        if len(str(v)) > 10_000:
            raise ValueError("Dictionary too large")
        return v


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

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
                self._timestamps = {k: v for k, v in self._timestamps.items() if v}
            return True


def create_app(
    rooms: dict[str, RoomConfig] = None,
    conversation_service=None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    if rooms is None:
        from server.rooms import make_default_room
        default = make_default_room()
        rooms = {default.room_id: default}

    app = FastAPI(
        title="Igor Conversation Agent",
        description="Text-in/text-out backend for Home Assistant voice pipeline",
        version="3.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Igor-Token"],
    )

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

    # ---- HA Custom Conversation Agent ----

    class _ConversationProcessRequest(BaseModel):
        """Payload from HA's conversation pipeline."""
        text: str = Field(..., min_length=1, max_length=10000)
        conversation_id: Optional[str] = Field(None, max_length=100)
        device_id: Optional[str] = Field(None, max_length=100)
        language: Optional[str] = Field(None, max_length=20)

    def _build_ctx_from_device(device_id: Optional[str]) -> InteractionContext:
        """Resolve a HA device_id to an InteractionContext via area lookup."""
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
        the satellite should keep listening.
        """
        expected_token = os.environ.get("IGOR_API_TOKEN", "")
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

    # ---- Health ----

    @app.get("/api/health", response_model=HealthCheckResponse)
    async def health_check():
        try:
            uptime = time.time() - app.state.start_time
            services: dict[str, str] = {"claude": "connected"}
            if conversation_service is not None:
                services["conversation"] = "ready"
            return HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services=services,
                uptime_seconds=uptime,
                additional_info={"rooms": list(rooms.keys())},
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status=HealthStatus.UNHEALTHY, services={}, uptime_seconds=0,
                additional_info={"error": "Health check failed"},
            )

    @app.get("/")
    async def root():
        return {
            "service": "Igor Conversation Agent",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "process": "/conversation/process",
            },
        }

    return app
