"""FastAPI app for Igor's HA Custom Conversation Agent."""
from __future__ import annotations
import logging
import time
from typing import Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.cognition.contracts import RoomConfig
from server.ha_io.contracts import ConversationRequest
from server.ha_io._internal.auth import check_token
from server.ha_io._internal.rate_limit import RateLimiter
from server.ha_io._internal.result_mapper import map_result
from server.ha_io._internal.voice_turn import build_voice_turn

logger = logging.getLogger(__name__)


def build_app(
    *,
    conversation,
    ha_client,
    known_rooms: Mapping[str, RoomConfig],
) -> FastAPI:
    app = FastAPI(
        title="Igor Conversation Agent",
        version="4.0.0",
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Igor-Token"],
    )
    app.state.started = time.time()
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

    @app.exception_handler(ValidationError)
    async def vexc(request, exc):
        logger.error("Validation error: %s", exc)
        return JSONResponse(status_code=422, content={"error": "Validation failed"})

    @app.exception_handler(Exception)
    async def gexc(request, exc):
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("Unhandled")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - app.state.started,
            "rooms": list(known_rooms.keys()),
        }

    @app.get("/")
    async def root():
        return {"service": "Igor", "status": "running"}

    @app.post("/conversation/process")
    async def conversation_process(req_model: ConversationRequest, req: Request):
        if not check_token(req.headers.get("X-Igor-Token")):
            raise HTTPException(status_code=401, detail="Invalid or missing token")
        if not rate_limiter.is_allowed(req.client.host):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        turn = build_voice_turn(req_model, ha_client, known_rooms)
        try:
            result = conversation.process(turn)
        except Exception:
            logger.exception("Conversation failed")
            raise HTTPException(status_code=500, detail="Processing failed")
        return map_result(result, req_model.conversation_id).model_dump()

    return app
