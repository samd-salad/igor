"""FastAPI server with endpoints for voice interaction processing."""
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from shared.models import (
    ProcessInteractionRequest,
    ProcessInteractionResponse,
    HealthCheckResponse,
    Status,
    HealthStatus
)
from shared.utils import decode_audio_base64
from server.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Dr. Butts Voice Assistant Server",
        description="PC backend for processing voice interactions",
        version="1.0.0"
    )

    # CORS middleware (allow Pi to make requests)
    # Restrict to Pi's IP for security
    from server.config import PI_HOST
    allowed_origins = [
        f"http://{PI_HOST}:8080",
        f"http://{PI_HOST}",
        "http://192.168.0.3:8080",  # Fallback to hardcoded Pi IP
        "http://localhost:8080",  # For local testing
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # Store orchestrator for request handlers
    app.state.orchestrator = orchestrator
    app.state.start_time = time.time()

    # Exception handler for validation errors
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logger.error(f"Validation error: {exc}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "details": exc.errors()
            }
        )

    # Exception handler for generic errors
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

    @app.post("/api/process_interaction", response_model=ProcessInteractionResponse)
    async def process_interaction(request: ProcessInteractionRequest):
        """
        Main endpoint: Process voice interaction from Pi.

        Receives audio from Pi, runs STT -> LLM -> TTS pipeline, returns response.
        """
        try:
            logger.info(f"Received interaction request (wake word: {request.wake_word})")

            # Decode audio from base64
            try:
                audio_bytes = decode_audio_base64(request.audio_base64)
            except Exception as e:
                logger.error(f"Failed to decode audio: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio encoding")

            # Process through orchestrator
            result = orchestrator.process_interaction(audio_bytes, request.wake_word)

            # Build response
            response = ProcessInteractionResponse(
                transcription=result['transcription'],
                response_text=result['response_text'],
                audio_base64=result['audio_base64'],
                commands_executed=result['commands_executed'],
                timings=result['timings'],
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

    @app.get("/api/health", response_model=HealthCheckResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            uptime = time.time() - app.state.start_time

            # Check service availability
            services = {
                'whisper': 'loaded' if orchestrator.transcriber.model else 'not_loaded',
                'ollama': 'connected',  # Could add actual connectivity check
                'piper': 'ready',
                'pi_client': 'ready'
            }

            # Check if Pi is reachable
            if orchestrator.pi_client.check_health():
                services['pi'] = 'reachable'
            else:
                services['pi'] = 'unreachable'

            return HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services=services,
                uptime_seconds=uptime,
                additional_info={
                    'conversation_messages': len(orchestrator.get_conversation_history())
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status=HealthStatus.UNHEALTHY,
                services={},
                uptime_seconds=0,
                additional_info={'error': str(e)}
            )

    @app.get("/api/conversation/history")
    async def get_conversation_history():
        """Get current conversation history."""
        return {
            "history": orchestrator.get_conversation_history()
        }

    @app.post("/api/conversation/clear")
    async def clear_conversation_history():
        """Clear conversation history."""
        orchestrator.clear_conversation_history()
        return {"status": "success", "message": "Conversation history cleared"}

    @app.get("/")
    async def root():
        """Root endpoint."""
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
