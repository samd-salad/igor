#!/usr/bin/env python3
"""Main entry point for PC server."""
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from server.transcription import Transcriber
from server.llm import LLM
from server.synthesis import Synthesizer
from server.pi_callback import PiCallbackClient
from server.orchestrator import Orchestrator
from server.api import create_app
from server.event_loop import initialize_event_loop
import server.commands as commands
from server.config import (
    SERVER_HOST,
    SERVER_PORT,
    WHISPER_MODEL,
    CLAUDE_MODEL,
    CLAUDE_API_KEY,
    PIPER_VOICE,
    PI_HOST,
    PI_PORT
)
from shared.utils import setup_logging

# Configure logging
logger = setup_logging('server', level=logging.INFO)

if not CLAUDE_API_KEY:
    logger.error("ANTHROPIC_API_KEY environment variable is not set. Exiting.")
    sys.exit(1)


def initialize_services():
    """Initialize all services required by the server."""
    logger.info("Initializing Dr. Butts Voice Assistant Server...")

    # Initialize Transcriber
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    transcriber = Transcriber(WHISPER_MODEL)
    if not transcriber.initialize():
        logger.error("Failed to initialize Whisper")
        sys.exit(1)

    # Initialize LLM
    logger.info(f"Connecting to Claude API ({CLAUDE_MODEL})")
    llm = LLM()

    # Initialize Synthesizer
    logger.info(f"Loading Piper voice: {PIPER_VOICE}")
    try:
        synthesizer = Synthesizer(PIPER_VOICE)
    except ValueError as e:
        logger.error(f"Failed to initialize Piper: {e}")
        sys.exit(1)

    # Initialize Pi callback client
    pi_url = f"http://{PI_HOST}:{PI_PORT}"
    logger.info(f"Pi client URL: {pi_url}")
    pi_client = PiCallbackClient(pi_url)

    # Inject pi_client into hardware commands
    commands.inject_pi_client(pi_client)

    # Initialize Event Loop (for timers with Pi callbacks)
    logger.info("Starting event loop for timers")
    event_loop = initialize_event_loop(pi_client, synthesizer)

    # Initialize Orchestrator
    orchestrator = Orchestrator(
        transcriber=transcriber,
        llm=llm,
        synthesizer=synthesizer,
        pi_client=pi_client
    )

    logger.info("All services initialized successfully")
    return orchestrator


def main():
    """Main entry point."""
    try:
        # Initialize services
        orchestrator = initialize_services()

        # Create FastAPI app
        app = create_app(orchestrator)

        # Start server
        logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
        logger.info("Press Ctrl+C to stop")

        uvicorn.run(
            app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="info",
            access_log=True
        )

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean shutdown - stop event loop
        try:
            from server.event_loop import get_event_loop
            event_loop = get_event_loop()
            event_loop.stop()
            logger.info("Event loop stopped")
        except Exception as e:
            logger.debug(f"Event loop cleanup: {e}")


if __name__ == "__main__":
    main()
