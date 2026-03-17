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
from server.rooms import load_rooms
from server.client_registry import ClientRegistry
from server.room_state import RoomStateManager
import server.commands as commands
from server.beeps import write_beep_files
from server.brain import init_brain
from server.config import (
    SERVER_HOST,
    SERVER_PORT,
    WHISPER_MODEL,
    CLAUDE_MODEL,
    CLAUDE_API_KEY,
    KOKORO_VOICE,
    PI_HOST,
    PI_PORT,
    DATA_DIR,
    TRUSTED_IPS,
    BRAIN_FILE,
)
from shared.utils import setup_logging

# Configure logging — console + file (file enables MCP tail_logs + AI auditing)
logger = setup_logging('server', level=logging.INFO,
                       log_file=str(DATA_DIR / 'server.log'))

if not CLAUDE_API_KEY:
    logger.error("ANTHROPIC_API_KEY environment variable is not set. Exiting.")
    sys.exit(1)


def initialize_services():
    """Initialize all services required by the server."""
    logger.info("Initializing Igor Voice Assistant Server...")

    # Initialize unified brain store
    brain = init_brain(BRAIN_FILE)
    brain.migrate_legacy_files(DATA_DIR)

    # Load room configuration
    rooms_yaml = DATA_DIR / "rooms.yaml"
    rooms = load_rooms(rooms_yaml)
    logger.info(f"Rooms: {', '.join(rooms.keys())}")

    # Create client registry and room state manager
    registry = ClientRegistry(trusted_ips=TRUSTED_IPS)
    room_state_mgr = RoomStateManager(rooms)

    # Initialize Transcriber
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    transcriber = Transcriber(WHISPER_MODEL)
    if not transcriber.initialize():
        logger.error("Failed to initialize Whisper")
        sys.exit(1)

    # Initialize LLM
    logger.info(f"Connecting to Claude API ({CLAUDE_MODEL})")
    llm = LLM()

    # Initialize Synthesizer — fully lazy. Kokoro model loads on first
    # synthesis request (~2s one-time cost). Pre-caching also deferred.
    synthesizer = Synthesizer()

    # Initialize Pi callback client (legacy singleton)
    pi_url = f"http://{PI_HOST}:{PI_PORT}"
    logger.info(f"Pi client URL: {pi_url}")
    pi_client = PiCallbackClient(pi_url)

    # Auto-register the legacy Pi client
    registry.register(
        client_id="default",
        room_id="default",
        client_type="audio",
        callback_url=pi_url,
        ip=PI_HOST,
    )

    # Inject dependencies into commands
    commands.inject_dependencies(
        registry=registry,
        room_state_mgr=room_state_mgr,
        pi_client=pi_client,
    )

    # Initialize Event Loop (for timers with Pi callbacks)
    logger.info("Starting event loop for timers")
    event_loop = initialize_event_loop(pi_client, synthesizer)
    event_loop.set_client_registry(registry)
    event_loop.load_pending_reminders()

    # Initialize Orchestrator
    orchestrator = Orchestrator(
        transcriber=transcriber,
        llm=llm,
        synthesizer=synthesizer,
        pi_client=pi_client,
        room_state_mgr=room_state_mgr,
    )

    # Start per-room TV pollers
    room_state_mgr.start_tv_pollers()

    # Wire Sonos TTS routing into event loop for timer alerts
    from server.config import SONOS_TTS_OUTPUT
    if SONOS_TTS_OUTPUT:
        event_loop.set_sonos_tts_func(orchestrator.route_tts_to_sonos)

    # Pre-generate beep WAV files for Sonos output
    write_beep_files()

    # Trigger initial memory consolidation if needed (background, non-blocking).
    # Generates the identity narrative on first startup with existing memories.
    if brain.should_consolidate():
        import threading
        threading.Thread(
            target=orchestrator._run_consolidation,
            daemon=True,
            name="InitialConsolidation",
        ).start()
        logger.info("Initial memory consolidation triggered")

    logger.info("All services initialized successfully")
    return orchestrator, registry, room_state_mgr, rooms


def main():
    """Main entry point."""
    try:
        # Initialize services
        orchestrator, registry, room_state_mgr, rooms = initialize_services()

        # Create FastAPI app
        app = create_app(
            orchestrator=orchestrator,
            registry=registry,
            room_state_mgr=room_state_mgr,
            rooms=rooms,
        )

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
