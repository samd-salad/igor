#!/usr/bin/env python3
"""Entry point for Igor — text-in/text-out conversation backend for HA.

Run:
    ANTHROPIC_API_KEY=...  HA_TOKEN=...  python -m server.main_text
"""
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from server.api import create_app
from server.brain import init_brain
from server.commands import inject_dependencies
from server.config import (
    BRAIN_FILE,
    CLAUDE_API_KEY,
    DATA_DIR,
    SERVER_HOST,
    SERVER_PORT,
)
from server.conversation import ConversationService
from server.ha_client import HAError, get_client
from server.llm import LLM
from server.rooms import RoomConfig, load_rooms


def _setup_logging() -> logging.Logger:
    """Configure root + server logger with a rotating file handler."""
    log_file = DATA_DIR / "server.log"
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        root.addHandler(sh)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(str(log_file), maxBytes=10*1024*1024,
                                  backupCount=3, encoding='utf-8')
        fh.setFormatter(formatter)
        root.addHandler(fh)
    return logging.getLogger("server")


logger = _setup_logging()


def _build_rooms_from_ha() -> dict[str, RoomConfig]:
    """Synthesize one RoomConfig per HA area."""
    try:
        areas = get_client().get_areas()
    except HAError as e:
        logger.warning(f"Could not enumerate HA areas: {e}; falling back to single 'default' room")
        return {"default": RoomConfig(room_id="default", display_name="Default")}
    if not areas:
        return {"default": RoomConfig(room_id="default", display_name="Default")}
    rooms: dict[str, RoomConfig] = {}
    for area in areas:
        room_id = area.lower().replace(" ", "_")
        rooms[room_id] = RoomConfig(room_id=room_id, display_name=area, ha_area=area)
    logger.info(f"Synthesized {len(rooms)} room(s) from HA areas: {list(rooms.keys())}")
    return rooms


def main() -> None:
    if not CLAUDE_API_KEY:
        logger.error("ANTHROPIC_API_KEY environment variable is not set")
        sys.exit(1)
    if not os.environ.get("HA_TOKEN"):
        logger.warning("HA_TOKEN not set — Home Assistant calls will fail")

    brain = init_brain(BRAIN_FILE)
    brain.migrate_legacy_files(DATA_DIR)

    rooms_yaml = DATA_DIR / "rooms.yaml"
    rooms = load_rooms(rooms_yaml) if rooms_yaml.exists() else _build_rooms_from_ha()

    inject_dependencies()

    llm = LLM()
    conversation_service = ConversationService(llm)

    if brain.should_consolidate():
        import threading
        threading.Thread(
            target=conversation_service._run_consolidation,
            daemon=True, name="InitialConsolidation",
        ).start()
        logger.info("Initial memory consolidation triggered")

    app = create_app(rooms=rooms, conversation_service=conversation_service)

    logger.info(f"Starting Igor on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
