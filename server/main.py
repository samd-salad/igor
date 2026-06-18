"""Composition root. The ONLY place adapters meet ports."""
from __future__ import annotations
import logging
import os
from pathlib import Path

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import RoomConfig
from server.cognition.services.conversation import Conversation
from server.cognition.services.consolidator import Consolidator
from server.cognition.services.session_summarizer import SessionSummarizer
from server.external import get_client as get_ha_client
from server.external._internal.async_runner import AsyncRunner
from server.external._internal.brain_json_migration import migrate_brain_json_if_needed
from server.external.claude_adapter import ClaudeAdapter
from server.external.composite_executor import CompositeToolExecutor
from server.external.ha_mcp_executor import HAMCPToolExecutor
from server.external.igor_native_tools import build_native_registry
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval
from server.external.system_clock import SystemClock
from server.external.weather_open_meteo import OpenMeteoWeather
from server.ha_io.api import build_app

logger = logging.getLogger(__name__)


def _brain_dir() -> Path:
    return Path(os.environ.get("BRAIN_DIR", "/app/data"))


def _build_rooms_from_ha() -> dict[str, RoomConfig]:
    try:
        areas = get_ha_client().get_areas()
    except Exception as e:
        logger.warning("Could not enumerate HA areas: %s", e)
        return {"default": RoomConfig(room_id="default", display_name="Default")}
    rooms: dict[str, RoomConfig] = {}
    for area in areas or []:
        rid = area.lower().replace(" ", "_")
        rooms[rid] = RoomConfig(room_id=rid, display_name=area, ha_area=area)
    return rooms or {"default": RoomConfig(room_id="default", display_name="Default")}


def build():
    """Build the FastAPI app with all adapters wired up."""
    brain_dir = _brain_dir()
    brain_dir.mkdir(parents=True, exist_ok=True)

    migrate_brain_json_if_needed(brain_dir / "brain.json", brain_dir / "brain.db")

    persistence = SqlitePersistence(brain_dir / "brain.db")
    retrieval = TagRetrieval(persistence)
    llm = ClaudeAdapter()
    clock = SystemClock()

    memory = MemoryStore(persistence)
    episodes = EpisodeStore(persistence)
    identity = IdentityStore(persistence)
    user_state = UserState(persistence)

    ha_url = os.environ.get("HA_URL", "http://10.0.40.5:8123").rstrip("/")
    ha_token = os.environ.get("HA_TOKEN", "")
    mcp_url = f"{ha_url}/api/mcp"
    default_location = os.environ.get("DEFAULT_LOCATION", "Arlington, VA")

    async_runner = AsyncRunner()
    weather = OpenMeteoWeather()
    native_tools = build_native_registry(
        memory=memory, user_state=user_state,
        weather=weather, default_location=default_location,
    )
    ha_tools = HAMCPToolExecutor(mcp_url, ha_token, async_runner)
    tools = CompositeToolExecutor(native_tools, ha_tools)

    summarizer = SessionSummarizer(episodes=episodes, memory=memory,
                                   llm=llm, clock=clock)
    summarizer.start()
    conversation = Conversation(
        memory=memory, episodes=episodes, identity=identity, user_state=user_state,
        retrieval=retrieval, llm=llm, tools=tools, clock=clock,
        summarizer=summarizer,
    )
    consolidator = Consolidator(
        memory=memory, episodes=episodes, identity=identity,
        llm=llm, clock=clock,
    )
    consolidator.start()

    rooms = _build_rooms_from_ha()
    return build_app(
        conversation=conversation,
        ha_client=get_ha_client(),
        known_rooms=rooms,
    )


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    app = build()
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
