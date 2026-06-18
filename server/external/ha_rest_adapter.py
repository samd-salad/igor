"""ToolExecutorPort implementation. Wraps the existing `server/commands/`
registry; the underlying commands talk to HA via _internal/ha_client."""
from __future__ import annotations
import inspect
import logging
from typing import Mapping

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


def _discover_default_commands() -> dict:
    """Lazy import of the existing server.commands registry."""
    try:
        from server.commands import get_all_commands  # type: ignore
        return get_all_commands()
    except Exception as e:
        logger.warning("No commands registry available: %s", e)
        return {}


class HARestToolExecutor:
    """Adapts the current server.commands registry to ToolExecutorPort."""

    def __init__(self, commands: Mapping[str, object] | None = None):
        self._commands = dict(commands) if commands is not None else _discover_default_commands()

    def list_schemas(self) -> list[dict]:
        out = []
        for cmd in self._commands.values():
            if hasattr(cmd, "to_tool"):
                out.append(cmd.to_tool())
        return out

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        cmd = self._commands.get(name)
        if cmd is None:
            return f"Unknown command: {name}"
        kwargs = dict(args)
        try:
            sig = inspect.signature(cmd.execute)
            if "_ctx" in sig.parameters:
                kwargs["_ctx"] = _legacy_ctx_from_turn(turn)
        except (ValueError, TypeError):
            pass
        try:
            return cmd.execute(**kwargs)
        except Exception as e:
            logger.exception("Command %s failed", name)
            return f"Error executing {name}: {e}"


def _legacy_ctx_from_turn(turn: VoiceTurn):
    """Minimal adapter for legacy commands that accept InteractionContext.
    Removed in Task 33 when commands/ is decomposed."""
    from server.context import InteractionContext  # type: ignore
    from server.rooms import RoomConfig as LegacyRoom  # type: ignore
    legacy_room = LegacyRoom(
        room_id=turn.room.room_id,
        display_name=turn.room.display_name,
        ha_area=turn.room.ha_area,
    )
    return InteractionContext(
        client_id=turn.device_id or "ha",
        room=legacy_room,
        client_type="text",
        callback_url=None,
        prefer_sonos=False,
        tv_state="unknown",
    )
