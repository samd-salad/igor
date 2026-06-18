"""ToolExecutorPort implementation.

The legacy `server/commands/` registry was deleted in Task 33; a follow-up
plan re-introduces HA-backed tools that take VoiceTurn directly. Today this
adapter starts with an empty registry — Igor responds via the LLM alone."""
from __future__ import annotations
import logging
from typing import Mapping

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


class HARestToolExecutor:
    """Adapts a tool registry to ToolExecutorPort. Empty registry by default."""

    def __init__(self, commands: Mapping[str, object] | None = None):
        self._commands = dict(commands) if commands is not None else {}

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
        try:
            return cmd.execute(**args)
        except Exception as e:
            logger.exception("Command %s failed", name)
            return f"Error executing {name}: {e}"
