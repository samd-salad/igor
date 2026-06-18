"""CompositeToolExecutor — Conversation's single ToolExecutorPort.
Routes tool execution by name to the first executor that claims it; lists
the union of every executor's schemas in registration order."""
from __future__ import annotations
from typing import Protocol

from server.cognition.contracts import VoiceTurn


class _ChildExecutor(Protocol):
    def list_schemas(self) -> list[dict]: ...
    def handles(self, name: str) -> bool: ...
    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str: ...


class CompositeToolExecutor:
    def __init__(self, *executors: _ChildExecutor):
        self._executors = list(executors)

    def list_schemas(self) -> list[dict]:
        out: list[dict] = []
        for ex in self._executors:
            out.extend(ex.list_schemas())
        return out

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        for ex in self._executors:
            if ex.handles(name):
                return ex.execute(name, args, turn)
        return f"Unknown tool: {name}"
