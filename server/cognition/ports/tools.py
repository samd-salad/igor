"""ToolExecutorPort — execute a named tool with args, return its string result."""
from __future__ import annotations
from typing import Protocol
from server.cognition.contracts import VoiceTurn


class ToolExecutorPort(Protocol):
    def list_schemas(self) -> list[dict]: ...
    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str: ...
