"""LLMPort — minimal chat interface."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable

from server.cognition.contracts import ToolSchema


@dataclass(frozen=True)
class ChatResult:
    text: str
    commands_executed: list[str]
    input_tokens: int
    output_tokens: int


class LLMPort(Protocol):
    def chat(
        self,
        system_prompt: str,
        user_text: str,
        tool_schemas: list[ToolSchema],
        tool_executor: Callable[[str, dict], str],
        history: list[dict] | None = None,
    ) -> ChatResult: ...
