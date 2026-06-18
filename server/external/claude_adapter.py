"""Claude API adapter implementing cognition.ports.LLMPort.

This is the ONLY file in the project allowed to import anthropic."""
from __future__ import annotations
import logging
from typing import Callable, Optional

import anthropic  # boundary_check enforces this only-here import

from server.cognition.ports.llm import ChatResult

logger = logging.getLogger(__name__)


class ClaudeAdapter:
    def __init__(self, client: Optional["anthropic.Anthropic"] = None,
                 model: str = "claude-haiku-4-5-20251001",
                 max_tokens: int = 1024,
                 max_rounds: int = 3):
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._max_rounds = max_rounds

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        tool_schemas: list[dict],
        tool_executor: Callable[[str, dict], str],
        history: list[dict] | None = None,
    ) -> ChatResult:
        messages = list(history or []) + [{"role": "user", "content": user_text}]
        commands: list[str] = []
        in_tok = 0
        out_tok = 0
        last_text = ""

        for _ in range(self._max_rounds):
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tool_schemas or [],
            )
            in_tok += getattr(resp.usage, "input_tokens", 0)
            out_tok += getattr(resp.usage, "output_tokens", 0)

            tool_calls = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            text_blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
            if text_blocks:
                last_text = text_blocks[-1].text

            if not tool_calls:
                break

            assistant_content = list(resp.content)
            tool_results = []
            for tc in tool_calls:
                args = getattr(tc, "input", {}) or {}
                result = tool_executor(tc.name, args)
                commands.append(tc.name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return ChatResult(
            text=last_text, commands_executed=commands,
            input_tokens=in_tok, output_tokens=out_tok,
        )
