"""LLM integration with Claude API for conversational AI."""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, Tuple
import anthropic

from datetime import datetime
from server.config import CLAUDE_API_KEY, CLAUDE_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_LLM_CALL_TIMEOUT = 45.0  # hard wall-clock timeout per API call (seconds)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_api")

# Respond tool — LLM must always call this to deliver its final message.
# await_followup is a schema-enforced bool; no [AWAIT] marker parsing needed.
_RESPOND_TOOL = {
    "name": "respond",
    "description": (
        "Send your spoken response to the user. Always call this last to deliver your final message. "
        "Set await_followup=true ONLY when the user's next response is required to complete the "
        "current task (e.g. a timer was requested but no duration was given). "
        "Never set it true after completing a task, giving information, or saying 'anything else?'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Your spoken response. Plain text only — no markdown, asterisks, or bullet points.",
            },
            "await_followup": {
                "type": "boolean",
                "description": "True only if the user must respond for the current task to be completed.",
            },
        },
        "required": ["text", "await_followup"],
    },
}


class LLM:
    """Handles LLM interactions with the Claude API."""

    def __init__(self, api_key: str = CLAUDE_API_KEY, model: str = CLAUDE_MODEL, max_history: int = MAX_CONVERSATION_HISTORY):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        logger.info(f"LLM initialized: Claude API ({model})")

    def _get_system_prompt(self, persistent_memory: str = "", speaker: str = None) -> str:
        """Build system prompt with current context injected."""
        now = datetime.now()
        time_context = f"Current: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
        if speaker:
            time_context += f" | Speaking: {speaker}"
        prompt = SYSTEM_PROMPT.format(persistent_memory=persistent_memory)
        prompt += f"\n<current_context>\n{time_context}\n</current_context>"
        return prompt

    @staticmethod
    def _serialize_content(content, exclude_names: set = None) -> list:
        """Convert Anthropic SDK content blocks to plain dicts. Optionally exclude tools by name."""
        result = []
        for block in content:
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                if exclude_names and block.name in exclude_names:
                    continue
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return result

    def _trim_history(self):
        """Trim history and ensure it starts with a plain-text user message.

        tool_result user messages without a preceding tool_use cause Claude API
        validation errors, so we walk past any such orphaned messages.
        """
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        while self.conversation_history:
            first = self.conversation_history[0]
            if first["role"] == "user" and isinstance(first["content"], str):
                break
            self.conversation_history.pop(0)

    def _api_call(self, system_prompt: str, tools: list) -> Optional[object]:
        """Single API call with hard wall-clock timeout."""
        try:
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=1024,
                timeout=30.0,
                system=system_prompt,
                tools=tools,
                tool_choice={"type": "any"},
                messages=self.conversation_history,
            )
            response = future.result(timeout=_LLM_CALL_TIMEOUT)
            self.last_usage["input_tokens"] += response.usage.input_tokens
            self.last_usage["output_tokens"] += response.usage.output_tokens
            return response
        except FutureTimeoutError:
            logger.error(f"LLM timed out after {_LLM_CALL_TIMEOUT}s (wall clock)")
            return None
        except anthropic.APIStatusError as e:
            logger.error(f"LLM request failed ({e.status_code}): {e.message}")
            return None
        except anthropic.APIConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None

    def chat(
        self,
        user_text: str,
        tools: List[Dict],
        tool_executor,
        persistent_memory: str = "",
        speaker: str = None
    ) -> Optional[Tuple[str, bool]]:
        """
        Send message to LLM and get response.

        The LLM must call respond(text, await_followup) as its final action.
        await_followup is a schema-enforced bool — no text marker parsing.

        Returns:
            Tuple of (reply_text, await_followup) or None on failure
        """
        self.conversation_history.append({"role": "user", "content": user_text})
        self._trim_history()

        system_prompt = self._get_system_prompt(persistent_memory, speaker)
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        all_tools = tools + [_RESPOND_TOOL]

        for round_num in range(6):  # max 5 command rounds + 1 respond round
            response = self._api_call(system_prompt, all_tools)
            if response is None:
                return None

            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Shouldn't happen with tool_choice=any — fall back gracefully
                logger.warning("LLM returned no tool calls despite tool_choice=any")
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if text:
                    self.conversation_history.append({"role": "assistant", "content": text})
                    self._trim_history()
                    return (text, False)
                return None

            respond_call = next((b for b in tool_calls if b.name == "respond"), None)
            command_calls = [b for b in tool_calls if b.name != "respond"]

            if respond_call and not command_calls:
                # Clean respond-only turn: store as plain text, skip the tool_use block
                text = respond_call.input.get("text", "").strip()
                await_followup = bool(respond_call.input.get("await_followup", False))
                if not text:
                    logger.warning("respond called with empty text")
                    return None
                logger.info(f"LLM responded: '{text}' (await={await_followup})")
                self.conversation_history.append({"role": "assistant", "content": text})
                self._trim_history()
                return (text, await_followup)

            if command_calls:
                # Store assistant turn, excluding the respond tool_use to keep history clean
                serialized = self._serialize_content(response.content, exclude_names={"respond"})
                self.conversation_history.append({"role": "assistant", "content": serialized})

                tool_results = []
                for call in command_calls:
                    logger.info(f"Executing tool: {call.name}({call.input})")
                    try:
                        result = tool_executor(call.name, **call.input)
                        logger.info(f"Tool '{call.name}' returned: {result}")
                    except Exception as e:
                        logger.error(f"Tool '{call.name}' failed: {e}")
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": str(result),
                    })

                self.conversation_history.append({"role": "user", "content": tool_results})

                if respond_call:
                    # Commands + respond in same turn — take the respond result
                    text = respond_call.input.get("text", "").strip()
                    await_followup = bool(respond_call.input.get("await_followup", False))
                    if text:
                        logger.info(f"LLM responded: '{text}' (await={await_followup})")
                        self.conversation_history.append({"role": "assistant", "content": text})
                        self._trim_history()
                        return (text, await_followup)
                    # empty text with commands — fall through to next round

                continue  # loop back for the respond call

        logger.error("LLM did not call respond within the round limit")
        return None

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def get_history_snapshot(self) -> List[Dict]:
        """Return a copy of conversation history for background summarization."""
        return self.conversation_history.copy()

    def extract_memories(self, history_snapshot: List[Dict]) -> list:
        """Lightweight bare API call to extract memorable facts from a conversation.

        Returns list of (category, key, value) tuples, or [] on any failure.
        Uses no tools — expects raw JSON back. Safe to call from a background thread.
        """
        if not history_snapshot:
            return []

        lines = []
        for msg in history_snapshot:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                tag = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{tag}: {content.strip()}")
        transcript = "\n".join(lines)
        if not transcript.strip():
            return []

        summarizer_prompt = (
            "You are a memory extraction assistant. "
            "Read the conversation below and identify any NEW facts worth remembering long-term. "
            "Only extract clear, stable, personal facts (names, preferences, schedules, relationships, "
            "habits, home config). Do NOT extract: transient requests (timers, weather), "
            "things already obvious from context, or behavioral instructions. "
            "Be conservative — if uncertain, skip it. "
            "Respond with ONLY a JSON array. Each element: "
            "{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\"}. "
            "Valid categories: preferences, schedule, people, personal, home, other. "
            "If nothing is worth saving, respond with exactly: []"
        )

        try:
            import json
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=512,
                timeout=20.0,
                system=summarizer_prompt,
                messages=[{"role": "user", "content": f"Conversation:\n{transcript}"}],
            )
            response = future.result(timeout=30.0)
            raw = ""
            for block in response.content:
                if block.type == "text":
                    raw = block.text.strip()
                    break
            if not raw or raw == "[]":
                return []
            items = json.loads(raw)
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cat = str(item.get("category", "")).strip().lower()
                key = str(item.get("key", "")).strip().lower().replace(" ", "_")
                val = str(item.get("value", "")).strip()
                if cat and key and val:
                    results.append((cat, key, val))
            return results
        except Exception as e:
            logger.debug(f"extract_memories failed: {e}")
            return []

    def set_history(self, history: List[Dict]):
        """Set conversation history (for restoring from saved state)."""
        self.conversation_history = history
        logger.info(f"Conversation history restored ({len(history)} messages)")
