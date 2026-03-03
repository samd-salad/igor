"""LLM integration with Claude API for conversational AI."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, Tuple
import anthropic

from datetime import datetime
from server.config import CLAUDE_API_KEY, CLAUDE_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_LLM_CALL_TIMEOUT = 12.0  # hard wall-clock timeout per API call (seconds)

_ERROR_FRAGMENTS = (
    'not found', 'not available', 'not connected', 'not installed',
    'not configured', 'unavailable', 'failed', 'adb error', 'error:',
    'pi not connected', 'no sonos', 'timed out',
)

def _is_tool_error(result: str) -> bool:
    """Return True if a tool result string indicates a failure."""
    r = result.lower()
    return any(frag in r for frag in _ERROR_FRAGMENTS)
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
        self._history_summary: str = ""
        self._summary_lock = threading.Lock()  # guards _history_summary across threads
        logger.info(f"LLM initialized: Claude API ({model})")

    def _compress_dropped_messages(self, messages: list):
        """Summarize messages being dropped from history into _history_summary.
        Runs in a background thread — summary is ready for the next interaction.
        """
        lines = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                tag = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{tag}: {content.strip()}")
        if not lines:
            return

        with self._summary_lock:
            prior_summary = self._history_summary
        prior = f"Prior summary:\n{prior_summary}\n\n" if prior_summary else ""
        transcript = "\n".join(lines)
        try:
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=150,
                timeout=15.0,
                system="Summarize the key facts and context from this conversation excerpt in 2-3 sentences. Focus on what matters for continuing the conversation naturally.",
                messages=[{"role": "user", "content": f"{prior}Conversation:\n{transcript}"}],
            )
            response = future.result(timeout=20.0)
            for block in response.content:
                if block.type == "text":
                    with self._summary_lock:
                        self._history_summary = block.text.strip()
                    logger.debug(f"History compressed: {self._history_summary[:80]}...")
                    break
        except Exception as e:
            logger.debug(f"History compression failed: {e}")

    def _get_system_prompt(self, persistent_memory: str = "", speaker: str = None, patterns: str = "") -> tuple:
        """Build system prompt split into (base, dynamic) for prompt caching.

        base   — persona + persistent memory; stable between save_memory calls.
                 Marked cache_control=ephemeral so Anthropic reuses the KV cache.
        dynamic — current time/speaker/context; changes every call, never cached.
        """
        base = SYSTEM_PROMPT.format(persistent_memory=persistent_memory)

        now = datetime.now()
        time_context = f"Current: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
        if speaker:
            time_context += f" | Speaking: {speaker}"
        dynamic = f"<current_context>\n{time_context}\n</current_context>"

        with self._summary_lock:
            summary = self._history_summary
        if summary:
            dynamic += f"\n<prior_context>\n{summary}\n</prior_context>"
        if patterns:
            dynamic += f"\n<observed_patterns>\n{patterns}\n</observed_patterns>"

        return base, dynamic

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

        When messages are dropped, fires history compression in a background
        thread so it never blocks the current response. Summary will be ready
        context isn't lost. tool_result orphans are still dropped to prevent
        Claude API role validation errors.
        """
        if len(self.conversation_history) > self.max_history:
            to_drop = self.conversation_history[:-self.max_history]
            self.conversation_history = self.conversation_history[-self.max_history:]
            if to_drop:
                threading.Thread(
                    target=self._compress_dropped_messages,
                    args=(to_drop,),
                    daemon=True,
                    name="HistoryCompressor",
                ).start()
        while self.conversation_history:
            first = self.conversation_history[0]
            if first["role"] == "user" and isinstance(first["content"], str):
                break
            self.conversation_history.pop(0)

    def _api_call(self, base_prompt: str, dynamic_context: str, tools: list) -> Optional[object]:
        """Single API call with prompt caching and hard wall-clock timeout.

        base_prompt is marked cache_control=ephemeral so Anthropic's KV cache
        can skip reprocessing the persona/memory on every turn.  dynamic_context
        (current time, speaker, patterns) changes each call and is never cached.
        The last tool is also marked cacheable — tool schemas are stable all session.
        """
        # System: two blocks — cached stable base, uncached dynamic context
        system_blocks = [
            {"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}},
        ]
        if dynamic_context:
            system_blocks.append({"type": "text", "text": dynamic_context})

        # Tools: mark the last entry so the entire list is in the cache key
        cached_tools = list(tools)
        if cached_tools:
            last = dict(cached_tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            cached_tools[-1] = last

        try:
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=512,
                timeout=10.0,
                system=system_blocks,
                tools=cached_tools,
                tool_choice={"type": "any"},
                messages=self.conversation_history,
            )
            response = future.result(timeout=_LLM_CALL_TIMEOUT)
            usage = response.usage
            self.last_usage["input_tokens"] += usage.input_tokens
            self.last_usage["output_tokens"] += usage.output_tokens
            # Log cache efficiency when tokens are served from cache
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_write:
                logger.debug(f"Cache: {cache_read} read / {cache_write} written")
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
        speaker: str = None,
        patterns: str = "",
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

        # Snapshot AFTER trim so rollback restores the already-compacted state.
        # If taken before trim, a rollback would restore pre-trim messages and
        # the next call would re-drop and re-compress the same messages.
        _history_snapshot = list(self.conversation_history)

        base_prompt, dynamic_context = self._get_system_prompt(persistent_memory, speaker, patterns)
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        all_tools = tools + [_RESPOND_TOOL]

        def _fail():
            """Roll back conversation history and return None."""
            self.conversation_history = _history_snapshot
            return None

        commands_succeeded = False  # True after at least one round with no tool errors

        for round_num in range(6):  # max 5 command rounds + 1 respond round
            response = self._api_call(base_prompt, dynamic_context, all_tools)
            if response is None:
                if commands_succeeded:
                    # Commands ran fine; only the respond round timed out.
                    # Store "Done." so the conversation loop closes properly,
                    # then return so it gets synthesized and played.
                    logger.warning("Respond round timed out after successful commands — returning 'Done.'")
                    self.conversation_history.append({"role": "assistant", "content": "Done."})
                    self._trim_history()
                    return ("Done.", False)
                return _fail()

            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Shouldn't happen with tool_choice=any — fall back gracefully
                logger.warning("LLM returned no tool calls despite tool_choice=any")
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if text:
                    self.conversation_history.append({"role": "assistant", "content": text})
                    self._trim_history()
                    return (text, False)
                return _fail()

            respond_call = next((b for b in tool_calls if b.name == "respond"), None)
            command_calls = [b for b in tool_calls if b.name != "respond"]

            if respond_call and not command_calls:
                # Clean respond-only turn: store as plain text, skip the tool_use block
                text = respond_call.input.get("text", "").strip()
                await_followup = bool(respond_call.input.get("await_followup", False))
                if not text:
                    logger.warning("respond called with empty text")
                    return _fail()
                logger.info(f"LLM responded: '{text}' (await={await_followup})")
                self.conversation_history.append({"role": "assistant", "content": text})
                self._trim_history()
                return (text, await_followup)

            if command_calls:
                # Store assistant turn, excluding the respond tool_use to keep history clean
                serialized = self._serialize_content(response.content, exclude_names={"respond"})
                self.conversation_history.append({"role": "assistant", "content": serialized})

                def _exec_call(call):
                    logger.info(f"Executing tool: {call.name}({call.input})")
                    try:
                        result = tool_executor(call.name, **call.input)
                        logger.info(f"Tool '{call.name}' returned: {result}")
                        return str(result), _is_tool_error(result)
                    except Exception as e:
                        logger.error(f"Tool '{call.name}' failed: {e}")
                        return f"Error: {e}", True

                any_tool_error = False
                if len(command_calls) == 1:
                    result_str, is_err = _exec_call(command_calls[0])
                    if is_err:
                        any_tool_error = True
                    tool_results = [{"type": "tool_result", "tool_use_id": command_calls[0].id, "content": result_str}]
                else:
                    # Multiple independent commands — execute in parallel, preserve order
                    futures = [(call, _executor.submit(_exec_call, call)) for call in command_calls]
                    tool_results = []
                    for call, fut in futures:
                        try:
                            result_str, is_err = fut.result(timeout=30.0)
                        except Exception as e:
                            logger.error(f"Tool '{call.name}' executor error: {e}")
                            result_str, is_err = f"Error: {e}", True
                        if is_err:
                            any_tool_error = True
                        tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result_str})

                self.conversation_history.append({"role": "user", "content": tool_results})

                if not any_tool_error:
                    commands_succeeded = True

                # Short-circuit: if tools failed and there's no respond in this turn,
                # skip the next LLM API call and surface the error immediately.
                if any_tool_error and not respond_call:
                    logger.warning("Tool error detected — skipping LLM response, surfacing error immediately")
                    return _fail()

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
        return _fail()

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
