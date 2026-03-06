"""LLM integration with Claude API for conversational AI.

High-level flow for each user utterance (chat() method):
  1. Append user message to conversation_history.
  2. Trim history to MAX_CONVERSATION_HISTORY (drops oldest, compresses to summary).
  3. Enter tool loop (up to 3 rounds):
       Round 1: API call with tool_choice=auto
         -> end_turn (text only) -> return text (pure conversation)
         -> tool_use -> execute tools
            -> action commands only -> return "Done." (short-circuit, no round 2)
            -> any narrated command -> continue to round 2
       Round 2: Tool results -> API call -> LLM narrates result
       Round 3: (rare) Multi-step or error recovery

  No rollback: on API failure, user message stays in history. Errors are surfaced
  to the user, not silently dropped.

History management:
  - history is stored as [{role, content}] where content is either a string
    (user/assistant text) or a list of tool_use/tool_result dicts.
  - History must always START with a plain-text user message — tool_result orphans
    at the head are dropped to prevent Claude API role validation errors.
  - When history overflows, dropped messages are summarized in a background thread
    and stored in _history_summary, injected as <prior_context> next call.

Prompt caching:
  - _get_system_prompt() returns (base, dynamic) tuple.
  - base = persona + persistent_memory — stable between save_memory calls,
    marked cache_control=ephemeral so Anthropic reuses the KV cache (~2000-4000 tokens).
  - dynamic = current time, speaker, patterns — changes every call, never cached.
  - Last tool schema is also marked cacheable — stable all session.
  - Cache efficiency logged at DEBUG: "Cache: N read / M written".

Timeouts:
  - _LLM_CALL_TIMEOUT: wall-clock timeout on the Future (12s).  If the API hangs
    or times out on Anthropic's side, this prevents indefinite blocking.
  - SDK timeout: 10s client-side HTTP timeout; fires first in normal timeout scenarios.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, NamedTuple
import anthropic

from datetime import datetime
from server.config import CLAUDE_API_KEY, CLAUDE_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Hard wall-clock timeout per API call.  Set LONGER than the SDK timeout (10s) so
# the SDK normally fires first and the exception is caught as APIConnectionError.
# _LLM_CALL_TIMEOUT is a backstop for the rare case where the SDK itself deadlocks
# and never raises — then Future.result() fires and we log it as a wall-clock timeout.
_LLM_CALL_TIMEOUT = 12.0

# Shared thread pool for API calls, parallel tool execution, and background
# compression/summarization.  max_workers=8 ensures that a burst of parallel
# tool calls (e.g. "turn off lights AND TV AND set volume") can't starve the
# pool and block the next API call or background compression.  Old value of 4
# could deadlock when 3 slow tool calls + 1 API call + 1 compression competed.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm_api")


class ChatResult(NamedTuple):
    """Return type for LLM.chat()."""
    text: str
    commands_executed: list  # Command names that were called


# Commands whose results must be narrated by the LLM (second API round).
# Everything else short-circuits with "Done." (no second API round).
NARRATED_COMMANDS = frozenset({
    'get_weather', 'get_time', 'calculate',
    'list_timers', 'cancel_timer',
    'list_feedback', 'list_sonos', 'list_lights', 'list_scenes',
    'get_volume', 'save_memory', 'forget_memory',
    'log_feedback', 'resolve_feedback',
})


class LLM:
    """Handles LLM interactions with the Claude API."""

    def __init__(self, api_key: str = CLAUDE_API_KEY, model: str = CLAUDE_MODEL, max_history: int = MAX_CONVERSATION_HISTORY):
        # Anthropic SDK client — single instance shared across all calls
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_history = max_history

        # conversation_history: list of {role, content} dicts.
        # Content is either a plain string or a list of tool_use/tool_result blocks.
        # Must always start with a plain-text user message (enforced by _trim_history).
        self.conversation_history: List[Dict] = []

        # Cumulative token usage for the current interaction (reset at start of chat()).
        # Exposed to orchestrator for cost logging.
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Compressed summary of messages that were dropped from history by _trim_history().
        # Injected into the system prompt as <prior_context> so the assistant retains
        # continuity even after the history window rolls over.
        self._history_summary: str = ""

        # Lock protecting _history_summary — a background compression thread writes it
        # while the main interaction thread reads it in _get_system_prompt().
        self._summary_lock = threading.Lock()

        logger.info(f"LLM initialized: Claude API ({model})")

    def _compress_dropped_messages(self, messages: list):
        """Summarize messages being dropped from history into _history_summary.

        Fires in a background thread from _trim_history() so it never blocks the
        current response.  The summary will be available for the *next* interaction.

        If a prior summary exists, it's prepended so the compressor can chain
        summaries over very long sessions without losing early context.

        Only plain-text user/assistant messages are included — tool_use/tool_result
        blocks are skipped (not meaningful as conversation text).
        """
        lines = []
        for msg in messages:
            content = msg.get("content", "")
            # Only include plain string content — skip tool call/result blocks
            if isinstance(content, str) and content.strip():
                tag = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{tag}: {content.strip()}")
        if not lines:
            return  # Nothing summarizable — all dropped messages were tool blocks

        # Chain with existing summary if present
        with self._summary_lock:
            prior_summary = self._history_summary
        prior = f"Prior summary:\n{prior_summary}\n\n" if prior_summary else ""
        transcript = "\n".join(lines)

        try:
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=150,   # Compact — just enough for 2-3 sentences
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
            # Non-critical — losing the summary just means slightly less context
            logger.debug(f"History compression failed: {e}")

    def _get_system_prompt(self, persistent_memory: str = "", speaker: str = None, patterns: str = "") -> tuple:
        """Build system prompt split into (base, dynamic) for prompt caching.

        Returns a 2-tuple so _api_call() can send them as separate system blocks:
          base   — persona + persistent_memory.  Stable between save_memory calls.
                   Marked cache_control=ephemeral → Anthropic KV-caches this.
          dynamic — current timestamp, speaker name, history summary, usage patterns.
                   Changes every call → never cached.

        Splitting avoids re-processing ~2000-4000 tokens of tool schemas + persona
        on every API call.  Cache read/write counts are logged at DEBUG.

        Args:
            persistent_memory: Contents of data/memory.json, injected by orchestrator.
            speaker: Identified speaker name from Resemblyzer, or None.
            patterns: Formatted string from routines.get_patterns(), or "".
        """
        # Base: stable persona + user memory (cached by Anthropic)
        base = SYSTEM_PROMPT.format(persistent_memory=persistent_memory)

        # Dynamic: changes every call — time, speaker, history context, usage patterns
        now = datetime.now()
        time_context = f"Current: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
        if speaker:
            time_context += f" | Speaking: {speaker}"
        dynamic = f"<current_context>\n{time_context}\n</current_context>"

        # Inject compressed history summary if overflow has occurred
        with self._summary_lock:
            summary = self._history_summary
        if summary:
            dynamic += f"\n<prior_context>\n{summary}\n</prior_context>"

        # Inject usage patterns (e.g. "get_weather: 6:00–8:00 (7×)")
        if patterns:
            dynamic += f"\n<observed_patterns>\n{patterns}\n</observed_patterns>"

        return base, dynamic

    @staticmethod
    def _serialize_content(content) -> list:
        """Convert Anthropic SDK content blocks to plain dicts for history storage.

        The SDK returns typed objects (TextBlock, ToolUseBlock); history needs
        plain dicts that can be serialised / compared.
        """
        result = []
        for block in content:
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return result

    def _trim_history(self):
        """Trim history to max_history messages and enforce starting invariant.

        Two operations:
        1. If history exceeds max_history, drop the oldest messages.  Compress
           dropped messages in a background thread — summary ready by next call.
        2. Drop leading non-text-user messages.  Claude API requires that the
           first history message is a plain-text user message.  tool_result
           orphans at the head (left over from a previous tool round) would
           cause an API validation error ("messages must start with user role
           with string content").

        Called:
          - After appending the user message at the start of chat().
          - After appending the assistant's final text response.
          - After the "Done." short-circuit path.
        """
        if len(self.conversation_history) > self.max_history:
            to_drop = self.conversation_history[:-self.max_history]
            self.conversation_history = self.conversation_history[-self.max_history:]
            if to_drop:
                # Background thread — never blocks current response
                threading.Thread(
                    target=self._compress_dropped_messages,
                    args=(to_drop,),
                    daemon=True,
                    name="HistoryCompressor",
                ).start()

        # Enforce: history must start with a plain-text user message
        while self.conversation_history:
            first = self.conversation_history[0]
            if first["role"] == "user" and isinstance(first["content"], str):
                break
            self.conversation_history.pop(0)

    def _api_call(self, base_prompt: str, dynamic_context: str, tools: list) -> Optional[object]:
        """Make a single Claude API call with prompt caching and wall-clock timeout.

        System prompt is split into two blocks:
          - base_prompt: marked cache_control=ephemeral → Anthropic caches it
          - dynamic_context: no cache marker → re-processed every call

        The last tool schema is also marked cacheable — tool schemas are stable
        all session, so the full tool list is cached after the first call.

        Timeout strategy:
          - SDK timeout=10.0s: client-side HTTP timeout; normally fires first
          - future.result(timeout=_LLM_CALL_TIMEOUT=12s): backstop for SDK deadlock
          - SDK timeout → APIConnectionError; Future timeout → FutureTimeoutError

        Returns the raw Anthropic response object, or None on any error.
        """
        # System: two blocks — cached stable base, uncached dynamic context
        system_blocks = [
            {"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}},
        ]
        if dynamic_context:
            system_blocks.append({"type": "text", "text": dynamic_context})

        # Tools: mark the last entry cacheable so the full list is in the cache key
        cached_tools = list(tools)
        if cached_tools:
            last = dict(cached_tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            cached_tools[-1] = last

        try:
            api_kwargs = {
                "model": self.model,
                "max_tokens": 512,   # Voice responses are short; 512 is ample
                "timeout": 10.0,     # SDK-level HTTP timeout (client-side; fires first)
                "system": system_blocks,
                "messages": self.conversation_history,
            }
            if cached_tools:
                api_kwargs["tools"] = cached_tools
                api_kwargs["tool_choice"] = {"type": "auto"}

            future = _executor.submit(
                self.client.messages.create, **api_kwargs
            )
            # Wall-clock timeout — fires even if the SDK is stuck waiting for bytes
            response = future.result(timeout=_LLM_CALL_TIMEOUT)

            # Accumulate token usage for cost reporting in orchestrator
            usage = response.usage
            self.last_usage["input_tokens"] += usage.input_tokens
            self.last_usage["output_tokens"] += usage.output_tokens

            # Log cache efficiency — useful for tuning what to cache
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
    ) -> Optional[ChatResult]:
        """Process user utterance. Returns ChatResult or None on total failure.

        Flow:
          Round 1: API call with tool_choice=auto
            -> end_turn -> return text (pure conversation)
            -> tool_use -> execute tools
               -> action commands only -> return "Done." (short-circuit, no round 2)
               -> any narrated command -> continue to round 2
          Round 2: Tool results -> API call -> LLM narrates result
          Round 3: (rare) Multi-step or error recovery

        No rollback: on API failure, user message stays in history. Errors are
        surfaced to the user, not silently dropped.

        Args:
            user_text: Transcribed speech from the user.
            tools: List of command tool schemas from commands.get_tools().
            tool_executor: Callable(name, **kwargs) -> str; wraps commands.execute().
            persistent_memory: Loaded memory.json contents for system prompt.
            speaker: Identified speaker name (for personalisation), or None.
            patterns: Usage pattern string from routines.get_patterns(), or "".

        Returns:
            ChatResult(text, commands_executed) on success, None on failure.
        """
        self.conversation_history.append({"role": "user", "content": user_text})
        self._trim_history()

        base_prompt, dynamic_context = self._get_system_prompt(
            persistent_memory, speaker, patterns
        )
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}
        commands_executed = []

        for round_num in range(3):
            response = self._api_call(base_prompt, dynamic_context, tools)

            if response is None:
                if commands_executed:
                    self.conversation_history.append(
                        {"role": "assistant", "content": "Done."}
                    )
                    self._trim_history()
                    return ChatResult("Done.", commands_executed)
                return None

            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Pure text response (tool_choice=auto, LLM chose not to use tools)
                text = "".join(
                    b.text for b in response.content if b.type == "text"
                ).strip()
                if not text:
                    logger.warning("LLM returned empty response")
                    return None
                self.conversation_history.append(
                    {"role": "assistant", "content": text}
                )
                self._trim_history()
                return ChatResult(text, commands_executed)

            # Capture any text the LLM sent alongside tool calls (e.g.
            # "Let me set that for you" + [tool_use]).  Used in place of
            # "Done." so the user hears the LLM's phrasing, not a generic ack.
            accompanying_text = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip()

            # Execute tool calls
            serialized = self._serialize_content(response.content)
            self.conversation_history.append(
                {"role": "assistant", "content": serialized}
            )

            tool_results = []
            needs_narration = False

            def _exec_one(call):
                logger.info(f"Executing tool: {call.name}({call.input})")
                try:
                    result = tool_executor(call.name, **call.input)
                    logger.info(f"Tool '{call.name}' returned: {result}")
                    return str(result)
                except Exception as e:
                    logger.error(f"Tool '{call.name}' failed: {e}")
                    return f"Error: {e}"

            # Execute all tool calls via thread pool with timeout (even single
            # calls — prevents indefinite hang if a command blocks).
            futures = [
                (call, _executor.submit(_exec_one, call))
                for call in tool_calls
            ]
            for call, fut in futures:
                try:
                    result_str = fut.result(timeout=30.0)
                except Exception as e:
                    result_str = f"Error: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result_str,
                })
                commands_executed.append(call.name)
                if call.name in NARRATED_COMMANDS:
                    needs_narration = True

            self.conversation_history.append(
                {"role": "user", "content": tool_results}
            )

            # Action commands: short-circuit without narration round.
            # Check for errors first — if any tool returned an error, let the LLM
            # see the results and narrate the failure instead of saying "Done." when
            # something actually broke.  This prevents the user hearing "Done." for
            # a timer that was never set or a light that didn't respond.
            has_tool_error = any(
                r["content"].startswith("Error:") for r in tool_results
            )
            if not needs_narration and not has_tool_error:
                short_reply = accompanying_text or "Done."
                self.conversation_history.append(
                    {"role": "assistant", "content": short_reply}
                )
                self._trim_history()
                return ChatResult(short_reply, commands_executed)

            # Narrated commands: loop back for LLM to read results and respond
            continue

        # Exhausted rounds
        logger.warning("LLM exhausted round limit")
        if commands_executed:
            self.conversation_history.append(
                {"role": "assistant", "content": "Done."}
            )
            self._trim_history()
            return ChatResult("Done.", commands_executed)
        return None

    def clear_history(self):
        """Clear conversation history (e.g. on explicit user request or /api/conversation/clear)."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Return a copy of current conversation history."""
        return self.conversation_history.copy()

    def get_history_snapshot(self) -> List[Dict]:
        """Return a copy of history for use by background threads (session summarizer).

        Caller must not modify the returned list — it's a shallow copy of the
        history at this point in time.
        """
        return self.conversation_history.copy()

    def extract_memories(self, history_snapshot: List[Dict]) -> list:
        """Extract memorable facts from a completed conversation.

        Called in a background thread by the session summarizer after each
        non-follow-up interaction.  Makes a bare API call (no tools) and expects
        raw JSON back.  Conservative — only extracts clear, stable personal facts.

        Returns:
            List of (category, key, value) tuples, e.g.:
            [("preferences", "coffee", "dark roast"), ("people", "sister", "Laura")]
            Empty list on any failure or if nothing is worth saving.
        """
        if not history_snapshot:
            return []

        # Build transcript from plain-text messages only (skip tool blocks)
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
        """Restore conversation history from a saved state."""
        self.conversation_history = history
        logger.info(f"Conversation history restored ({len(history)} messages)")
