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
  - base = persona + behavior_rules — stable between save_memory calls,
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
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, NamedTuple
import anthropic

from datetime import datetime
from server.config import CLAUDE_API_KEY, CLAUDE_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _safe_log_params(name, params):
    """Redact sensitive values from tool params for logging."""
    if not params:
        return "{}"
    safe = {}
    for k, v in params.items():
        if k == "value" and name in ("save_memory", "log_feedback"):
            safe[k] = f"{str(v)[:20]}..." if len(str(v)) > 20 else str(v)
        else:
            safe[k] = v
    return safe


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
    'get_volume',
    'log_feedback', 'resolve_feedback',
    'save_memory', 'forget_memory',
    'delayed_command',
})

# Response texture: rotate confirmations instead of always "Done."
_CONFIRMATIONS = {
    "light": ["Lights adjusted.", "Done.", "Got it.", "Set."],
    "volume": ["Volume set.", "Done.", "Got it.", "Adjusted."],
    "tv": ["Done.", "Got it.", "All set."],
    "timer": ["Timer set.", "Done.", "Got it."],
    "default": ["Done.", "Got it.", "All set.", "Handled.", "Sorted."],
}


def _pick_confirmation(commands: list[str]) -> str:
    """Pick a contextual confirmation from the rotation pool."""
    for cmd in commands:
        if any(k in cmd for k in ("light", "bright", "color", "scene", "hue")):
            return random.choice(_CONFIRMATIONS["light"])
        if any(k in cmd for k in ("volume", "sonos", "mute")):
            return random.choice(_CONFIRMATIONS["volume"])
        if "tv" in cmd or "adb" in cmd:
            return random.choice(_CONFIRMATIONS["tv"])
        if "timer" in cmd:
            return random.choice(_CONFIRMATIONS["timer"])
    return random.choice(_CONFIRMATIONS["default"])


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

    def _get_system_prompt(self, behavior_rules: str = "", speaker: str = None,
                           patterns: str = "", relevant_memories: str = "",
                           recent_episodes: str = "",
                           identity_narrative: str = "") -> tuple:
        """Build system prompt split into (base, dynamic) for prompt caching.

        Returns a 2-tuple so _api_call() can send them as separate system blocks:
          base   — persona + behavior_rules + identity_narrative.  Stable between
                   memory changes.  Marked cache_control=ephemeral → Anthropic
                   KV-caches this (~2000-4000 tokens saved per call).
          dynamic — current timestamp, speaker name, relevant memories, recent
                   episodes, history summary, usage patterns.  Changes every call.

        Args:
            behavior_rules: Behavior rules from BrainStore, injected by orchestrator.
            speaker: Identified speaker name from Resemblyzer, or None.
            patterns: Formatted string from routines.get_patterns(), or "".
            relevant_memories: Pre-formatted relevant memories for this query.
            recent_episodes: Formatted recent episodic memories.
            identity_narrative: Living narrative about the user (from consolidation).
        """
        # Base: stable persona + behavior rules + identity (cached by Anthropic)
        base = SYSTEM_PROMPT.format(behavior_rules=behavior_rules)

        # Identity narrative: living profile of the user, always present
        if identity_narrative:
            base += f"\n\n<my_person>\n{identity_narrative}\n</my_person>"

        # Dynamic: changes every call — time, speaker, episodes, memories, patterns
        now = datetime.now()
        time_context = f"Current: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
        if speaker and speaker != "unknown":
            time_context += f" | Speaking: {speaker}"
        elif speaker == "unknown":
            time_context += " | Speaking: unrecognized voice (not an enrolled household member — be helpful but don't assume identity or save personal details)"
        dynamic = f"<current_context>\n{time_context}\n</current_context>"

        # Inject relevant memories retrieved for this specific query
        if relevant_memories:
            dynamic += f"\n<relevant_memories>\n{relevant_memories}\n</relevant_memories>"

        # Inject recent episodic memories (replaces old session summaries)
        if recent_episodes:
            dynamic += f"\n<recent_episodes>\n{recent_episodes}\n</recent_episodes>"

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
        behavior_rules: str = "",
        speaker: str = None,
        patterns: str = "",
        relevant_memories: str = "",
        recent_episodes: str = "",
        identity_narrative: str = "",
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
            behavior_rules: Behavior rules from BrainStore for system prompt.
            speaker: Identified speaker name (for personalisation), or None.
            patterns: Usage pattern string from routines.get_patterns(), or "".
            relevant_memories: Pre-formatted relevant memories for this query.
            recent_episodes: Formatted recent episodic memories.
            identity_narrative: Living narrative about the user (from consolidation).

        Returns:
            ChatResult(text, commands_executed) on success, None on failure.
        """
        self.conversation_history.append({"role": "user", "content": user_text})
        self._trim_history()

        base_prompt, dynamic_context = self._get_system_prompt(
            behavior_rules, speaker, patterns, relevant_memories,
            recent_episodes, identity_narrative
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
                    if commands_executed:
                        # LLM returned empty narration after tool execution —
                        # fall back to "Done." instead of failing the interaction.
                        logger.warning("LLM returned empty narration after tools, falling back to Done.")
                        text = "Done."
                    else:
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
                logger.info(f"Executing tool: {call.name}({_safe_log_params(call.name, call.input)})")
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
                short_reply = accompanying_text or _pick_confirmation(commands_executed)
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

    def analyze_conversation(self, history_snapshot: List[Dict],
                             commands_executed: list = None) -> dict:
        """Extract facts AND generate an episode summary from a conversation.

        Combines memory extraction and episode generation into a single API call
        (zero extra cost vs the old extract_memories approach).

        Returns:
            {"facts": [(cat, key, val), ...],
             "episode": {"summary": str, "topics": [str], "emotional_tone": str}}
        """
        if not history_snapshot:
            return {"facts": [], "episode": None}

        lines = []
        for msg in history_snapshot:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                tag = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{tag}: {content.strip()}")
        transcript = "\n".join(lines)
        if not transcript.strip():
            return {"facts": [], "episode": None}

        cmds_note = ""
        if commands_executed:
            cmds_note = f"\nCommands used: {', '.join(dict.fromkeys(commands_executed))}"

        prompt = (
            "You are a memory extraction and conversation analysis assistant. "
            "Read the conversation below and produce TWO things:\n\n"
            "1. FACTS: Any NEW facts worth remembering long-term about the user. "
            "Only clear, stable, personal facts (names, preferences, schedules, "
            "relationships, habits, home config). Be conservative.\n\n"
            "2. EPISODE: A brief (1-2 sentence) summary of what happened in this "
            "interaction. Write it as a third-person narrative for the assistant's "
            "episodic memory. Include the user's apparent mood/tone if notable.\n\n"
            "Respond with ONLY JSON:\n"
            '{"facts": [{"category": "...", "key": "...", "value": "..."}], '
            '"episode": {"summary": "...", "topics": ["..."], "emotional_tone": "..."}}\n\n'
            "Valid fact categories: preferences, schedule, people, personal, home, other.\n"
            "emotional_tone: one word (neutral, upbeat, stressed, playful, tired, frustrated) or empty.\n"
            "If no facts worth saving, use empty facts array. Always produce an episode."
        )

        try:
            import json
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=512,
                timeout=20.0,
                system=prompt,
                messages=[{"role": "user", "content": f"Conversation:\n{transcript}{cmds_note}"}],
            )
            response = future.result(timeout=30.0)
            raw = ""
            for block in response.content:
                if block.type == "text":
                    raw = block.text.strip()
                    break
            if not raw:
                return {"facts": [], "episode": None}

            data = json.loads(raw)

            # Parse facts
            facts = []
            for item in data.get("facts", []):
                if isinstance(item, dict):
                    cat = str(item.get("category", "")).strip().lower()
                    key = str(item.get("key", "")).strip().lower().replace(" ", "_")
                    val = str(item.get("value", "")).strip()
                    if cat and key and val:
                        facts.append((cat, key, val))

            # Parse episode
            episode = None
            ep_data = data.get("episode")
            if isinstance(ep_data, dict) and ep_data.get("summary"):
                episode = {
                    "summary": str(ep_data["summary"])[:300],
                    "topics": [str(t).lower() for t in ep_data.get("topics", []) if t][:5],
                    "emotional_tone": str(ep_data.get("emotional_tone", ""))[:20].lower(),
                }

            return {"facts": facts, "episode": episode}
        except Exception as e:
            logger.debug(f"analyze_conversation failed: {e}")
            return {"facts": [], "episode": None}

    def generate_identity_narrative(self, memories: dict, episodes: list,
                                     gaps: list) -> str:
        """Generate a living narrative paragraph about the user.

        Called by the consolidation engine.  Synthesizes all known memories and
        recent episodes into a warm, factual paragraph that becomes the LLM's
        core knowledge of its person.  One Haiku call (~$0.001).

        Args:
            memories: {category: {key: value}} from brain.get_all_memories().
            episodes: Recent episode entries from brain.get_recent_episodes().
            gaps: Unfilled profile questions from brain.get_knowledge_gaps().

        Returns:
            Narrative paragraph string, or "" on failure.
        """
        # Format memories
        mem_lines = []
        for cat, items in sorted(memories.items()):
            for key, val in sorted(items.items()):
                mem_lines.append(f"  [{cat}] {key}: {val}")
        mem_text = "\n".join(mem_lines) if mem_lines else "(no memories yet)"

        # Format recent episodes
        ep_lines = []
        for ep in episodes[:10]:
            data = ep.get("data", {})
            ts = ep.get("created", "")[:10]
            summary = data.get("summary", "")
            tone = data.get("emotional_tone", "")
            if summary:
                tone_note = f" [{tone}]" if tone else ""
                ep_lines.append(f"  [{ts}] {summary}{tone_note}")
        ep_text = "\n".join(ep_lines) if ep_lines else "(no interactions recorded yet)"

        # Format gaps
        gaps_text = "\n".join(f"  - {g}" for g in gaps[:8]) if gaps else "(profile complete)"

        prompt = (
            "You are a memory consolidation system for a voice assistant named Igor. "
            "Igor is a dry, sardonic assistant who lives in a household.\n\n"
            "Given the raw memories and recent interaction episodes below, write a "
            "concise household narrative (4-6 sentences) that captures who lives here — "
            "each person's name, identity, daily rhythm, and notable preferences. "
            "Cover all household members mentioned in memories, not just the primary user.\n\n"
            "Write in third person present tense. Be specific (use names, times, details). "
            "This paragraph is Igor's core knowledge of his household — it should read like "
            "how a close friend would describe a home, not like a database printout.\n\n"
            "If there are notable gaps in what Igor knows, end with ONE sentence noting "
            "the 2-3 most interesting unknowns (phrased as things Igor is curious about, "
            "not as database fields).\n\n"
            f"Raw memories:\n{mem_text}\n\n"
            f"Recent interactions:\n{ep_text}\n\n"
            f"Known gaps:\n{gaps_text}\n\n"
            "Write ONLY the narrative paragraph, nothing else:"
        )

        try:
            future = _executor.submit(
                self.client.messages.create,
                model=self.model,
                max_tokens=400,
                timeout=20.0,
                system="You write concise, warm, factual household narratives.",
                messages=[{"role": "user", "content": prompt}],
            )
            response = future.result(timeout=30.0)
            for block in response.content:
                if block.type == "text":
                    narrative = block.text.strip()
                    # Strip quotes the LLM might wrap it in
                    if narrative.startswith('"') and narrative.endswith('"'):
                        narrative = narrative[1:-1]
                    return narrative
            return ""
        except Exception as e:
            logger.debug(f"generate_identity_narrative failed: {e}")
            return ""

    def set_history(self, history: List[Dict]):
        """Restore conversation history from a saved state."""
        self.conversation_history = history
        logger.info(f"Conversation history restored ({len(history)} messages)")
