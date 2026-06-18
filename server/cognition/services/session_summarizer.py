"""SessionSummarizer — drains turn results in a background thread, updates Episode.summary."""
from __future__ import annotations
import logging
import queue
import threading
from dataclasses import replace
from typing import Optional

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import ConversationResult, VoiceTurn
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort

logger = logging.getLogger(__name__)

_STOP = object()


class SessionSummarizer:
    def __init__(self, episodes: EpisodeStore, memory: MemoryStore,
                 llm: LLMPort, clock: ClockPort):
        self._episodes = episodes
        self._memory = memory
        self._llm = llm
        self._clock = clock
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="SessionSummarizer")
        self._thread.start()

    def enqueue(self, turn: VoiceTurn, result: ConversationResult) -> None:
        self._queue.put((turn, result))

    def shutdown(self, timeout: float = 2.0) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._running = False

    def _run(self) -> None:
        while self._running:
            item = self._queue.get()
            if item is _STOP:
                self._running = False
                return
            try:
                self._summarize(*item)
            except Exception:
                logger.exception("Summarization failed")

    def _summarize(self, turn: VoiceTurn, result: ConversationResult) -> None:
        ep = self._episodes.load(turn.correlation_id)
        if ep is None:
            return
        chat = self._llm.chat(
            system_prompt="Summarize this assistant turn in <= 12 words.",
            user_text=f"User: {turn.input_text}\nIgor: {result.response_text}",
            tool_schemas=[],
            tool_executor=lambda n, a: "",
        )
        updated = replace(ep, summary=chat.text.strip())
        self._episodes.add(updated)
