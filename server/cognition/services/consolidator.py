"""Consolidator — sleep-time identity regeneration. Background thread + crash-replay."""
from __future__ import annotations
import logging
import threading
from typing import Optional

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort

logger = logging.getLogger(__name__)


class Consolidator:
    def __init__(
        self,
        memory: MemoryStore,
        episodes: EpisodeStore,
        identity: IdentityStore,
        llm: LLMPort,
        clock: ClockPort,
        *,
        episodes_per_run: int = 5,
        poll_interval_seconds: float = 60.0,
    ):
        self._memory = memory
        self._episodes = episodes
        self._identity = identity
        self._llm = llm
        self._clock = clock
        self._episodes_per_run = episodes_per_run
        self._poll = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.replay_if_pending()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Consolidator")
        self._thread.start()

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._maybe_run()
            except Exception:
                logger.exception("Consolidator iteration failed")
            self._stop.wait(self._poll)

    def _maybe_run(self) -> None:
        if len(self._episodes.get_unconsolidated()) >= self._episodes_per_run:
            self.run_once()

    def replay_if_pending(self) -> None:
        if len(self._episodes.get_unconsolidated()) >= self._episodes_per_run:
            self.run_once()

    def run_once(self) -> None:
        episodes = self._episodes.get_unconsolidated()[: self._episodes_per_run]
        if not episodes:
            return
        prior_identity = self._identity.get_narrative()
        ep_lines = [f"- {e.occurred_at.isoformat()}: {e.summary or e.raw_utterance[:120]}"
                    for e in episodes]
        chat = self._llm.chat(
            system_prompt=(
                "Synthesize a brief living narrative about the user (single paragraph, "
                "<=4 sentences). Use prior narrative + recent episodes. "
                "Do not invent details."
            ),
            user_text=(
                f"Prior narrative:\n{prior_identity or '(empty)'}\n\n"
                "Recent episodes:\n" + "\n".join(ep_lines)
            ),
            tool_schemas=[], tool_executor=lambda n, a: "",
        )
        now = self._clock.now()
        last_id = episodes[-1].episode_id
        self._identity.replace_narrative(chat.text.strip(),
                                         last_consolidated_at=now,
                                         last_consolidated_episode_id=last_id)
        self._episodes.mark_consolidated([e.episode_id for e in episodes], at=now)
