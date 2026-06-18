"""Conversation — the turn orchestrator. The only service ha_io knows about."""
from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Optional

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import (
    ConversationResult, Episode, ToolCallRecord, VoiceTurn,
)
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort
from server.cognition.ports.retrieval import RetrievalPort
from server.cognition.ports.tools import ToolExecutorPort
from server.cognition.services.intent_router import IntentRouter, Tier1Match
from server.cognition.services.quality_gate import QualityGate
from server.cognition.services.session_summarizer import SessionSummarizer
from server.cognition._internal.prompt_builder import build_system_prompt, build_user_context

logger = logging.getLogger(__name__)

_ECHO_WINDOW_SECS = 8.0
_ECHO_OVERLAP_THRESHOLD = 0.5
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _token_set(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


class Conversation:
    def __init__(
        self,
        memory: MemoryStore,
        episodes: EpisodeStore,
        identity: IdentityStore,
        user_state: UserState,
        retrieval: RetrievalPort,
        llm: LLMPort,
        tools: ToolExecutorPort,
        clock: ClockPort,
        summarizer: Optional[SessionSummarizer] = None,
    ):
        self._memory = memory
        self._episodes = episodes
        self._identity = identity
        self._user_state = user_state
        self._retrieval = retrieval
        self._llm = llm
        self._tools = tools
        self._clock = clock
        self._summarizer = summarizer
        self._quality_gate = QualityGate()
        self._intent_router = IntentRouter()
        self._last_response: Optional[tuple[datetime, set[str]]] = None

    def process(self, turn: VoiceTurn) -> ConversationResult:
        if self._is_self_echo(turn):
            logger.info("Echo detected; staying silent (overlap with prior response within %.0fs)",
                        _ECHO_WINDOW_SECS)
            self._persist_episode(turn, "", [], intent="echo_suppressed")
            return ConversationResult(
                correlation_id=turn.correlation_id,
                response_text="",
                commands_executed=[],
                end_conversation=True,
                silent=True,
            )

        gate = self._quality_gate.filter(turn)
        if gate.rejected:
            result = ConversationResult(
                correlation_id=turn.correlation_id,
                response_text="Didn't catch that.",
                commands_executed=[], end_conversation=True,
            )
            self._persist_episode(turn, result.response_text, [], intent="rejected")
            self._remember_response(result.response_text)
            return result

        tier1: Tier1Match | None = self._intent_router.route(turn)
        if tier1 is not None:
            try:
                self._tools.execute(tier1.command, tier1.params, turn)
            except Exception:
                logger.exception("Tier1 execution failed")
            result = ConversationResult(
                correlation_id=turn.correlation_id,
                response_text=tier1.response,
                commands_executed=[tier1.command],
                end_conversation=True,
            )
            self._persist_episode(
                turn, tier1.response,
                [ToolCallRecord(tier1.command, tier1.params, tier1.response)],
                intent="tier1",
            )
            self._remember_response(result.response_text)
            self._enqueue_summary(turn, result)
            return result

        relevant = self._retrieval.query(turn, k=3)
        recent_eps = self._episodes.get_recent(5)
        system_prompt = build_system_prompt(self._identity.get_narrative())
        user_context = build_user_context(turn, relevant, recent_eps)
        tool_results_log: list[ToolCallRecord] = []

        def _exec(name: str, args: dict) -> str:
            result_text = self._tools.execute(name, args, turn)
            tool_results_log.append(ToolCallRecord(name=name, args=args, result=result_text))
            return result_text

        chat = self._llm.chat(
            system_prompt=system_prompt,
            user_text=user_context,
            tool_schemas=self._tools.list_schemas(),
            tool_executor=_exec,
        )

        self._persist_episode(turn, chat.text, tool_results_log, intent="llm")
        result = ConversationResult(
            correlation_id=turn.correlation_id,
            response_text=chat.text,
            commands_executed=chat.commands_executed,
            end_conversation=True,
        )
        self._remember_response(result.response_text)
        self._enqueue_summary(turn, result)
        return result

    def _is_self_echo(self, turn: VoiceTurn) -> bool:
        """True when the just-arrived transcript looks like Igor's own voice
        bouncing back through the mic. Window-bounded so a real user reply
        moments after Igor speaks is never suppressed."""
        if self._last_response is None:
            return False
        last_at, last_tokens = self._last_response
        if not last_tokens:
            return False
        delta = (turn.started_at - last_at).total_seconds()
        if delta < 0 or delta > _ECHO_WINDOW_SECS:
            return False
        input_tokens = _token_set(turn.input_text)
        if not input_tokens:
            return False
        overlap = len(input_tokens & last_tokens) / len(input_tokens)
        return overlap >= _ECHO_OVERLAP_THRESHOLD

    def _remember_response(self, response_text: str) -> None:
        if not response_text:
            return
        self._last_response = (self._clock.now(), _token_set(response_text))

    def _enqueue_summary(self, turn: VoiceTurn, result: ConversationResult) -> None:
        if self._summarizer is None:
            return
        try:
            self._summarizer.enqueue(turn, result)
        except Exception:
            logger.exception("Failed to enqueue summarizer")

    def _persist_episode(self, turn: VoiceTurn, response: str,
                         tool_calls: list[ToolCallRecord], intent: str) -> None:
        ep = Episode(
            episode_id=turn.correlation_id,
            occurred_at=turn.started_at,
            speaker_id=turn.speaker_id,
            participants=[turn.speaker_id or "user", "igor"],
            intent=intent,
            raw_utterance=turn.input_text,
            tool_calls=tool_calls,
            emotional_tone=None,
            summary=None,
            consolidated_at=None,
            response_text=response or None,
        )
        self._episodes.add(ep)
