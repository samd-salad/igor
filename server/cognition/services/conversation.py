"""Conversation — the turn orchestrator. The only service ha_io knows about."""
from __future__ import annotations
import logging

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
from server.cognition._internal.prompt_builder import build_system_prompt, build_user_context

logger = logging.getLogger(__name__)


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
    ):
        self._memory = memory
        self._episodes = episodes
        self._identity = identity
        self._user_state = user_state
        self._retrieval = retrieval
        self._llm = llm
        self._tools = tools
        self._clock = clock
        self._quality_gate = QualityGate()
        self._intent_router = IntentRouter()

    def process(self, turn: VoiceTurn) -> ConversationResult:
        gate = self._quality_gate.filter(turn)
        if gate.rejected:
            self._persist_episode(turn, "Didn't catch that.", [], intent="rejected")
            return ConversationResult(
                correlation_id=turn.correlation_id,
                response_text="Didn't catch that.",
                commands_executed=[], end_conversation=True,
            )

        tier1: Tier1Match | None = self._intent_router.route(turn)
        if tier1 is not None:
            try:
                self._tools.execute(tier1.command, tier1.params, turn)
            except Exception:
                logger.exception("Tier1 execution failed")
            self._persist_episode(turn, tier1.response,
                                  [ToolCallRecord(tier1.command, tier1.params, tier1.response)],
                                  intent="tier1")
            return ConversationResult(
                correlation_id=turn.correlation_id,
                response_text=tier1.response,
                commands_executed=[tier1.command],
                end_conversation=True,
            )

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

        return ConversationResult(
            correlation_id=turn.correlation_id,
            response_text=chat.text,
            commands_executed=chat.commands_executed,
            end_conversation=True,
        )

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
        )
        self._episodes.add(ep)
