"""ConversationService — text-in, text-out brain core.

Extracted from Orchestrator.process_text_interaction so the new
/conversation/process endpoint can run without any audio dependencies
(Whisper, Kokoro, PyAudio, etc.). The orchestrator + Pi/PC clients +
audio code is being deleted in a follow-up step; this is the
replacement entry point.

Pipeline (same shape as before, minus STT/TTS):
  text → quality_gate → intent_router (Tier 1) → LLM (Tier 2+)
       → response text + executed commands

Memory hooks (brain, episodes, identity narrative, behavior rules,
session summarizer, consolidation engine) all still run here. Per-room
context comes in via InteractionContext (HA area drives entity scoping
in the HA-backed device commands).
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

from server.brain import get_brain
from server.context import InteractionContext
from server.intent_router import route as route_intent
from server.llm import LLM
from server.quality_gate import filter_transcription
from server.routines import log_command, get_patterns
import server.commands as commands

logger = logging.getLogger(__name__)


# Commands that, after running, leave a follow-up question on the table —
# we set end_conversation=False so HA's pipeline keeps the mic open.
_FOLLOWUP_FILLER = (
    "anything else", "help you with", "else can i", "want me to",
    "like to know", "need anything", "can i do", "what else", "something else",
)


class ConversationService:
    """Stateless-ish text conversation. State lives in the brain + LLM history.

    Construct once at server startup; share across requests.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def process(self, text: str, ctx: Optional[InteractionContext] = None) -> dict:
        """Run text through the brain pipeline.

        Args:
            text: Already-transcribed user speech (HA's STT did the work).
            ctx: Optional room context. ctx.room.ha_area drives per-room
                 entity scoping in the device commands.

        Returns:
            {
                "response": str,
                "commands_executed": list[str],
                "end_conversation": bool,   # False when Igor asked a follow-up question
                "tier1": bool,              # True when Tier 1 router handled it (no LLM call)
            }
        """
        commands_executed: list[str] = []

        # Quality gate — runs on text now (not transcription); same logic.
        # tv_state is left "unknown" until we wire HA's media_player.<tv> state in.
        gate = filter_transcription(text, tv_playing=False)
        if gate.text is None:
            logger.info(f"Quality gate rejected: {gate.reason}")
            return {
                "response": "Didn't catch that.",
                "commands_executed": [],
                "end_conversation": True,
                "tier1": False,
            }
        text = gate.text

        # Tier 1: direct intent router
        tier1 = route_intent(text)
        if tier1 is not None:
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(self._execute_command, tier1.command, ctx, **tier1.params)
                    result = fut.result(timeout=30.0)
                # Surface real errors instead of the canned response
                rl = (result or "").lower().strip()
                _err_prefixes = (
                    "error", "failed", "not found", "not available",
                    "not connected", "unknown command", "no ", "didn't",
                )
                if any(rl.startswith(p) for p in _err_prefixes):
                    response_text = result
                else:
                    response_text = tier1.response
            except FutureTimeout:
                logger.error(f"Tier 1 '{tier1.command}' timed out")
                response_text = f"{tier1.command} timed out."
            except Exception as e:
                logger.error(f"Tier 1 '{tier1.command}' failed: {e}", exc_info=True)
                response_text = "Sorry, that command failed."
            commands_executed.append(tier1.command)
            return {
                "response": response_text,
                "commands_executed": commands_executed,
                "end_conversation": True,
                "tier1": True,
            }

        # Tier 2+: LLM with tools
        brain = get_brain()
        behavior_rules = brain.get_behavior_rules()
        relevant = brain.retrieve_relevant(text)
        relevant_memories = brain.format_relevant(relevant)
        recent_episodes = brain.format_episodes(brain.get_recent_episodes(limit=5))
        identity_narrative = brain.get_identity_narrative()
        patterns = get_patterns()
        tools = commands.get_tools()

        def tool_executor(command_name: str, **kwargs) -> str:
            commands_executed.append(command_name)
            return self._execute_command(command_name, ctx, **kwargs)

        llm_result = self.llm.chat(
            user_text=text,
            tools=tools,
            tool_executor=tool_executor,
            behavior_rules=behavior_rules,
            patterns=patterns,
            relevant_memories=relevant_memories,
            recent_episodes=recent_episodes,
            identity_narrative=identity_narrative,
        )
        if not llm_result:
            logger.error("LLM failed")
            return {
                "response": "Sorry, I'm having trouble thinking right now.",
                "commands_executed": commands_executed,
                "end_conversation": True,
                "tier1": False,
            }

        response_text = llm_result.text
        words = response_text.split()
        # Open the mic again only on a short, genuine clarification question
        end_conversation = True
        if response_text.rstrip().endswith("?") and not commands_executed and len(words) < 12:
            lower = response_text.lower()
            if not any(f in lower for f in _FOLLOWUP_FILLER):
                end_conversation = False

        # Background: session summarizer (memory + episode extraction)
        if end_conversation:
            snap = self.llm.get_history_snapshot()
            threading.Thread(
                target=self._run_session_summarizer,
                args=(snap, list(commands_executed)),
                daemon=True, name="SessionSummarizer",
            ).start()

        return {
            "response": response_text,
            "commands_executed": commands_executed,
            "end_conversation": end_conversation,
            "tier1": False,
        }

    # -- internals --

    def _execute_command(self, command_name: str, ctx: Optional[InteractionContext], **kwargs) -> str:
        try:
            result = commands.execute(command_name, _ctx=ctx, **kwargs)
            threading.Thread(target=log_command, args=(command_name,), daemon=True).start()
            return result
        except Exception as e:
            logger.error(f"Command '{command_name}' failed: {e}", exc_info=True)
            return "Command failed."

    def _run_session_summarizer(self, history_snapshot: list, commands_executed: list):
        """Same logic as Orchestrator._run_session_summarizer, minus speaker handling."""
        try:
            from server.commands.memory_cmd import _sanitize
            analysis = self.llm.analyze_conversation(history_snapshot, commands_executed)
            brain = get_brain()
            allowed = frozenset({"preferences", "schedule", "people", "personal", "home", "other"})
            for cat, key, val in analysis.get("facts", []):
                cat = _sanitize(cat, max_len=50).lower().strip()
                key = _sanitize(key, max_len=50).lower().strip().replace(" ", "_")
                val = _sanitize(val, max_len=500)
                if cat and key and val and cat in allowed:
                    brain.save_memory(cat, key, val)
            episode = analysis.get("episode")
            if episode and episode.get("summary"):
                brain.add_episode(
                    summary=_sanitize(episode["summary"], max_len=300),
                    topics=episode.get("topics", []),
                    commands=list(dict.fromkeys(commands_executed or [])),
                    emotional_tone=episode.get("emotional_tone", ""),
                )
            if brain.should_consolidate():
                self._run_consolidation()
        except Exception as e:
            logger.debug(f"Session summarizer failed (non-critical): {e}")

    def _run_consolidation(self):
        try:
            brain = get_brain()
            memories = brain.get_all_memories()
            episodes = brain.get_recent_episodes(limit=10)
            gaps = brain.get_knowledge_gaps()
            narrative = self.llm.generate_identity_narrative(memories, episodes, gaps)
            if narrative:
                brain.update_identity_narrative(narrative)
            unconsolidated = brain.get_unconsolidated_episodes()
            if unconsolidated:
                brain.mark_episodes_consolidated([e["id"] for e in unconsolidated])
            brain.compact()
        except Exception as e:
            logger.debug(f"Consolidation failed (non-critical): {e}")
