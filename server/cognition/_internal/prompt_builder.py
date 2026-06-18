"""Build the cached system prompt + dynamic context for the LLM."""
from __future__ import annotations
from typing import Iterable

from server.cognition.contracts import VoiceTurn, Fact, Episode


def build_system_prompt(identity_narrative: str) -> str:
    return (
        "You are Igor, a personal voice assistant. Be brief, helpful, and warm.\n"
        "Confirm device actions concisely. Do not narrate what you're about to do.\n"
        "<my_person>\n"
        f"{identity_narrative or '(identity unknown yet)'}\n"
        "</my_person>\n"
    )


def build_user_context(
    turn: VoiceTurn,
    relevant_facts: Iterable[Fact],
    recent_episodes: Iterable[Episode],
) -> str:
    bits = []
    facts_list = list(relevant_facts)
    if facts_list:
        lines = [f"- [{f.category}/{f.key}] {f.value}" for f in facts_list]
        bits.append("<relevant_memories>\n" + "\n".join(lines) + "\n</relevant_memories>")
    episodes_list = list(recent_episodes)
    if episodes_list:
        lines = [f"- {e.occurred_at.isoformat()}: {e.summary or e.raw_utterance[:80]}"
                 for e in episodes_list]
        bits.append("<recent_episodes>\n" + "\n".join(lines) + "\n</recent_episodes>")
    bits.append(turn.input_text)
    return "\n\n".join(bits)
