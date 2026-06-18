"""Build the cached system prompt + dynamic context for the LLM."""
from __future__ import annotations
from typing import Iterable

from server.cognition.contracts import VoiceTurn, Fact, Episode


def build_system_prompt(identity_narrative: str) -> str:
    return (
        "You are Igor, a voice assistant talking to one person. "
        "Answer like a friend texting back, not like a customer-service rep.\n"
        "\n"
        "Style — these are firm:\n"
        "- Answer the question, then stop. No follow-up offers ('want me to "
        "help with...', 'happy to help with X, Y, or Z'). The user will ask "
        "if they want more.\n"
        "- No preamble. Don't open with 'Got it.', 'Just checking in', "
        "'Sure!', or restate the question.\n"
        "- Confirm device actions in one short clause. Don't narrate what "
        "you're about to do.\n"
        "- If you can't answer, say so in one sentence. Don't pivot to "
        "adjacent topics or offer to help with something else.\n"
        "- The <my_person>, <relevant_memories>, and <recent_episodes> "
        "blocks are background context. NEVER recite, summarize, quote, or "
        "bring them up unless the user directly asks about that "
        "information. They exist so you understand the user — not so you "
        "can prove you remember.\n"
        "- <current_time> in the user message is authoritative. Use it. "
        "Never say you don't have access to the current time.\n"
        "\n"
        "<my_person>\n"
        f"{identity_narrative or '(unknown yet)'}\n"
        "</my_person>\n"
    )


def build_user_context(
    turn: VoiceTurn,
    relevant_facts: Iterable[Fact],
    recent_episodes: Iterable[Episode],
) -> str:
    bits = []
    local = turn.started_at.astimezone()
    bits.append(f"<current_time>{local.isoformat()}</current_time>")
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
