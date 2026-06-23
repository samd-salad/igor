"""Locks the prompt's anti-regurgitation + anti-upsell rules and the
<current_time> injection. Regressions here directly degrade the user-facing
voice experience."""
from datetime import datetime, UTC

from server.cognition.contracts import Fact, VoiceTurn
from server.cognition._internal.prompt_builder import (
    build_system_prompt,
    build_user_context,
)


def _turn(text: str = "what time is it") -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1",
        started_at=datetime(2026, 6, 17, 6, 17, tzinfo=UTC),
        device_id=None, room=None,
        input_text=text,
        speaker_id=None, metadata={},
    )


def test_system_prompt_forbids_followup_offers():
    p = build_system_prompt("Sam is a software engineer.")
    assert "follow-up offers" in p.lower() or "want me to help" in p.lower()


def test_system_prompt_forbids_memory_regurgitation():
    p = build_system_prompt("Sam wakes up at 8 AM on weekdays.")
    assert "background context" in p.lower()
    assert "never recite" in p.lower() or "do not recite" in p.lower() \
        or "NEVER recite" in p


def test_system_prompt_tells_claude_to_use_current_time():
    p = build_system_prompt("anything")
    assert "<current_time>" in p
    assert "authoritative" in p.lower() or "use it" in p.lower()


def test_system_prompt_includes_identity_narrative():
    p = build_system_prompt("Sam loves dark roast coffee.")
    assert "<my_person>" in p
    assert "Sam loves dark roast coffee." in p


def test_system_prompt_keeps_sardonic_personality():
    """Igor's namesake. Don't accidentally sand off the personality."""
    p = build_system_prompt("anything")
    assert "sardonic" in p.lower()
    assert "dark humor" in p.lower() or "existential" in p.lower()
    assert "no groveling" in p.lower()


def test_system_prompt_forbids_follow_up_questions():
    """The 'gaming, or just awake?' kind of probe reads as nagging."""
    p = build_system_prompt("anything")
    assert "no follow-up questions" in p.lower() or \
           "don't probe" in p.lower() or \
           "don't ask back" in p.lower()


def test_system_prompt_softens_time_framing():
    """At 2 AM Friday, the user often thinks Thursday night. Don't lead
    with the day."""
    p = build_system_prompt("anything")
    lowered = p.lower()
    assert "day" in lowered
    assert "thursday" in lowered or "friday" in lowered or "day-of-week" in lowered


def test_system_prompt_has_ambient_speech_defense():
    """When the mic catches TV/podcast/other-conversation audio, Claude
    must have a way to elect silence."""
    p = build_system_prompt("anything")
    assert "<ambient_speech>" in p
    assert "[silent]" in p
    assert "tv" in p.lower() or "podcast" in p.lower() or "music" in p.lower()


def test_user_context_injects_current_time():
    ctx = build_user_context(_turn(), [], [])
    assert "<current_time>" in ctx
    assert "2026-06-17" in ctx


def test_user_context_keeps_user_text_last():
    ctx = build_user_context(_turn("turn off the lights"), [], [])
    assert ctx.rstrip().endswith("turn off the lights")


def test_user_context_renders_memories_when_present():
    fact = Fact(
        fact_id="f-1", category="schedule", key="wake_time",
        value="8 AM weekdays",
        tags=["schedule", "morning"], source_episode_id=None,
        embedding=None,
        valid_at=datetime(2026, 1, 1, tzinfo=UTC),
        invalid_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    ctx = build_user_context(_turn(), [fact], [])
    assert "<relevant_memories>" in ctx
    assert "8 AM weekdays" in ctx
