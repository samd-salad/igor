from datetime import datetime, UTC

from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.services.intent_router import IntentRouter


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_unrelated_returns_none():
    assert IntentRouter().route(_turn("tell me a joke about ducks")) is None


def test_lights_off_no_longer_short_circuits():
    """Regression: the old Tier 1 emitted `set_light`, which no executor in
    the post-HA-MCP world handles. It returned a canned 'Lights off.' while
    the lights stayed on. The patterns are gone; routing must hand the
    utterance to Tier 2 (LLM) so the real HA tool can be picked."""
    assert IntentRouter().route(_turn("lights off")) is None
    assert IntentRouter().route(_turn("turn off the lights")) is None
    assert IntentRouter().route(_turn("kill the lights")) is None


def test_playback_no_longer_short_circuits():
    assert IntentRouter().route(_turn("pause")) is None
    assert IntentRouter().route(_turn("next")) is None
    assert IntentRouter().route(_turn("skip")) is None


def test_volume_no_longer_short_circuits():
    assert IntentRouter().route(_turn("volume to 50")) is None


def test_mute_no_longer_short_circuits():
    assert IntentRouter().route(_turn("mute")) is None
