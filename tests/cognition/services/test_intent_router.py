from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.services.intent_router import IntentRouter


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_pause_routes_to_tier1():
    res = IntentRouter().route(_turn("pause"))
    assert res is not None
    assert res.command == "play_pause"


def test_unrelated_returns_none():
    assert IntentRouter().route(_turn("tell me a joke about ducks")) is None


def test_volume_pattern_matches():
    res = IntentRouter().route(_turn("volume to 50"))
    assert res is not None
    assert res.command == "set_volume"
    assert res.params["level"] == 50
