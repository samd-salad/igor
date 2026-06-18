from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.services.quality_gate import QualityGate


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_meaningful_request_passes():
    res = QualityGate().filter(_turn("turn off the lights"))
    assert res.text == "turn off the lights"
    assert res.rejected is False


def test_single_filler_rejected():
    res = QualityGate().filter(_turn("um"))
    assert res.rejected is True


def test_hallucination_rejected():
    assert QualityGate().filter(_turn("thank you.")).rejected is True
