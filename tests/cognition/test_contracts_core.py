from datetime import datetime, UTC
import dataclasses
import pytest
from server.cognition.contracts import (
    VoiceTurn, ConversationResult, RoomConfig, ToolCallRecord,
)


def test_voice_turn_is_frozen_dataclass():
    turn = VoiceTurn(
        correlation_id="abc",
        started_at=datetime.now(UTC),
        device_id="dev1",
        room=RoomConfig(room_id="office", display_name="Office", ha_area="Office"),
        input_text="what time is it",
        speaker_id=None,
        metadata={"language": "en"},
    )
    assert dataclasses.is_dataclass(turn)
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.input_text = "different"  # type: ignore


def test_conversation_result_carries_correlation():
    result = ConversationResult(
        correlation_id="abc",
        response_text="hi",
        commands_executed=[],
        end_conversation=True,
    )
    assert result.correlation_id == "abc"


def test_tool_call_record_minimal():
    rec = ToolCallRecord(name="get_time", args={"include_date": True}, result="3:00 PM")
    assert rec.name == "get_time"
    assert rec.args == {"include_date": True}
