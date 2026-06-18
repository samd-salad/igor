from server.cognition.contracts import RoomConfig
from server.ha_io.contracts import ConversationRequest
from server.ha_io._internal.voice_turn import build_voice_turn


class _FakeHAClient:
    def area_of_device(self, device_id):
        return "Office" if device_id else ""


def _office():
    return RoomConfig(room_id="office", display_name="Office", ha_area="Office")


def _default():
    return RoomConfig(room_id="default", display_name="Default")


def test_minted_correlation_id_is_uuid_like():
    req = ConversationRequest(text="hi", device_id="dev-1", language="en")
    turn = build_voice_turn(req, _FakeHAClient(), known_rooms={"office": _office()})
    assert len(turn.correlation_id) >= 16
    assert turn.input_text == "hi"
    assert turn.room.room_id == "office"


def test_unknown_device_falls_back_to_default_room():
    req = ConversationRequest(text="hi", device_id=None, language="en")
    turn = build_voice_turn(req, _FakeHAClient(), known_rooms={"default": _default()})
    assert turn.room.room_id == "default"
