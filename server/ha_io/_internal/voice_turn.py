"""Build a VoiceTurn from an incoming ConversationRequest."""
from __future__ import annotations
import uuid
from datetime import datetime, UTC
from typing import Mapping, Optional

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.ha_io.contracts import ConversationRequest


def build_voice_turn(
    req: ConversationRequest,
    ha_client,
    known_rooms: Mapping[str, RoomConfig],
) -> VoiceTurn:
    return VoiceTurn(
        correlation_id=str(uuid.uuid4()),
        started_at=datetime.now(UTC),
        device_id=req.device_id,
        room=_resolve_room(req.device_id, ha_client, known_rooms),
        input_text=req.text,
        speaker_id=None,
        metadata={"language": req.language,
                  "ha_conversation_id": req.conversation_id},
    )


def _resolve_room(device_id: Optional[str], ha_client,
                  known_rooms: Mapping[str, RoomConfig]) -> RoomConfig:
    ha_area = ""
    if device_id:
        try:
            ha_area = ha_client.area_of_device(device_id) or ""
        except Exception:
            ha_area = ""
    if ha_area:
        for room in known_rooms.values():
            if (room.ha_area or "").lower() == ha_area.lower():
                return room
        return RoomConfig(
            room_id=ha_area.lower().replace(" ", "_"),
            display_name=ha_area, ha_area=ha_area,
        )
    return next(iter(known_rooms.values()))
