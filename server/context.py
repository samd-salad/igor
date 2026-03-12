"""InteractionContext — per-request context that flows through the entire pipeline.

Created in api.py for each incoming request. Replaces scattered prefer_sonos,
tv_playing, pi_client parameters with a single structured object.
"""
from dataclasses import dataclass
from typing import Optional

from server.rooms import RoomConfig


@dataclass
class InteractionContext:
    """Per-request context flowing through the pipeline."""
    client_id: str
    room: RoomConfig
    client_type: str  # "audio" | "text"
    callback_url: Optional[str] = None  # Pi HTTP URL, None for text clients
    prefer_sonos: bool = False
    tv_state: str = "unknown"  # snapshot at interaction start from RoomState
