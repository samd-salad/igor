"""Room configuration loaded from data/rooms.yaml.

Each room defines what devices exist (Sonos zone, TV, lights, etc.).
If rooms.yaml doesn't exist, a default room is auto-generated from
server/config.py constants for backward compatibility.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoomConfig:
    """Immutable configuration for a single room."""
    room_id: str
    display_name: str
    sonos_zone: Optional[str] = None
    indicator_light: Optional[str] = None
    tv_host: Optional[str] = None
    light_group: list[str] = field(default_factory=list)
    default_volume_target: str = "local"  # "sonos" or "local"
    text_only: bool = False

    @property
    def has_tv(self) -> bool:
        return self.tv_host is not None

    @property
    def has_sonos(self) -> bool:
        return self.sonos_zone is not None


def load_rooms(yaml_path: Path) -> dict[str, RoomConfig]:
    """Load room configs from YAML file. Returns dict keyed by room_id."""
    if not yaml_path.exists():
        logger.info(f"No rooms.yaml found at {yaml_path}, using default room from config.py")
        default = make_default_room()
        return {default.room_id: default}

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; using default room. Run: pip install pyyaml")
        default = make_default_room()
        return {default.room_id: default}

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load rooms.yaml: {e}; using default room")
        default = make_default_room()
        return {default.room_id: default}

    rooms_data = data.get('rooms', {})
    if not rooms_data:
        logger.warning("rooms.yaml has no 'rooms' key; using default room")
        default = make_default_room()
        return {default.room_id: default}

    rooms = {}
    for room_id, cfg in rooms_data.items():
        rooms[room_id] = RoomConfig(
            room_id=room_id,
            display_name=cfg.get('display_name', room_id),
            sonos_zone=cfg.get('sonos_zone'),
            indicator_light=cfg.get('indicator_light'),
            tv_host=cfg.get('tv_host'),
            light_group=cfg.get('light_group', []),
            default_volume_target=cfg.get('default_volume_target', 'local'),
            text_only=cfg.get('text_only', False),
        )

    logger.info(f"Loaded {len(rooms)} room(s): {', '.join(rooms.keys())}")
    return rooms


def make_default_room() -> RoomConfig:
    """Create a default room from current server/config.py constants."""
    from server.config import SONOS_DEFAULT_ZONE, GOOGLE_TV_HOST, LIGHT_GROUPS

    # Use living room light group if available
    light_group = LIGHT_GROUPS.get('living room', [])

    return RoomConfig(
        room_id='default',
        display_name='Living Room',
        sonos_zone=SONOS_DEFAULT_ZONE,
        indicator_light=None,
        tv_host=GOOGLE_TV_HOST if GOOGLE_TV_HOST and '0.X' not in GOOGLE_TV_HOST else None,
        light_group=light_group,
        default_volume_target='sonos',
    )
