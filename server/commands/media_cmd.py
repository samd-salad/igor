"""HA-backed media_player commands. Replaces sonos_cmd.py and the
Pi-side volume RPC parts of system_cmd.py.

Targets are HA media_player entities. Resolution:
  - Empty label + room context → prefer speaker (device_class=speaker) in
    the room's HA area, falling back to first media_player.
  - Keyword hints: 'tv'/'television' → device_class=tv;
    'music'/'speaker'/'sonos' → device_class=speaker.
  - Otherwise: friendly-name exact, then substring match.

HA media_player service field names (verified against /api/services on the
live instance, 2026-04-18):
  volume_set     → volume_level (float 0..1)
  volume_mute    → is_volume_muted (bool)
  play_media     → media_content_id, media_content_type
  media_*        → entity_id only
"""
import logging
from typing import Optional

from .base import Command
from ._utils import parse_amount, parse_direction_updown
from server.ha_client import HAError, get_client

logger = logging.getLogger(__name__)


# Volume step sizes (percentage points)
_VOL_S, _VOL_M, _VOL_L = 5, 10, 20

# Keyword → device_class hint
_TV_WORDS = {"tv", "television", "screen", "telly"}
_SPEAKER_WORDS = {"music", "speaker", "sonos", "audio", "stereo"}


def _friendly_name(state: dict) -> str:
    return (state.get("attributes") or {}).get("friendly_name", "") or ""


def _device_class(state: dict) -> str:
    return (state.get("attributes") or {}).get("device_class", "") or ""


def _filter_class(states: list[dict], device_class: str) -> list[dict]:
    return [s for s in states if _device_class(s) == device_class]


def _resolve_media_targets(label: str, _ctx=None, prefer: Optional[str] = None) -> list[str]:
    """Resolve a label + room context into media_player entity_ids.

    prefer: 'speaker' or 'tv' to bias resolution when label is empty/ambiguous.
    """
    ha = get_client()
    all_media = ha.states_in_domain("media_player")
    if not all_media:
        return []

    label_lower = (label or "").strip().lower()

    # Keyword hints take precedence — "music" overrides ambiguity
    keyword_class: Optional[str] = None
    if any(w in label_lower for w in _TV_WORDS):
        keyword_class = "tv"
        # Strip the keyword so further matching doesn't double-count it
        for w in _TV_WORDS:
            label_lower = label_lower.replace(w, "").strip()
    elif any(w in label_lower for w in _SPEAKER_WORDS):
        keyword_class = "speaker"
        for w in _SPEAKER_WORDS:
            label_lower = label_lower.replace(w, "").strip()

    # Scope by area when room context is present. If the room has no media
    # players, return empty — don't grab from other rooms (the user can
    # always disambiguate by area name in their request).
    in_area: list[dict] = all_media
    scoped_to_area = False
    if _ctx is not None and getattr(_ctx, "room", None) is not None:
        ha_area = getattr(_ctx.room, "ha_area", "") or ""
        if ha_area:
            ids = set(ha.entities_in_area(ha_area, domain="media_player"))
            in_area = [s for s in all_media if s["entity_id"] in ids]
            scoped_to_area = True

    # Apply class filter
    target_class = keyword_class or prefer
    if target_class:
        narrowed = _filter_class(in_area, target_class)
        if narrowed:
            in_area = narrowed

    # If label is empty after keyword stripping, return what we have
    if not label_lower:
        return [s["entity_id"] for s in in_area]

    # Area-name match
    for area in ha.get_areas():
        if area.lower() == label_lower:
            ids = set(ha.entities_in_area(area, domain="media_player"))
            scoped = [s["entity_id"] for s in all_media if s["entity_id"] in ids]
            if target_class:
                scoped = [s["entity_id"] for s in all_media
                          if s["entity_id"] in ids and _device_class(s) == target_class]
            return scoped

    # Friendly name exact match
    exact = [s["entity_id"] for s in in_area if _friendly_name(s).lower() == label_lower]
    if exact:
        return exact

    # Substring match on friendly_name or entity_id
    return [
        s["entity_id"] for s in in_area
        if label_lower in _friendly_name(s).lower() or label_lower in s["entity_id"].lower()
    ]


def _call_media(service: str, data: dict, label: str, _ctx, action_desc: str,
                prefer: Optional[str] = None) -> str:
    targets = _resolve_media_targets(label, _ctx=_ctx, prefer=prefer)
    if not targets:
        desc = f"'{label}'" if label else "any media player"
        return f"No media player found for {desc}"
    payload = {"entity_id": targets, **data}
    try:
        get_client().call_service("media_player", service, payload)
    except HAError as e:
        logger.warning(f"media_player.{service} failed: {e}")
        return f"Failed: {e}"
    target_desc = f"'{label}'" if label else f"{len(targets)} player(s)"
    return f"{action_desc} {target_desc}"


# -- commands --

class SetVolumeCommand(Command):
    name = "set_volume"
    description = (
        "Set the volume on a speaker or TV (0-100). "
        "Defaults to the speaker in the current room."
    )

    @property
    def parameters(self) -> dict:
        return {
            "level": {"type": "integer", "description": "Volume 0-100"},
            "label": {"type": "string", "description": "'tv', 'music', area name, or specific player name. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return ["level"]

    def execute(self, level: int, label: str = "", _ctx=None) -> str:
        level = max(0, min(100, int(level)))
        return _call_media(
            "volume_set", {"volume_level": level / 100.0}, label, _ctx,
            f"Set volume to {level}% on", prefer="speaker",
        )


class AdjustVolumeCommand(Command):
    name = "adjust_volume"
    description = (
        "Increase or decrease the volume on a speaker or TV relative to its current level. "
        "Use for 'turn it up', 'a bit louder', 'much quieter', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {"type": "string", "description": "'up'/'louder' or 'down'/'quieter'"},
            "amount": {"type": "string", "description": "'slightly', 'medium' (default), 'a lot'"},
            "label": {"type": "string", "description": "'tv', 'music', area, or specific player. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
        d = parse_direction_updown(direction.replace("louder", "up").replace("quieter", "down"))
        if d is None:
            return f"Unknown direction '{direction}'. Use 'up'/'down' or 'louder'/'quieter'."

        # Use volume_up/volume_down — works on every device (Sonos, AndroidTV via
        # ADB key events, Cast, etc.). volume_set fails on integrations that only
        # support relative steps.
        # Repeat the step N times based on amount (each call is one device step).
        step_count = parse_amount(amount, 1, 3, 6)
        service = "volume_up" if d == "up" else "volume_down"

        targets = _resolve_media_targets(label, _ctx=_ctx, prefer="speaker")
        if not targets:
            return f"No media player found for {f"'{label}'" if label else 'current room'}"
        ha = get_client()
        applied = 0
        for eid in targets:
            try:
                for _ in range(step_count):
                    ha.call_service("media_player", service, {"entity_id": eid})
                applied += 1
            except HAError as e:
                logger.warning(f"adjust_volume {service} on {eid} failed: {e}")
        if applied == 0:
            return "Volume adjust failed"
        verb = "Increased" if d == "up" else "Decreased"
        return f"{verb} volume ({amount}) on {applied} player(s)"


class GetVolumeCommand(Command):
    name = "get_volume"
    description = "Get the current volume level of a speaker or TV"

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "'tv', 'music', area, or specific player. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        targets = _resolve_media_targets(label, _ctx=_ctx, prefer="speaker")
        if not targets:
            return f"No media player found for {f"'{label}'" if label else 'current room'}"
        ha = get_client()
        out = []
        for eid in targets:
            try:
                a = ha.get_state(eid).get("attributes", {})
                vol = a.get("volume_level")
                muted = a.get("is_volume_muted")
                pct = f"{int(round(vol * 100))}%" if isinstance(vol, (int, float)) else "?"
                m = " (muted)" if muted else ""
                out.append(f"{a.get('friendly_name') or eid}: {pct}{m}")
            except HAError as e:
                logger.warning(f"get_volume on {eid} failed: {e}")
        return "; ".join(out) if out else "Volume unknown"


class MuteCommand(Command):
    name = "mute"
    description = "Mute a speaker or TV"

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "'tv', 'music', area, or specific player. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        return _call_media(
            "volume_mute", {"is_volume_muted": True}, label, _ctx, "Muted",
            prefer="speaker",
        )


class UnmuteCommand(Command):
    name = "unmute"
    description = "Unmute a speaker or TV"

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "'tv', 'music', area, or specific player. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        return _call_media(
            "volume_mute", {"is_volume_muted": False}, label, _ctx, "Unmuted",
            prefer="speaker",
        )


class PlayPauseCommand(Command):
    name = "play_pause"
    description = "Toggle play/pause on a media player. Use for 'pause', 'resume', 'play'."

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "'tv', 'music', area, or specific player. Omit for current room media."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        return _call_media("media_play_pause", {}, label, _ctx, "Toggled play/pause on")


class NextTrackCommand(Command):
    name = "next_track"
    description = "Skip to the next track on a media player"

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "Player name. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        return _call_media("media_next_track", {}, label, _ctx, "Skipped to next track on", prefer="speaker")


class PreviousTrackCommand(Command):
    name = "previous_track"
    description = "Go to the previous track on a media player"

    @property
    def parameters(self) -> dict:
        return {
            "label": {"type": "string", "description": "Player name. Omit for room speaker."},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, label: str = "", _ctx=None) -> str:
        return _call_media("media_previous_track", {}, label, _ctx, "Skipped to previous track on", prefer="speaker")


class ListMediaPlayersCommand(Command):
    name = "list_media_players"
    description = "List all media players (speakers, TVs) and their state"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        ha = get_client()
        states = ha.states_in_domain("media_player", force_refresh=True)
        if not states:
            return "No media players found"
        lines = []
        for s in states:
            a = s.get("attributes", {})
            cls = _device_class(s) or "media"
            area = ha.area_of(s["entity_id"]) or "(no area)"
            vol = a.get("volume_level")
            vol_s = f" vol={int(round(vol * 100))}%" if isinstance(vol, (int, float)) else ""
            title = a.get("media_title")
            title_s = f' playing "{title}"' if title else ""
            lines.append(f"- {a.get('friendly_name') or s['entity_id']} [{cls}] ({area}) — {s['state']}{vol_s}{title_s}")
        return "Media players:\n" + "\n".join(lines)
