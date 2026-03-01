"""Sonos soundbar/speaker volume control via SoCo (local LAN, no API key).

Use these commands when the user refers to TV volume, music volume, soundbar,
or room speakers. Use Pi system volume commands only when the user says
'your volume' or 'assistant volume'.
"""
import logging
import threading
import time

from .base import Command
from ._utils import parse_amount, parse_direction_updown, parse_volume_word
from server.config import SONOS_DISCOVERY_CACHE_TTL, SONOS_DEFAULT_ZONE

logger = logging.getLogger(__name__)

try:
    import soco
    _SOCO_AVAILABLE = True
except ImportError:
    _SOCO_AVAILABLE = False
    logger.warning("soco not installed; Sonos commands unavailable. Run: pip install soco")

_cache: list = []
_cache_time: float = 0.0
_cache_lock = threading.Lock()

_VOL_S, _VOL_M, _VOL_L = 5, 15, 30


def _get_devices(force: bool = False) -> list:
    global _cache, _cache_time
    now = time.monotonic()
    with _cache_lock:
        if not force and _cache and (now - _cache_time) < SONOS_DISCOVERY_CACHE_TTL:
            return list(_cache)
        try:
            _cache = list(soco.discover() or [])
            _cache_time = now
            logger.debug(f"Sonos discovered {len(_cache)} device(s)")
        except Exception as e:
            logger.warning(f"Sonos discovery failed: {e}")
            _cache = []
            _cache_time = now
        return list(_cache)


def _resolve_target(zone: str) -> list:
    devices = _get_devices()
    if not devices:
        return []
    target = (zone.strip() if zone.strip() else SONOS_DEFAULT_ZONE).lower()
    if target == "all":
        return devices
    return [d for d in devices if d.player_name.lower() == target]


def _apply_to_targets(zone: str, action, action_desc: str) -> str:
    if not _SOCO_AVAILABLE:
        return "Sonos unavailable: soco not installed"
    targets = _resolve_target(zone)
    if not targets:
        effective = zone.strip() if zone.strip() else SONOS_DEFAULT_ZONE
        return f"No Sonos devices found for zone '{effective}'"
    errors = []
    for device in targets:
        try:
            action(device)
        except Exception as e:
            errors.append(str(e))
    target_desc = f"'{zone}'" if zone else f"all {len(targets)} zone(s)"
    if errors:
        return f"{action_desc} {target_desc} — {len(errors)} error(s): {errors[0]}"
    return f"{action_desc} {target_desc}"


class SetSonosVolumeCommand(Command):
    name = "set_sonos_volume"
    description = (
        "Set the Sonos soundbar/speaker volume. "
        "Use this when the user mentions TV volume, music volume, soundbar, or room speakers. "
        "Accepts 0-100 or words: quiet, low, medium, loud, max."
    )

    @property
    def parameters(self) -> dict:
        return {
            "level": {
                "type": "string",
                "description": "Volume level: 0-100 or a word (quiet, low, medium, loud, max)"
            },
            "zone": {
                "type": "string",
                "description": "Sonos zone/player name (e.g. 'Living Room', 'Bedroom'). Omit for default (Living Room). Use 'all' for all zones."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["level"]

    def execute(self, level, zone: str = "") -> str:
        level_str = str(level).lower().strip()
        word_val = parse_volume_word(level_str)
        if word_val is not None:
            level_int = word_val
        else:
            try:
                level_int = max(0, min(100, int(float(level_str.rstrip("%")))))
            except ValueError:
                return f"Couldn't understand volume '{level}'. Try a number or: quiet, low, medium, loud, max."

        def action(device):
            device.volume = level_int

        return _apply_to_targets(zone, action, f"Set Sonos volume to {level_int}% on")


class AdjustSonosVolumeCommand(Command):
    name = "adjust_sonos_volume"
    description = (
        "Increase or decrease the Sonos soundbar/speaker volume relative to its current level. "
        "Use this when the user says 'louder', 'quieter', 'turn it up/down', etc. "
        "referring to TV, music, soundbar, or room audio."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {
                "type": "string",
                "description": "Direction: 'up' / 'louder' to increase, 'down' / 'quieter' to decrease"
            },
            "amount": {
                "type": "string",
                "description": "Step size: 'slightly', 'medium' (default), or 'a lot'"
            },
            "zone": {
                "type": "string",
                "description": "Sonos zone/player name (e.g. 'Living Room', 'Bedroom'). Omit for default (Living Room). Use 'all' for all zones."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", zone: str = "") -> str:
        d = parse_direction_updown(direction)
        if d is None:
            return f"Unknown direction '{direction}'. Use 'up'/'louder' or 'down'/'quieter'."
        up = d == "up"
        step = parse_amount(amount, _VOL_S, _VOL_M, _VOL_L)

        delta = step if up else -step

        def action(device):
            device.set_relative_volume(delta)

        desc = f"{'Increased' if up else 'Decreased'} Sonos volume by {step}% on"
        return _apply_to_targets(zone, action, desc)


class SonosMuteCommand(Command):
    name = "sonos_mute"
    description = (
        "Mute, unmute, or toggle the Sonos soundbar/speaker. "
        "Use when the user says mute/unmute referring to TV, music, or soundbar audio."
    )

    @property
    def parameters(self) -> dict:
        return {
            "state": {
                "type": "string",
                "description": "Mute action: 'on' (mute), 'off' (unmute), or 'toggle'"
            },
            "zone": {
                "type": "string",
                "description": "Sonos zone/player name (e.g. 'Living Room', 'Bedroom'). Omit for default (Living Room). Use 'all' for all zones."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["state"]

    def execute(self, state: str, zone: str = "") -> str:
        state_lower = state.lower().strip()
        if state_lower not in ("on", "off", "toggle"):
            return f"Unknown mute state '{state}'. Use: on, off, or toggle."

        def action(device):
            if state_lower == "toggle":
                device.mute = not device.mute
            else:
                device.mute = (state_lower == "on")

        label = {"on": "Muted", "off": "Unmuted", "toggle": "Toggled mute on"}[state_lower]
        return _apply_to_targets(zone, action, f"{label} Sonos")


class ListSonosCommand(Command):
    name = "list_sonos"
    description = "List all discovered Sonos zones and their current volume"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        if not _SOCO_AVAILABLE:
            return "Sonos unavailable: soco not installed"
        devices = _get_devices(force=True)
        if not devices:
            return "No Sonos devices found on the network"
        lines = []
        for d in devices:
            try:
                lines.append(f"- '{d.player_name}' vol={d.volume}% muted={d.mute}")
            except Exception as e:
                lines.append(f"- (unreadable: {e})")
        return "Sonos zones:\n" + "\n".join(lines)
