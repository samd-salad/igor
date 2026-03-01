"""LIFX smart bulb commands via lifxlan (local LAN UDP, no API key)."""
import colorsys
import logging
import threading
import time

from .base import Command
from ._utils import parse_amount, parse_direction_updown
from server.config import LIFX_DISCOVERY_CACHE_TTL

logger = logging.getLogger(__name__)

try:
    from lifxlan import LifxLAN
    _LIFXLAN_AVAILABLE = True
except ImportError:
    _LIFXLAN_AVAILABLE = False
    logger.warning("lifxlan not installed; LIFX commands unavailable. Run: pip install lifxlan")

# Module-level discovery cache — avoids 1s UDP probe on every command
_lan = None
_cache: list = []
_cache_time: float = 0.0
_cache_lock = threading.Lock()

NAMED_COLORS = {
    "red":    [0,      65535, 65535, 3500],
    "orange": [6553,   65535, 65535, 3500],
    "yellow": [10923,  65535, 65535, 3500],
    "green":  [21845,  65535, 65535, 3500],
    "blue":   [43690,  65535, 65535, 3500],
    "purple": [50938,  65535, 65535, 3500],
    "pink":   [54613,  65535, 65535, 3500],
    "white":  [0,      0,     65535, 5500],
}

COLOR_TEMPS = {
    "warm":     2700,
    "neutral":  4000,
    "cool":     5000,
    "daylight": 6500,
}

# Step sizes (LIFX units, 0-65535)
_BRI_S, _BRI_M, _BRI_L   = 4000, 10000, 20000
_KELVIN_S, _KELVIN_M, _KELVIN_L = 300, 700, 1500
_HUE_S, _HUE_M, _HUE_L   = 2730, 5460, 10920

# Target hue values for color names (LIFX 0-65535)
_COLOR_HUES = {k: v[0] for k, v in NAMED_COLORS.items() if v[1] > 0}  # exclude white


def _get_lights(force: bool = False) -> list:
    global _lan, _cache, _cache_time
    now = time.monotonic()
    with _cache_lock:
        if not force and _cache and (now - _cache_time) < LIFX_DISCOVERY_CACHE_TTL:
            return list(_cache)
        try:
            if _lan is None:
                _lan = LifxLAN()
            _cache = _lan.get_lights()
            _cache_time = now
            logger.debug(f"LIFX discovered {len(_cache)} light(s)")
        except Exception as e:
            logger.warning(f"LIFX discovery failed: {e}")
            _cache = []
            _cache_time = now
        return list(_cache)


def _resolve_target(label: str) -> list:
    lights = _get_lights()
    if not lights:
        return []
    if not label:
        return lights
    label_lower = label.lower().strip()
    try:
        return [light for light in lights if light.get_label().lower() == label_lower]
    except Exception:
        return []


def _hex_to_hsbk(hex_str: str, kelvin: int = 3500) -> list | None:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        return None
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return [int(h * 65535), int(s * 65535), int(v * 65535), kelvin]


def _apply_to_targets(label: str, action, action_desc: str) -> str:
    """Resolve bulbs, apply action to each, return status string."""
    if not _LIFXLAN_AVAILABLE:
        return "LIFX unavailable: lifxlan not installed"
    targets = _resolve_target(label)
    if not targets:
        desc = f"bulb '{label}'" if label else "any LIFX bulbs"
        return f"No lights found for {desc}"
    errors = []
    for light in targets:
        try:
            action(light)
        except Exception as e:
            errors.append(str(e))
    bulb_desc = f"'{label}'" if label else f"all {len(targets)} bulb(s)"
    if errors:
        return f"{action_desc} {bulb_desc} — {len(errors)} error(s): {errors[0]}"
    return f"{action_desc} {bulb_desc}"


class SetLightCommand(Command):
    name = "set_light"
    description = "Turn a LIFX smart bulb on or off"

    @property
    def parameters(self) -> dict:
        return {
            "power": {
                "type": "string",
                "description": "Power state: 'on' or 'off'"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["power"]

    def execute(self, power: str, label: str = "") -> str:
        power = power.lower().strip()
        if power not in ("on", "off"):
            return "Power must be 'on' or 'off'"
        return _apply_to_targets(label, lambda l: l.set_power(power), f"Turned {power}")


class SetBrightnessCommand(Command):
    name = "set_brightness"
    description = "Set LIFX bulb brightness from 0 to 100"

    @property
    def parameters(self) -> dict:
        return {
            "level": {
                "type": "integer",
                "description": "Brightness level 0-100"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["level"]

    def execute(self, level: int, label: str = "") -> str:
        level = max(0, min(100, int(level)))
        bri = int(level / 100 * 65535)

        def action(light):
            hsbk = list(light.get_color())
            hsbk[2] = bri
            light.set_color(hsbk)

        return _apply_to_targets(label, action, f"Set brightness to {level}% on")


class SetColorCommand(Command):
    name = "set_color"
    description = "Set a LIFX bulb to a named color or hex value"

    @property
    def parameters(self) -> dict:
        return {
            "color": {
                "type": "string",
                "description": (
                    "Color name (red, orange, yellow, green, blue, purple, pink, white) "
                    "or hex value (#rrggbb)"
                )
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, label: str = "") -> str:
        color_lower = color.lower().strip()
        if color_lower in NAMED_COLORS:
            hsbk = NAMED_COLORS[color_lower]
        elif color_lower.startswith("#"):
            hsbk = _hex_to_hsbk(color_lower)
            if hsbk is None:
                return f"Invalid hex color '{color}'. Use #rrggbb format."
        else:
            valid = ", ".join(sorted(NAMED_COLORS))
            return f"Unknown color '{color}'. Use a name ({valid}) or hex (#rrggbb)."
        return _apply_to_targets(label, lambda l: l.set_color(hsbk), f"Set color to {color} on")



class AdjustBrightnessCommand(Command):
    name = "adjust_brightness"
    description = (
        "Increase or decrease LIFX bulb brightness relative to its current level. "
        "Use for 'slightly brighter', 'a lot dimmer', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {
                "type": "string",
                "description": "Direction: 'up' / 'brighter' to increase, 'down' / 'dimmer' to decrease"
            },
            "amount": {
                "type": "string",
                "description": "Step size: 'slightly', 'medium' (default), or 'a lot'"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "") -> str:
        d = parse_direction_updown(direction)
        if d is None:
            return f"Unknown direction '{direction}'. Use 'brighter'/'dimmer' or 'up'/'down'."
        up = d == "up"
        step = parse_amount(amount, _BRI_S, _BRI_M, _BRI_L)
        delta = step if up else -step

        def action(light):
            hsbk = list(light.get_color())
            hsbk[2] = max(0, min(65535, hsbk[2] + delta))
            light.set_color(hsbk)

        desc = f"{'Increased' if up else 'Decreased'} brightness {amount} on"
        return _apply_to_targets(label, action, desc)


class AdjustColorTempCommand(Command):
    name = "adjust_color_temp"
    description = (
        "Shift LIFX bulb color temperature warmer or cooler relative to its current setting. "
        "Use for 'slightly warmer', 'a lot cooler', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {
                "type": "string",
                "description": "Direction: 'warmer' / 'warm' to increase warmth, 'cooler' / 'cool' / 'bluer' to increase coolness"
            },
            "amount": {
                "type": "string",
                "description": "Step size: 'slightly', 'medium' (default), or 'a lot'"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "") -> str:
        d = direction.lower().strip()
        warmer = d in ("warmer", "warm", "orange", "yellow", "hot", "cozy", "cosy")
        cooler = d in ("cooler", "cool", "bluer", "blue", "cold", "white", "crisp", "bright")
        if not warmer and not cooler:
            return f"Unknown direction '{direction}'. Use 'warmer' or 'cooler'."
        step = parse_amount(amount, _KELVIN_S, _KELVIN_M, _KELVIN_L)
        delta = -step if warmer else step  # warmer = lower kelvin

        def action(light):
            hsbk = list(light.get_color())
            hsbk[3] = max(1500, min(9000, hsbk[3] + delta))
            light.set_color(hsbk)

        desc = f"Made {'warmer' if warmer else 'cooler'} ({amount}) on"
        return _apply_to_targets(label, action, desc)


class ShiftHueCommand(Command):
    name = "shift_hue"
    description = (
        "Shift a LIFX bulb's hue toward a named color relative to its current hue. "
        "Use for 'a little more blue', 'slightly more purple', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "color": {
                "type": "string",
                "description": "Target color to shift toward: red, orange, yellow, green, blue, purple, pink"
            },
            "amount": {
                "type": "string",
                "description": "Step size: 'slightly', 'medium' (default), or 'a lot'"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, amount: str = "medium", label: str = "") -> str:
        color_lower = color.lower().strip()
        if color_lower not in _COLOR_HUES:
            valid = ", ".join(sorted(_COLOR_HUES))
            return f"Unknown color '{color}'. Use: {valid}."

        step = parse_amount(amount, _HUE_S, _HUE_M, _HUE_L)
        target_hue = _COLOR_HUES[color_lower]

        def action(light):
            hsbk = list(light.get_color())
            current = hsbk[0]
            # Shortest path on the hue circle (0-65535 wraps)
            diff = (target_hue - current) % 65536
            if diff > 32767:
                diff -= 65536
            move = max(-step, min(step, diff))
            hsbk[0] = (current + move) % 65536
            # Ensure saturation is high enough to see the color
            if hsbk[1] < 30000:
                hsbk[1] = 50000
            light.set_color(hsbk)

        desc = f"Shifted hue toward {color} ({amount}) on"
        return _apply_to_targets(label, action, desc)


class ListLightsCommand(Command):
    name = "list_lights"
    description = "List all discovered LIFX bulbs and their labels"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        if not _LIFXLAN_AVAILABLE:
            return "LIFX unavailable: lifxlan not installed"
        lights = _get_lights(force=True)
        if not lights:
            return "No LIFX bulbs found on the network"
        lines = []
        for light in lights:
            try:
                label = light.get_label()
                ip = light.ip_addr
                lines.append(f"- '{label}' ({ip})")
            except Exception as e:
                lines.append(f"- (unreadable: {e})")
        return "Found bulbs:\n" + "\n".join(lines)


class SetColorTempCommand(Command):
    name = "set_color_temp"
    description = "Set LIFX bulb color temperature: warm, neutral, cool, or daylight"

    @property
    def parameters(self) -> dict:
        return {
            "temperature": {
                "type": "string",
                "description": "Temperature preset: warm (~2700K), neutral (~4000K), cool (~5000K), daylight (~6500K)"
            },
            "label": {
                "type": "string",
                "description": "Bulb name/label. Omit to affect all bulbs."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["temperature"]

    def execute(self, temperature: str, label: str = "") -> str:
        temp_lower = temperature.lower().strip()
        if temp_lower not in COLOR_TEMPS:
            valid = ", ".join(sorted(COLOR_TEMPS))
            return f"Unknown temperature '{temperature}'. Use: {valid}."
        kelvin = COLOR_TEMPS[temp_lower]

        def action(light):
            hsbk = list(light.get_color())
            hsbk[1] = 0        # zero saturation for true white
            hsbk[3] = kelvin
            light.set_color(hsbk)

        return _apply_to_targets(label, action, f"Set color temp to {temperature} ({kelvin}K) on")
