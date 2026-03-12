"""LIFX smart bulb commands via lifxlan (local LAN UDP, no API key)."""
import colorsys
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
# Thread pool for parallel UDP commands (one thread per bulb, max 16)
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="lifx")

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


def _safe_label(light) -> str:
    try:
        return light.get_label().lower()
    except Exception:
        return ""


def _resolve_target(label: str, _ctx=None) -> list:
    """Resolve light target from label, room context, or config groups.

    Resolution order:
      1. If label is empty and _ctx has a room light_group → use that
      2. Configured room group (from LIGHT_GROUPS or rooms.yaml)
      3. Group name embedded in label ("lights in the living room")
      4. Exact label match
      5. Substring match
    """
    lights = _get_lights()
    if not lights:
        return []

    # Default to room's light group when no label specified
    if not label and _ctx and hasattr(_ctx, 'room') and _ctx.room.light_group:
        group_set = {g.lower() for g in _ctx.room.light_group}
        labeled = [(l, _safe_label(l)) for l in lights]
        matched = [l for l, lbl in labeled if lbl in group_set]
        if matched:
            return matched

    if not label:
        return lights
    label_lower = label.lower().strip()
    labeled = [(l, _safe_label(l)) for l in lights]

    # 1. Configured room group (exact) — check rooms.yaml groups first
    # Build combined groups from config + all rooms
    from server.config import LIGHT_GROUPS
    combined_groups = dict(LIGHT_GROUPS)  # start with config groups

    if label_lower in combined_groups:
        group_set = {g.lower() for g in combined_groups[label_lower]}
        return [l for l, lbl in labeled if lbl in group_set]

    # 2. Group name embedded in label ("lights in the living room")
    for group_name, members in combined_groups.items():
        if group_name in label_lower:
            group_set = {g.lower() for g in members}
            return [l for l, lbl in labeled if lbl in group_set]

    # 3. Exact label match
    exact = [l for l, lbl in labeled if lbl == label_lower]
    if exact:
        return exact

    # 4. Substring match ("office" → "office lamp")
    return [l for l, lbl in labeled if label_lower in lbl or lbl in label_lower]


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


def _apply_to_targets(label: str, action, action_desc: str, _ctx=None) -> str:
    """Resolve bulbs, apply action to each in parallel, return status string."""
    if not _LIFXLAN_AVAILABLE:
        return "LIFX unavailable: lifxlan not installed"
    targets = _resolve_target(label, _ctx=_ctx)
    if not targets:
        desc = f"'{label}'" if label else "any LIFX bulbs"
        return f"No lights found for {desc}"
    errors = []
    if len(targets) == 1:
        try:
            action(targets[0])
        except Exception as e:
            errors.append(str(e))
    else:
        futures = [(light, _executor.submit(action, light)) for light in targets]
        for _, fut in futures:
            try:
                fut.result(timeout=5.0)
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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["power"]

    def execute(self, power: str, label: str = "", _ctx=None) -> str:
        power = power.lower().strip()
        if power not in ("on", "off"):
            return "Power must be 'on' or 'off'"
        return _apply_to_targets(label, lambda l: l.set_power(power), f"Turned {power}", _ctx=_ctx)


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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["level"]

    def execute(self, level: int, label: str = "", _ctx=None) -> str:
        level = max(0, min(100, int(level)))
        bri = int(level / 100 * 65535)

        def action(light):
            hsbk = list(light.get_color())
            hsbk[2] = bri
            light.set_color(hsbk)

        return _apply_to_targets(label, action, f"Set brightness to {level}% on", _ctx=_ctx)


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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, label: str = "", _ctx=None) -> str:
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
        return _apply_to_targets(label, lambda l: l.set_color(hsbk), f"Set color to {color} on", _ctx=_ctx)



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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
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
        return _apply_to_targets(label, action, desc, _ctx=_ctx)


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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
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
        return _apply_to_targets(label, action, desc, _ctx=_ctx)


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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
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
        return _apply_to_targets(label, action, desc, _ctx=_ctx)


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
        result = "Found bulbs:\n" + "\n".join(lines)
        from server.config import LIGHT_GROUPS
        if LIGHT_GROUPS:
            groups = "; ".join(f"{k}: {', '.join(v)}" for k, v in LIGHT_GROUPS.items())
            result += f"\nGroups: {groups}"
        return result


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
                "description": "Light label, room group ('living room', 'office'), or omit for all lights."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["temperature"]

    def execute(self, temperature: str, label: str = "", _ctx=None) -> str:
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

        return _apply_to_targets(label, action, f"Set color temp to {temperature} ({kelvin}K) on", _ctx=_ctx)


class SetSceneCommand(Command):
    name = "set_scene"
    description = (
        "Apply a named lighting scene that sets multiple lights at once. "
        "Available scenes: warm mix, bright, evening, movie, focus. "
        "Use this when the user asks for a scene or preset by name."
    )

    @property
    def parameters(self) -> dict:
        return {
            "scene": {
                "type": "string",
                "description": "Scene name (e.g. 'warm mix', 'movie', 'bright', 'evening', 'focus')"
            }
        }

    def execute(self, scene: str) -> str:
        from server.config import LIGHT_SCENES
        scene_lower = scene.lower().strip()

        scene_config = LIGHT_SCENES.get(scene_lower)
        matched_name = scene_lower
        if not scene_config:
            for name, cfg in LIGHT_SCENES.items():
                if scene_lower in name or name in scene_lower:
                    scene_config = cfg
                    matched_name = name
                    break

        if not scene_config:
            available = ", ".join(LIGHT_SCENES.keys())
            return f"Scene '{scene}' not found. Available: {available}"

        lights = _get_lights()
        if not lights:
            return "No lights found"

        # Build (light, label, settings) tasks first (label lookup is usually cached)
        tasks = []
        for light in lights:
            label = _safe_label(light)
            settings = scene_config.get(label) or scene_config.get("*")
            if settings is not None:
                tasks.append((light, label, settings))

        if not tasks:
            return f"No lights matched scene '{matched_name}'"

        applied = []
        applied_lock = threading.Lock()

        def _apply_one(light, label, settings):
            try:
                if settings.get("power") is False:
                    light.set_power(0, duration=500, rapid=False)
                    with applied_lock:
                        applied.append(f"{label} off")
                    return
                light.set_power(65535, duration=0, rapid=True)
                if "kelvin" in settings or "brightness" in settings:
                    hsbk = [0, 0, 65535, 3500]
                    if "brightness" in settings:
                        hsbk[2] = int(settings["brightness"] * 65535)
                    if "kelvin" in settings:
                        hsbk[3] = settings["kelvin"]
                    light.set_color(hsbk, duration=1000)
                with applied_lock:
                    applied.append(label)
            except Exception as e:
                logger.warning(f"Scene '{matched_name}' failed on {label}: {e}")

        if len(tasks) == 1:
            _apply_one(*tasks[0])
        else:
            futures = [_executor.submit(_apply_one, *t) for t in tasks]
            for f in futures:
                try:
                    f.result(timeout=5.0)
                except Exception as e:
                    logger.warning(f"Scene task exception: {e}")

        if not applied:
            return f"No lights matched scene '{matched_name}'"
        return f"Applied scene '{matched_name}': {', '.join(applied)}"


class ListScenesCommand(Command):
    name = "list_scenes"
    description = "List all available named lighting scenes"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        from server.config import LIGHT_SCENES
        if not LIGHT_SCENES:
            return "No scenes configured"
        return "Available scenes: " + ", ".join(LIGHT_SCENES.keys())
