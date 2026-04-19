"""HA-backed light commands. Replaces lifx_cmd.py.

Same intent-shaped surface (set_light / set_brightness / set_color / ...) but
the implementation calls Home Assistant via ha_client. Targeting is by HA
area (`_ctx.room.ha_area`), explicit area name, entity friendly_name, or
substring — falling back to all lights when nothing is specified.

When lifx_cmd.py is deleted, light_cmd.py owns the `set_light`, `set_brightness`,
etc. command names. While both files coexist (during the migration), pkgutil
loads them in alphabetical order so light_cmd wins on collision.
"""
import logging
from typing import Optional

from .base import Command
from ._utils import parse_amount, parse_direction_updown
from server.ha_client import HAError, get_client

logger = logging.getLogger(__name__)


# Brightness step sizes (percentage points)
_BRI_S, _BRI_M, _BRI_L = 10, 25, 50
# Kelvin step sizes
_KELVIN_S, _KELVIN_M, _KELVIN_L = 300, 700, 1500
# Hue step sizes (degrees on HA's hs scale 0-360)
_HUE_S, _HUE_M, _HUE_L = 15, 30, 60

# Map internal preset names to kelvin
COLOR_TEMPS = {"warm": 2700, "neutral": 4000, "cool": 5000, "daylight": 6500}

# Map our color words to HA `color_name` (CSS color names HA accepts)
NAMED_COLORS = {
    "red": "red", "orange": "orange", "yellow": "yellow",
    "green": "green", "blue": "blue", "purple": "purple",
    "pink": "pink", "white": "white",
}

# Hue degrees (0-360) for relative shift_hue
_COLOR_HUE_DEG = {
    "red": 0, "orange": 30, "yellow": 60, "green": 120,
    "blue": 240, "purple": 280, "pink": 320,
}


def _friendly_name(state: dict) -> str:
    return (state.get("attributes") or {}).get("friendly_name", "") or ""


def _resolve_light_targets(label: str, _ctx=None) -> list[str]:
    """Resolve a label + room context into a list of light entity_ids.

    Resolution order:
      1. label empty + _ctx has a room with ha_area → entities in that area
      2. label empty + no ctx → all lights
      3. label matches an HA area name (case-insensitive) → entities in that area
      4. label exactly matches a light's friendly_name → that entity
      5. label is a substring of a friendly_name → matching entities
    """
    ha = get_client()
    all_lights = ha.states_in_domain("light")
    if not all_lights:
        return []

    label = (label or "").strip()
    label_lower = label.lower()

    # 1+2 — empty label
    if not label:
        if _ctx is not None and getattr(_ctx, "room", None) is not None:
            ha_area = getattr(_ctx.room, "ha_area", "") or ""
            if ha_area:
                return ha.entities_in_area(ha_area, domain="light")
        return [s["entity_id"] for s in all_lights]

    # 3 — area name match
    for area in ha.get_areas():
        if area.lower() == label_lower:
            return ha.entities_in_area(area, domain="light")

    # 4 — exact friendly name
    exact = [s["entity_id"] for s in all_lights if _friendly_name(s).lower() == label_lower]
    if exact:
        return exact

    # 5 — substring
    return [
        s["entity_id"] for s in all_lights
        if label_lower in _friendly_name(s).lower() or label_lower in s["entity_id"].lower()
    ]


def _call_lights(service: str, data: dict, label: str, _ctx, action_desc: str) -> str:
    """Resolve targets, call light.{service} on them, return a status string."""
    targets = _resolve_light_targets(label, _ctx=_ctx)
    if not targets:
        desc = f"'{label}'" if label else "any lights"
        return f"No lights found for {desc}"
    payload = {"entity_id": targets, **data}
    try:
        get_client().call_service("light", service, payload)
    except HAError as e:
        logger.warning(f"light.{service} failed: {e}")
        return f"Failed: {e}"
    bulb_desc = f"'{label}'" if label else f"all {len(targets)} light(s)"
    return f"{action_desc} {bulb_desc}"


# -- commands --

class SetLightCommand(Command):
    name = "set_light"
    description = "Turn lights on or off. Defaults to the lights in the current room."

    @property
    def parameters(self) -> dict:
        return {
            "power": {"type": "string", "description": "Power state: 'on' or 'off'"},
            "label": {"type": "string", "description": "Light name, area ('Living Room', 'Office'), or omit for current room."},
        }

    @property
    def required_parameters(self) -> list:
        return ["power"]

    def execute(self, power: str, label: str = "", _ctx=None) -> str:
        power = power.lower().strip()
        if power not in ("on", "off"):
            return "Power must be 'on' or 'off'"
        service = "turn_on" if power == "on" else "turn_off"
        return _call_lights(service, {}, label, _ctx, f"Turned {power}")


class SetBrightnessCommand(Command):
    name = "set_brightness"
    description = "Set light brightness from 0 to 100"

    @property
    def parameters(self) -> dict:
        return {
            "level": {"type": "integer", "description": "Brightness level 0-100"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
        }

    @property
    def required_parameters(self) -> list:
        return ["level"]

    def execute(self, level: int, label: str = "", _ctx=None) -> str:
        level = max(0, min(100, int(level)))
        if level == 0:
            return _call_lights("turn_off", {}, label, _ctx, "Turned off")
        return _call_lights("turn_on", {"brightness_pct": level}, label, _ctx, f"Set brightness to {level}% on")


class AdjustBrightnessCommand(Command):
    name = "adjust_brightness"
    description = (
        "Increase or decrease light brightness relative to its current level. "
        "Use for 'slightly brighter', 'a lot dimmer', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {"type": "string", "description": "'up'/'brighter' or 'down'/'dimmer'"},
            "amount": {"type": "string", "description": "'slightly', 'medium' (default), 'a lot'"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
        d = parse_direction_updown(direction)
        if d is None:
            return f"Unknown direction '{direction}'. Use 'brighter'/'dimmer' or 'up'/'down'."
        step = parse_amount(amount, _BRI_S, _BRI_M, _BRI_L)
        delta = step if d == "up" else -step
        return _call_lights(
            "turn_on", {"brightness_step_pct": delta}, label, _ctx,
            f"{'Increased' if d == 'up' else 'Decreased'} brightness ({amount}) on",
        )


class SetColorCommand(Command):
    name = "set_color"
    description = "Set a light to a named color (red, orange, yellow, green, blue, purple, pink, white) or hex (#rrggbb)"

    @property
    def parameters(self) -> dict:
        return {
            "color": {"type": "string", "description": "Color name or hex value"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, label: str = "", _ctx=None) -> str:
        color_lower = color.lower().strip()
        data: dict
        if color_lower in NAMED_COLORS:
            if color_lower == "white":
                # White via warm-ish kelvin keeps lamps natural
                data = {"color_temp_kelvin": 4000}
            else:
                data = {"color_name": NAMED_COLORS[color_lower]}
        elif color_lower.startswith("#") and len(color_lower) == 7:
            try:
                r = int(color_lower[1:3], 16)
                g = int(color_lower[3:5], 16)
                b = int(color_lower[5:7], 16)
                data = {"rgb_color": [r, g, b]}
            except ValueError:
                return f"Invalid hex color '{color}'. Use #rrggbb format."
        else:
            valid = ", ".join(sorted(NAMED_COLORS))
            return f"Unknown color '{color}'. Use a name ({valid}) or hex (#rrggbb)."
        return _call_lights("turn_on", data, label, _ctx, f"Set color to {color} on")


class SetColorTempCommand(Command):
    name = "set_color_temp"
    description = "Set light color temperature: warm, neutral, cool, or daylight"

    @property
    def parameters(self) -> dict:
        return {
            "temperature": {"type": "string", "description": "warm (~2700K), neutral (~4000K), cool (~5000K), daylight (~6500K)"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
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
        return _call_lights("turn_on", {"color_temp_kelvin": kelvin}, label, _ctx, f"Set color temp to {temperature} ({kelvin}K) on")


class AdjustColorTempCommand(Command):
    name = "adjust_color_temp"
    description = "Shift light color temperature warmer or cooler relative to its current setting."

    @property
    def parameters(self) -> dict:
        return {
            "direction": {"type": "string", "description": "'warmer' or 'cooler'"},
            "amount": {"type": "string", "description": "'slightly', 'medium', 'a lot'"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
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
        delta = -step if warmer else step

        # HA has no native kelvin_step; per-target read+write
        targets = _resolve_light_targets(label, _ctx=_ctx)
        if not targets:
            return f"No lights found for {f"'{label}'" if label else 'current room'}"
        ha = get_client()
        applied = 0
        for eid in targets:
            try:
                state = ha.get_state(eid)
                attrs = state.get("attributes", {})
                current_k = attrs.get("color_temp_kelvin") or attrs.get("kelvin")
                if current_k is None:
                    # Light may be in color mode; default to neutral as the basis
                    current_k = 4000
                new_k = max(1500, min(6500, int(current_k) + delta))
                ha.call_service("light", "turn_on", {"entity_id": eid, "color_temp_kelvin": new_k})
                applied += 1
            except HAError as e:
                logger.warning(f"adjust_color_temp on {eid} failed: {e}")
        if applied == 0:
            return "Color temperature adjust failed"
        return f"Made {'warmer' if warmer else 'cooler'} ({amount}) on {applied} light(s)"


class ShiftHueCommand(Command):
    name = "shift_hue"
    description = "Shift a light's hue toward a named color relative to its current hue."

    @property
    def parameters(self) -> dict:
        return {
            "color": {"type": "string", "description": "Target color: red, orange, yellow, green, blue, purple, pink"},
            "amount": {"type": "string", "description": "'slightly', 'medium', 'a lot'"},
            "label": {"type": "string", "description": "Light name, area, or omit for current room."},
        }

    @property
    def required_parameters(self) -> list:
        return ["color"]

    def execute(self, color: str, amount: str = "medium", label: str = "", _ctx=None) -> str:
        color_lower = color.lower().strip()
        if color_lower not in _COLOR_HUE_DEG:
            valid = ", ".join(sorted(_COLOR_HUE_DEG))
            return f"Unknown color '{color}'. Use: {valid}."
        target_hue = _COLOR_HUE_DEG[color_lower]
        step = parse_amount(amount, _HUE_S, _HUE_M, _HUE_L)

        targets = _resolve_light_targets(label, _ctx=_ctx)
        if not targets:
            return f"No lights found for {f"'{label}'" if label else 'current room'}"
        ha = get_client()
        applied = 0
        for eid in targets:
            try:
                state = ha.get_state(eid)
                attrs = state.get("attributes", {})
                hs = attrs.get("hs_color") or [0, 0]
                current_hue = hs[0]
                # Shortest signed angular distance
                diff = (target_hue - current_hue + 540) % 360 - 180
                move = max(-step, min(step, diff))
                new_hue = (current_hue + move) % 360
                new_sat = max(50.0, hs[1])  # ensure color visible
                ha.call_service("light", "turn_on", {"entity_id": eid, "hs_color": [new_hue, new_sat]})
                applied += 1
            except HAError as e:
                logger.warning(f"shift_hue on {eid} failed: {e}")
        if applied == 0:
            return "Hue shift failed"
        return f"Shifted hue toward {color} ({amount}) on {applied} light(s)"


class ListLightsCommand(Command):
    name = "list_lights"
    description = "List all lights and their state"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        ha = get_client()
        lights = ha.states_in_domain("light", force_refresh=True)
        if not lights:
            return "No lights found in Home Assistant"
        lines = []
        for s in lights:
            name = _friendly_name(s) or s["entity_id"]
            area = ha.area_of(s["entity_id"]) or "(no area)"
            lines.append(f"- {name} ({area}) — {s['state']}")
        return "Lights:\n" + "\n".join(lines)


class SetSceneCommand(Command):
    name = "set_scene"
    description = "Apply a named lighting scene (configured in Home Assistant)."

    @property
    def parameters(self) -> dict:
        return {
            "scene": {"type": "string", "description": "Scene name as defined in HA"},
        }

    def execute(self, scene: str) -> str:
        ha = get_client()
        scene_lower = scene.lower().strip()
        scenes = ha.states_in_domain("scene")
        if not scenes:
            return "No scenes configured in Home Assistant"
        # Try friendly_name match first, then entity_id substring
        match = None
        for s in scenes:
            if _friendly_name(s).lower() == scene_lower:
                match = s["entity_id"]; break
        if not match:
            for s in scenes:
                if scene_lower in _friendly_name(s).lower() or scene_lower in s["entity_id"]:
                    match = s["entity_id"]; break
        if not match:
            available = ", ".join(_friendly_name(s) or s["entity_id"] for s in scenes)
            return f"Scene '{scene}' not found. Available: {available}"
        try:
            ha.call_service("scene", "turn_on", {"entity_id": match})
        except HAError as e:
            return f"Failed to apply scene: {e}"
        return f"Applied scene '{scene}'"


class ListScenesCommand(Command):
    name = "list_scenes"
    description = "List all lighting scenes configured in Home Assistant"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        scenes = get_client().states_in_domain("scene", force_refresh=True)
        if not scenes:
            return "No scenes configured in Home Assistant"
        names = [(_friendly_name(s) or s["entity_id"]) for s in scenes]
        return "Available scenes: " + ", ".join(names)
