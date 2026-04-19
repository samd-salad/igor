"""Tier 1 intent router. Maps short unambiguous phrases to direct commands.

Saves ~1.5-2s latency for the most common voice commands by skipping the LLM entirely.
Only matches phrases where there is ZERO ambiguity. Everything else falls through to LLM.

Two matching passes:
  1. Exact match — normalized phrase looked up in _DIRECT_COMMANDS dict.
  2. Pattern match — keyword-based matchers (OVOS Adapt-style) for common variations
     like "can you pause", "volume to 50", "turn the living room lights off please".

Safety rule: pattern matchers must bail out when extra context words suggest the user
wants something more nuanced (color, brightness, zone, etc.) that only the LLM can handle.
"""
import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Tier1Match:
    command: str           # Command name to execute
    params: Dict[str, Any] # Command parameters
    response: str          # Pre-baked spoken response (skips LLM)


# Normalized phrase -> (command_name, params, spoken_response)
# All commands route through HA-backed implementations in light_cmd / media_cmd / tv_cmd.
_DIRECT_COMMANDS = {
    "pause":               ("play_pause", {}, "Paused."),
    "play":                ("play_pause", {}, "Playing."),
    "resume":              ("play_pause", {}, "Resuming."),
    "stop":                ("play_pause", {}, "Stopped."),
    "next":                ("next_track", {}, "Next."),
    "skip":                ("next_track", {}, "Skipped."),
    "previous":            ("previous_track", {}, "Previous."),
    "lights off":          ("set_light", {"power": "off"}, "Lights off."),
    "lights on":           ("set_light", {"power": "on"}, "Lights on."),
    "turn off the lights": ("set_light", {"power": "off"}, "Lights off."),
    "turn on the lights":  ("set_light", {"power": "on"}, "Lights on."),
    "kill the lights":     ("set_light", {"power": "off"}, "Lights off."),
    "mute":                ("mute", {}, "Muted."),
    "unmute":              ("unmute", {}, "Unmuted."),
}

# Playback-only subset of _DIRECT_COMMANDS — used by _match_playback_filler
# to avoid matching "mute"/"unmute"/"lights on"/etc. through the wrong matcher.
_PLAYBACK_COMMANDS = frozenset({
    "pause", "play", "resume", "stop", "next", "skip", "previous",
})


# ---------------------------------------------------------------------------
# Pattern matchers — second pass when exact match fails.
# Each function takes a normalized string and returns Tier1Match or None.
# ---------------------------------------------------------------------------

_PATTERN_MATCHERS: List[Callable[[str], Optional[Tier1Match]]] = []

_FILLER_WORDS = frozenset({
    "can", "you", "could", "please", "go", "ahead", "and", "it", "the", "that",
    "would", "hey", "igor", "just",
})

# Words that signal the user wants a light attribute (color, brightness, temp)
# rather than a simple on/off toggle. Bail out to LLM when any of these appear.
_LIGHT_ATTRIBUTE_WORDS = frozenset({
    "blue", "red", "green", "yellow", "orange", "purple", "pink", "white",
    "warm", "cool", "cold", "dim", "bright", "brighter", "dimmer",
    "color", "colour", "temperature", "temp", "percent", "low", "high",
})

# Room/zone words — when these appear with "light", the user likely wants a specific
# room's lights. Bail to LLM since Tier 1 can't pass a label parameter.
_ZONE_WORDS = frozenset({
    "living", "bedroom", "kitchen", "bathroom", "office", "dining",
    "hallway", "garage", "basement", "upstairs", "downstairs",
})

# Negation words — bail to LLM when present (user means the opposite of the command)
_NEGATION_WORDS = frozenset({"don't", "dont", "not", "never", "stop"})


def _pattern(fn: Callable[[str], Optional[Tier1Match]]):
    """Decorator to register a pattern matcher."""
    _PATTERN_MATCHERS.append(fn)
    return fn


@_pattern
def _match_lights(text: str) -> Optional[Tier1Match]:
    """Match light on/off commands with filler words.

    Examples that match: "turn the lights off please", "shut off the lights",
    "kill the lights"

    Examples that bail to LLM: "turn on the light blue", "set the lights to 50",
    "lights on warm", "living room lights off" (needs label param)
    """
    words = text.split()
    has_light = any(w in ("light", "lights") for w in words)
    if not has_light:
        return None

    # Bail if any word suggests a color/brightness/temperature attribute —
    # the user wants more than a simple on/off toggle.
    if any(w in _LIGHT_ATTRIBUTE_WORDS for w in words):
        return None

    # Bail if a zone/room name is present — LLM needs to pass the label param
    if any(w in _ZONE_WORDS for w in words):
        return None

    # Bail if a number is present (brightness level like "lights on 50")
    if re.search(r'\b\d+\b', text):
        return None

    if any(w in ("off", "kill", "shut") for w in words):
        return Tier1Match("set_light", {"power": "off"}, "Lights off.")
    if any(w in ("on",) for w in words):
        return Tier1Match("set_light", {"power": "on"}, "Lights on.")
    return None


@_pattern
def _match_playback_filler(text: str) -> Optional[Tier1Match]:
    """Match playback commands with filler words.

    Examples: "can you pause", "please resume", "go ahead and play"

    Only matches playback verbs (pause/play/resume/stop/next/skip/previous),
    not mute/lights — those have their own dedicated matchers.
    """
    words = text.split()
    core = [w for w in words if w not in _FILLER_WORDS]
    if len(core) != 1:
        return None
    # Only match playback commands, not mute/lights/other entries
    if core[0] not in _PLAYBACK_COMMANDS:
        return None
    cmd_name, params, response = _DIRECT_COMMANDS[core[0]]
    return Tier1Match(cmd_name, params, response)


@_pattern
def _match_volume_set(text: str) -> Optional[Tier1Match]:
    r"""Match "volume to N" / "set volume at N" / "volume N".

    Only matches when a number 0-100 is present alongside "volume".
    Bails to LLM when a zone name is present (e.g. "bedroom volume to 40")
    since the Tier 1 path can't pass a zone parameter.
    """
    words = text.split()
    if "volume" not in words:
        return None
    # Bail if a zone name is present — LLM needs to route to the right zone
    if any(w in ("bedroom", "kitchen", "bathroom", "office", "dining") for w in words):
        return None
    m = re.search(r'\b(\d{1,3})\b', text)
    if not m:
        return None
    level = int(m.group(1))
    if not 0 <= level <= 100:
        return None
    return Tier1Match("set_volume", {"level": level, "label": "music"}, f"Volume {level}.")


@_pattern
def _match_mute(text: str) -> Optional[Tier1Match]:
    """Match mute/unmute with filler.

    Examples: "mute the tv", "unmute please", "mute the sound", "tv mute"
    Bails on negation: "don't mute" falls through to LLM.
    """
    words = text.split()
    # Bail on negation — "don't mute" means the opposite
    if any(w in _NEGATION_WORDS for w in words):
        return None
    core = [w for w in words if w not in _FILLER_WORDS]
    if not core:
        return None
    # Check anywhere in core, not just first position ("tv mute" should match)
    if "unmute" in core:
        return Tier1Match("unmute", {}, "Unmuted.")
    if "mute" in core:
        return Tier1Match("mute", {}, "Muted.")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cacheable_responses() -> set:
    """Return set of all pre-baked response strings used by Tier 1 commands."""
    return {response for _, _, response in _DIRECT_COMMANDS.values()}


def route(transcription: str) -> Optional[Tier1Match]:
    """Return Tier1Match if matched, None to fall through to LLM."""
    words = transcription.split()
    if len(words) > 10:
        return None  # Too long for Tier 1 — let LLM handle nuance

    normalized = transcription.lower().strip().rstrip(".!?,")

    # Pass 1: exact match
    match = _DIRECT_COMMANDS.get(normalized)
    if match:
        cmd_name, params, response = match
        logger.info(f"Tier 1 exact match: '{normalized}' -> {cmd_name}({params})")
        return Tier1Match(command=cmd_name, params=params, response=response)

    # Pass 2: pattern matchers
    for matcher in _PATTERN_MATCHERS:
        result = matcher(normalized)
        if result is not None:
            logger.info(f"Tier 1 pattern match: '{normalized}' -> {result.command}({result.params})")
            return result

    return None
