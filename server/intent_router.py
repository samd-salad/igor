"""Tier 1 intent router. Maps short unambiguous phrases to direct commands.

Saves ~1.5-2s latency for the most common voice commands by skipping the LLM entirely.
Only matches phrases where there is ZERO ambiguity. Everything else falls through to LLM.

Two matching passes:
  1. Exact match — normalized phrase looked up in _DIRECT_COMMANDS dict.
  2. Pattern match — keyword-based matchers (OVOS Adapt-style) for common variations
     like "can you pause", "volume to 50", "turn the living room lights off please".
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
_DIRECT_COMMANDS = {
    "pause":               ("tv_playback", {"action": "pause"}, "Paused."),
    "play":                ("tv_playback", {"action": "play"}, "Playing."),
    "resume":              ("tv_playback", {"action": "play"}, "Resuming."),
    "stop":                ("tv_playback", {"action": "stop"}, "Stopped."),
    "next":                ("tv_playback", {"action": "next"}, "Next."),
    "skip":                ("tv_playback", {"action": "next"}, "Skipped."),
    "previous":            ("tv_playback", {"action": "previous"}, "Previous."),
    "lights off":          ("set_light", {"power": "off"}, "Lights off."),
    "lights on":           ("set_light", {"power": "on"}, "Lights on."),
    "turn off the lights": ("set_light", {"power": "off"}, "Lights off."),
    "turn on the lights":  ("set_light", {"power": "on"}, "Lights on."),
    "kill the lights":     ("set_light", {"power": "off"}, "Lights off."),
    "mute":                ("sonos_mute", {"state": "on"}, "Muted."),
    "unmute":              ("sonos_mute", {"state": "off"}, "Unmuted."),
}


# ---------------------------------------------------------------------------
# Pattern matchers — second pass when exact match fails.
# Each function takes a normalized string and returns Tier1Match or None.
# ---------------------------------------------------------------------------

_PATTERN_MATCHERS: List[Callable[[str], Optional[Tier1Match]]] = []

_FILLER_WORDS = frozenset({
    "can", "you", "could", "please", "go", "ahead", "and", "it", "the", "that",
    "would", "hey", "igor", "just",
})


def _pattern(fn: Callable[[str], Optional[Tier1Match]]):
    """Decorator to register a pattern matcher."""
    _PATTERN_MATCHERS.append(fn)
    return fn


@_pattern
def _match_lights(text: str) -> Optional[Tier1Match]:
    """Match light on/off commands with filler words.

    Examples: "turn the living room lights off", "shut off the lights please",
    "kill the lights", "switch on the light"
    """
    words = text.split()
    has_light = any(w in ("light", "lights") for w in words)
    if not has_light:
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
    """
    words = text.split()
    core = [w for w in words if w not in _FILLER_WORDS]
    if len(core) != 1:
        return None
    match = _DIRECT_COMMANDS.get(core[0])
    if match:
        cmd_name, params, response = match
        return Tier1Match(cmd_name, params, response)
    return None


@_pattern
def _match_volume_set(text: str) -> Optional[Tier1Match]:
    r"""Match "volume to N" / "set volume at N" / "volume N".

    Only matches when a number 0-100 is present alongside "volume".
    """
    words = text.split()
    if "volume" not in words:
        return None
    m = re.search(r'\b(\d{1,3})\b', text)
    if not m:
        return None
    level = int(m.group(1))
    if not 0 <= level <= 100:
        return None
    return Tier1Match("set_sonos_volume", {"level": level}, "Done.")


@_pattern
def _match_mute(text: str) -> Optional[Tier1Match]:
    """Match mute/unmute with filler.

    Examples: "mute the tv", "unmute please", "mute the sound"
    """
    words = text.split()
    core = [w for w in words if w not in _FILLER_WORDS]
    # Allow "mute" + optional non-filler words like "tv", "sound", "speakers"
    if not core:
        return None
    if core[0] == "unmute":
        return Tier1Match("sonos_mute", {"state": "off"}, "Unmuted.")
    if core[0] == "mute":
        return Tier1Match("sonos_mute", {"state": "on"}, "Muted.")
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
