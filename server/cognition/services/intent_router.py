"""IntentRouter — VoiceTurn-shaped Tier 1 direct-match router.

Saves ~1.5-2s latency for common voice commands by skipping the LLM.
Two passes: exact match, then pattern matchers. Bails on any ambiguity."""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tier1Match:
    command: str
    params: dict[str, Any]
    response: str


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

_PLAYBACK_COMMANDS = frozenset({"pause", "play", "resume", "stop", "next", "skip", "previous"})

_FILLER_WORDS = frozenset({
    "can", "you", "could", "please", "go", "ahead", "and", "it", "the", "that",
    "would", "hey", "igor", "just",
})

_LIGHT_ATTRIBUTE_WORDS = frozenset({
    "blue", "red", "green", "yellow", "orange", "purple", "pink", "white",
    "warm", "cool", "cold", "dim", "bright", "brighter", "dimmer",
    "color", "colour", "temperature", "temp", "percent", "low", "high",
})

_ZONE_WORDS = frozenset({
    "living", "bedroom", "kitchen", "bathroom", "office", "dining",
    "hallway", "garage", "basement", "upstairs", "downstairs",
})

_NEGATION_WORDS = frozenset({"don't", "dont", "not", "never", "stop"})


def _match_lights(text: str) -> Optional[Tier1Match]:
    words = text.split()
    if not any(w in ("light", "lights") for w in words):
        return None
    if any(w in _LIGHT_ATTRIBUTE_WORDS for w in words):
        return None
    if any(w in _ZONE_WORDS for w in words):
        return None
    if re.search(r"\b\d+\b", text):
        return None
    if any(w in ("off", "kill", "shut") for w in words):
        return Tier1Match("set_light", {"power": "off"}, "Lights off.")
    if "on" in words:
        return Tier1Match("set_light", {"power": "on"}, "Lights on.")
    return None


def _match_playback_filler(text: str) -> Optional[Tier1Match]:
    words = text.split()
    core = [w for w in words if w not in _FILLER_WORDS]
    if len(core) != 1:
        return None
    if core[0] not in _PLAYBACK_COMMANDS:
        return None
    cmd_name, params, response = _DIRECT_COMMANDS[core[0]]
    return Tier1Match(cmd_name, params, response)


def _match_volume_set(text: str) -> Optional[Tier1Match]:
    words = text.split()
    if "volume" not in words:
        return None
    if any(w in ("bedroom", "kitchen", "bathroom", "office", "dining") for w in words):
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    level = int(m.group(1))
    if not 0 <= level <= 100:
        return None
    return Tier1Match("set_volume", {"level": level, "label": "music"}, f"Volume {level}.")


def _match_mute(text: str) -> Optional[Tier1Match]:
    words = text.split()
    if any(w in _NEGATION_WORDS for w in words):
        return None
    core = [w for w in words if w not in _FILLER_WORDS]
    if not core:
        return None
    if "unmute" in core:
        return Tier1Match("unmute", {}, "Unmuted.")
    if "mute" in core:
        return Tier1Match("mute", {}, "Muted.")
    return None


_PATTERN_MATCHERS: list[Callable[[str], Optional[Tier1Match]]] = [
    _match_lights, _match_playback_filler, _match_volume_set, _match_mute,
]


def _route(transcription: str) -> Optional[Tier1Match]:
    words = transcription.split()
    if len(words) > 10:
        return None  # Too long for Tier 1
    normalized = transcription.lower().strip().rstrip(".!?,")
    direct = _DIRECT_COMMANDS.get(normalized)
    if direct:
        cmd_name, params, response = direct
        logger.info("Tier 1 exact: '%s' -> %s(%s)", normalized, cmd_name, params)
        return Tier1Match(cmd_name, params, response)
    for matcher in _PATTERN_MATCHERS:
        result = matcher(normalized)
        if result is not None:
            logger.info("Tier 1 pattern: '%s' -> %s(%s)", normalized, result.command, result.params)
            return result
    return None


class IntentRouter:
    def route(self, turn: VoiceTurn) -> Optional[Tier1Match]:
        return _route(turn.input_text)
