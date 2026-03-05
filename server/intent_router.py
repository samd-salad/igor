"""Tier 1 intent router. Maps short unambiguous phrases to direct commands.

Saves ~1.5-2s latency for the most common voice commands by skipping the LLM entirely.
Only matches phrases where there is ZERO ambiguity. Everything else falls through to LLM.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

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


def route(transcription: str) -> Optional[Tier1Match]:
    """Return Tier1Match if matched, None to fall through to LLM."""
    words = transcription.split()
    if len(words) > 6:
        return None  # Too long for Tier 1 — let LLM handle nuance

    normalized = transcription.lower().strip().rstrip(".!?,")
    match = _DIRECT_COMMANDS.get(normalized)
    if match:
        cmd_name, params, response = match
        logger.info(f"Tier 1 match: '{normalized}' -> {cmd_name}({params})")
        return Tier1Match(command=cmd_name, params=params, response=response)
    return None
