"""IntentRouter — Tier 1 direct-match shell.

Currently a no-op. The previous patterns (lights/playback/volume/mute) all
emitted tool names from before the HA-MCP migration (`set_light`, `play_pause`,
`set_volume`, `mute`) — names no executor in the current system handles.
Tier 1 was returning canned responses while nothing actually executed, so the
patterns were ripped out.

This shell stays because the right rebuild is an Intent-emitting layer:
matchers return value objects (e.g. `TurnLightsOff(scope=All)`) and a separate
resolver translates them against whatever executor catalog is live. Until
then, every utterance falls through to Tier 2 (LLM) which already picks the
correct HA MCP tool from the live catalog.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Optional

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tier1Match:
    command: str
    params: dict[str, Any]
    response: str


def _route(transcription: str) -> Optional[Tier1Match]:
    return None


class IntentRouter:
    def route(self, turn: VoiceTurn) -> Optional[Tier1Match]:
        return _route(turn.input_text)
