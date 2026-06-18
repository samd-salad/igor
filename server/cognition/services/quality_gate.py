"""QualityGate — VoiceTurn-shaped post-STT filter. Rejects garbage before the LLM."""
from __future__ import annotations
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    text: str
    rejected: bool
    reason: str


# Whisper hallucinates these from silence, music, or ambiguous audio.
_HALLUCINATIONS = frozenset({
    "thank you.", "thanks for watching.", "bye.", "goodbye.",
    "thank you for watching.", "thanks for watching",
    "thank you so much.", "thank you very much.",
    "please subscribe.", "like and subscribe.",
    "see you next time.", "see you in the next video.",
    "you", "the", "i", "a", "it", "so", "hmm", "um", "uh",
    "...", "…",
})

_SINGLE_WORD_COMMANDS = frozenset({
    "pause", "play", "resume", "stop", "mute", "unmute",
    "next", "skip", "previous", "louder", "quieter",
})

_DISMISSALS = (
    "bad wake word", "bad wakeword", "false trigger",
    "ignore that", "ignore this", "never mind", "nevermind",
    "didn't mean to", "not talking to you", "wasn't talking to you",
    "sorry igor", "sorry, igor", "go away", "not you",
)


def _filter(text: str, tv_playing: bool = False) -> GateResult:
    if not text or not text.strip():
        return GateResult(text="", rejected=True, reason="empty")

    cleaned = text.strip()
    lowered = cleaned.lower().rstrip(".")

    if cleaned.lower() in _HALLUCINATIONS or lowered in _HALLUCINATIONS:
        logger.info("Quality gate: rejected hallucination '%s'", cleaned[:30])
        return GateResult(text="", rejected=True, reason="hallucination")

    if any(d in lowered for d in _DISMISSALS):
        logger.info("Quality gate: rejected dismissal '%s'", cleaned[:30])
        return GateResult(text="", rejected=True, reason="dismissal")

    words = cleaned.split()

    if len(words) == 1 and lowered.rstrip(".!?,") not in _SINGLE_WORD_COMMANDS:
        logger.info("Quality gate: rejected single word '%s'", cleaned)
        return GateResult(text="", rejected=True, reason="single_word")

    if len(words) >= 6:
        sentences = [s.strip().lower() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        if sentences and len(sentences) >= 3:
            if Counter(sentences).most_common(1)[0][1] >= 3:
                logger.info("Quality gate: rejected repetitive text")
                return GateResult(text="", rejected=True, reason="repetitive")

    if tv_playing and len(words) > 40:
        logger.info("Quality gate: rejected long TV-time transcript (%d words)", len(words))
        return GateResult(text="", rejected=True, reason="tv_long")

    return GateResult(text=cleaned, rejected=False, reason="ok")


class QualityGate:
    def filter(self, turn: VoiceTurn) -> GateResult:
        return _filter(turn.input_text, tv_playing=False)
