"""Post-STT quality gate. Rejects garbage before it reaches the LLM."""
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of quality gate filtering."""
    text: Optional[str]  # Cleaned text, or None if rejected
    reason: str          # "ok", "dismissal", "hallucination", "single_word", "repetitive", "tv_long"

# Whisper hallucinates these from silence, music, or ambiguous audio
_HALLUCINATIONS = frozenset({
    "thank you.", "thanks for watching.", "bye.", "goodbye.",
    "thank you for watching.", "thanks for watching",
    "thank you so much.", "thank you very much.",
    "please subscribe.", "like and subscribe.",
    "see you next time.", "see you in the next video.",
    "you", "the", "i", "a", "it", "so", "hmm", "um", "uh",
    "...", "\u2026",
})

# Single words that are valid commands — exempt from "too short" filter
_SINGLE_WORD_COMMANDS = frozenset({
    "pause", "play", "resume", "stop", "mute", "unmute",
    "next", "skip", "previous", "louder", "quieter",
})

# User explicitly dismissing a false wake word trigger — reject immediately
# so it never reaches the LLM (saves ~2s + API cost per false trigger).
_DISMISSALS = (
    "bad wake word", "bad wakeword", "false trigger",
    "ignore that", "ignore this", "never mind", "nevermind",
    "didn't mean to", "not talking to you", "wasn't talking to you",
    "sorry igor", "sorry, igor", "go away", "not you",
)


def filter_transcription(text: str, tv_playing: bool = False) -> GateResult:
    """Filter transcription. Returns GateResult with text and rejection reason.

    Callers that only need pass/fail can check `result.text is not None`.
    Callers that need the reason (e.g. auto-relabeling false wake words)
    can check `result.reason`.
    """
    if not text or not text.strip():
        return GateResult(None, "empty")

    cleaned = text.strip()
    lowered = cleaned.lower().rstrip(".")

    # Known hallucination phrases
    if cleaned.lower() in _HALLUCINATIONS or lowered in _HALLUCINATIONS:
        logger.info(f"Quality gate: rejected hallucination '{cleaned[:30]}'")
        return GateResult(None, "hallucination")

    # User dismissing a false wake word trigger
    if any(d in lowered for d in _DISMISSALS):
        logger.info(f"Quality gate: rejected dismissal '{cleaned[:30]}'")
        return GateResult(None, "dismissal")

    words = cleaned.split()

    # Single-word: must be a known command word
    if len(words) == 1 and lowered.rstrip(".!?,") not in _SINGLE_WORD_COMMANDS:
        logger.info(f"Quality gate: rejected single word '{cleaned}'")
        return GateResult(None, "single_word")

    # Repetition detection (Whisper loops)
    if len(words) >= 6:
        sentences = [s.strip().lower() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
        if sentences and len(sentences) >= 3:
            from collections import Counter
            if Counter(sentences).most_common(1)[0][1] >= 3:
                logger.info(f"Quality gate: rejected repetitive text")
                return GateResult(None, "repetitive")

    # TV playing: reject long narrative (likely TV dialogue).
    if tv_playing and len(words) > 40:
        logger.info(f"Quality gate: rejected long transcription during TV ({len(words)} words)")
        return GateResult(None, "tv_long")

    return GateResult(cleaned, "ok")
