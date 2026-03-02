"""Text-to-speech synthesis using Kokoro (kokoro-onnx)."""
import io
import logging
import re
import wave
from typing import Optional

import numpy as np

from server.config import (
    KOKORO_MODEL_FILE,
    KOKORO_VOICES_FILE,
    KOKORO_VOICE,
    KOKORO_SPEED,
    KOKORO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def _preprocess_text(text: str) -> str:
    """Clean text for TTS synthesis."""
    # Normalize quotes
    text = text.replace('\u201c', '').replace('\u201d', '').replace('"', '')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Replace semicolons with commas (more natural pause)
    text = text.replace(';', ',')

    # Replace colons mid-sentence with commas (except time like 3:00)
    text = re.sub(r':(?!\d)', ',', text)

    # Remove asterisks (markdown artifacts)
    text = text.replace('*', '')

    # Normalize dashes to simple pauses
    text = text.replace('\u2014', ', ').replace('\u2013', ', ').replace(' - ', ', ')

    # Remove parentheses (speak content naturally)
    text = text.replace('(', ', ').replace(')', ', ')

    # Clean up multiple commas/spaces from replacements
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


class Synthesizer:
    """Handles text-to-speech using Kokoro ONNX."""

    def __init__(self):
        self.voice = KOKORO_VOICE
        self.speed = KOKORO_SPEED
        self.sample_rate = KOKORO_SAMPLE_RATE
        self._kokoro = None
        logger.info(f"Synthesizer initialized (voice={self.voice}, speed={self.speed})")

    def initialize(self) -> bool:
        """Load Kokoro model into memory. Returns success."""
        try:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(KOKORO_MODEL_FILE, KOKORO_VOICES_FILE)
            logger.info("Kokoro TTS loaded successfully")
            return True
        except ImportError:
            logger.error("kokoro-onnx not installed. Run: pip install kokoro-onnx")
            return False
        except FileNotFoundError as e:
            logger.error(
                f"Kokoro model files not found: {e}\n"
                "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases and place in kokoro/"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return False

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech and return WAV bytes."""
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None

        text = _preprocess_text(text)
        if not text:
            logger.warning("Text empty after preprocessing")
            return None

        if not self._kokoro:
            logger.error("Synthesizer not initialized")
            return None

        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self.voice,
                speed=self.speed,
                lang="en-us",
            )

            if samples is None or len(samples) == 0:
                logger.error("Kokoro returned empty audio")
                return None

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((samples * 32767).astype(np.int16).tobytes())

            audio_bytes = buf.getvalue()
            logger.info(f"Synthesized {len(audio_bytes)} bytes of audio")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS synthesis failed: {type(e).__name__}: {str(e)[:200]}")
            return None
