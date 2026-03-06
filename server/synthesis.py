"""Text-to-speech synthesis using Kokoro (kokoro-onnx)."""
import asyncio
import io
import logging
import re
import wave
from typing import Dict, List, Optional

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
        self._tts_cache: Dict[str, bytes] = {}
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

    def pre_generate(self, texts: List[str]) -> int:
        """Pre-generate and cache TTS for a list of short response strings.

        Returns count of successfully cached responses.
        """
        count = 0
        for raw_text in texts:
            cleaned = _preprocess_text(raw_text)
            if not cleaned or cleaned in self._tts_cache:
                continue
            audio = self.synthesize(cleaned)
            if audio:
                self._tts_cache[cleaned] = audio
                count += 1
        logger.info(f"Pre-cached TTS: {count} responses ready")
        return count

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech and return WAV bytes."""
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None

        text = _preprocess_text(text)
        if not text:
            logger.warning("Text empty after preprocessing")
            return None

        if text in self._tts_cache:
            logger.debug(f"TTS cache hit: '{text}'")
            return self._tts_cache[text]

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

    def synthesize_fast(self, text: str) -> Optional[bytes]:
        """Synthesize using streaming for long text, sync for short.

        For text >= 80 chars, uses Kokoro's create_stream() which yields audio
        at phoneme batch boundaries — overlapping inference of batch N+1 with
        post-processing of batch N. Falls back to synthesize() on any error.
        """
        if not text:
            return None

        cleaned = _preprocess_text(text)
        if not cleaned:
            return None

        # Cache hit — fast path
        if cleaned in self._tts_cache:
            logger.debug(f"TTS cache hit: '{cleaned}'")
            return self._tts_cache[cleaned]

        # Short text — no streaming benefit (single phoneme batch)
        if len(cleaned) < 80:
            return self.synthesize(cleaned)

        if not self._kokoro:
            logger.error("Synthesizer not initialized")
            return None

        try:
            loop = asyncio.new_event_loop()
            try:
                chunks = loop.run_until_complete(self._stream_collect(cleaned))
            finally:
                loop.close()

            if not chunks:
                logger.warning("Streaming TTS returned no chunks, falling back")
                return self.synthesize(cleaned)

            all_samples = np.concatenate(chunks)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((all_samples * 32767).astype(np.int16).tobytes())

            audio_bytes = buf.getvalue()
            logger.info(f"Synthesized (stream) {len(audio_bytes)} bytes from {len(chunks)} chunks")
            return audio_bytes

        except Exception as e:
            logger.warning(f"Streaming TTS failed ({type(e).__name__}), falling back to sync")
            return self.synthesize(cleaned)

    async def _stream_collect(self, text: str) -> List[np.ndarray]:
        """Collect all chunks from create_stream into a list."""
        chunks = []
        async for samples, _sr in self._kokoro.create_stream(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        ):
            if samples is not None and len(samples) > 0:
                chunks.append(samples)
        return chunks
