"""Text-to-speech synthesis using Kokoro (kokoro-onnx)."""
import asyncio
import io
import logging
import re
import wave
from typing import Dict, List, Optional, Tuple

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


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack float32 samples into WAV bytes, clipping to int16 range."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class Synthesizer:
    """Handles text-to-speech using Kokoro ONNX."""

    def __init__(self):
        self.voice = KOKORO_VOICE
        self.speed = KOKORO_SPEED
        self.sample_rate = KOKORO_SAMPLE_RATE
        self._kokoro = None
        self._tts_cache: Dict[str, bytes] = {}
        self._init_failed = False
        logger.info(f"Synthesizer created (voice={self.voice}, speed={self.speed})")

    def _ensure_loaded(self) -> bool:
        """Lazy-load Kokoro model on first use. Returns True if ready."""
        if self._kokoro is not None:
            return True
        if self._init_failed:
            return False
        return self.initialize()

    def initialize(self) -> bool:
        """Load Kokoro model into memory. Returns success.

        Called automatically on first synthesis. Can also be called explicitly
        at startup for eager loading.
        """
        try:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(KOKORO_MODEL_FILE, KOKORO_VOICES_FILE)
            logger.info("Kokoro TTS loaded successfully")
            return True
        except ImportError:
            logger.error("kokoro-onnx not installed. Run: pip install kokoro-onnx")
            self._init_failed = True
            return False
        except FileNotFoundError as e:
            logger.error(
                f"Kokoro model files not found: {e}\n"
                "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases and place in kokoro/"
            )
            self._init_failed = True
            return False
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            self._init_failed = True
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
            audio = self._synthesize_cleaned(cleaned)
            if audio:
                self._tts_cache[cleaned] = audio
                count += 1
        logger.info(f"Pre-cached TTS: {count} responses ready")
        return count

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech and return WAV bytes.

        Public API — preprocesses text before synthesis. For already-cleaned
        text, use _synthesize_cleaned() internally.
        """
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None

        text = _preprocess_text(text)
        if not text:
            logger.warning("Text empty after preprocessing")
            return None

        return self._synthesize_cleaned(text)

    def _synthesize_cleaned(self, text: str) -> Optional[bytes]:
        """Synthesize already-preprocessed text. Checks cache, then Kokoro."""
        if text in self._tts_cache:
            logger.debug(f"TTS cache hit: '{text}'")
            return self._tts_cache[text]

        if not self._ensure_loaded():
            logger.error("Synthesizer not available")
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

            audio_bytes = _samples_to_wav(samples, sample_rate)
            logger.info(f"Synthesized {len(audio_bytes)} bytes of audio")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS synthesis failed: {type(e).__name__}: {str(e)[:200]}")
            return None

    def synthesize_fast(self, text: str) -> Optional[bytes]:
        """Synthesize using streaming for long text, sync for short.

        For text >= 80 chars, uses Kokoro's create_stream() which yields audio
        at phoneme batch boundaries — overlapping inference of batch N+1 with
        post-processing of batch N. Falls back to sync on any error.
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
            return self._synthesize_cleaned(cleaned)

        if not self._ensure_loaded():
            logger.error("Synthesizer not available")
            return None

        try:
            loop = asyncio.new_event_loop()
            try:
                chunks, stream_sr = loop.run_until_complete(
                    self._stream_collect(cleaned)
                )
            finally:
                loop.close()

            if not chunks:
                logger.warning("Streaming TTS returned no chunks, falling back")
                return self._synthesize_cleaned(cleaned)

            all_samples = np.concatenate(chunks)
            audio_bytes = _samples_to_wav(all_samples, stream_sr)
            logger.info(
                f"Synthesized (stream) {len(audio_bytes)} bytes from {len(chunks)} chunks"
            )
            return audio_bytes

        except Exception as e:
            logger.warning(f"Streaming TTS failed ({type(e).__name__}), falling back to sync")
            return self._synthesize_cleaned(cleaned)

    async def _stream_collect(self, text: str) -> Tuple[List[np.ndarray], int]:
        """Collect all chunks from create_stream. Returns (chunks, sample_rate)."""
        chunks = []
        sample_rate = self.sample_rate  # fallback
        async for samples, sr in self._kokoro.create_stream(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        ):
            if samples is not None and len(samples) > 0:
                chunks.append(samples)
                sample_rate = sr  # use Kokoro's actual rate
        return chunks, sample_rate
