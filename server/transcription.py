"""Speech-to-text transcription using Faster Whisper."""
import io
import logging
import wave
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from server.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)


def _wav_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 mono array at native sample rate.

    Validates WAV header to prevent pathological allocations (e.g. a header
    claiming billions of frames with only a few bytes of data).
    """
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        n_frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        # Sanity: reject files claiming > 5 min at any reasonable sample rate,
        # or rates outside the plausible range for speech audio.
        if n_frames > 5 * 60 * 48000:
            raise ValueError(f"WAV claims {n_frames} frames — too large")
        if rate < 8000 or rate > 48000:
            raise ValueError(f"WAV sample rate {rate} Hz outside valid range")
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples


class Transcriber:
    """Handles speech-to-text transcription using Faster Whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model: Optional[WhisperModel] = None
        logger.info(f"Transcriber initialized with model: {model_name}")

    def initialize(self) -> bool:
        """Load Whisper model. Returns success."""
        try:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.model = WhisperModel(
                self.model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Whisper initialization failed: {e}")
            return False

    def _run_transcription(self, audio) -> Optional[str]:
        """Run transcription on a file path or numpy float32 array."""
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,  # permissive — per-segment filter below handles quality
        )
        segments = list(segments)
        if not segments:
            logger.warning("Transcription: no speech detected")
            return None

        # Per-segment confidence filtering — reject high-confidence garbage
        filtered_segments = []
        for seg in segments:
            if seg.no_speech_prob > 0.7:
                logger.info(f"STT dropping segment (no_speech={seg.no_speech_prob:.2f}): '{seg.text[:50]}'")
                continue
            if seg.avg_logprob < -0.8:
                logger.info(f"STT dropping segment (logprob={seg.avg_logprob:.2f}): '{seg.text[:50]}'")
                continue
            filtered_segments.append(seg)
        if not filtered_segments:
            # Log what was rejected so we can tune thresholds
            for seg in segments:
                logger.warning(
                    f"STT rejected: '{seg.text[:50]}' "
                    f"(no_speech={seg.no_speech_prob:.2f}, logprob={seg.avg_logprob:.2f})"
                )
            return None
        segments = filtered_segments

        avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
        if avg_no_speech > 0.6:
            logger.warning(f"Transcription rejected: no_speech_prob={avg_no_speech:.2f}")
            return None

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            logger.warning("Transcription resulted in empty text")
            return None

        # Debug level — orchestrator logs the authoritative truncated version at INFO
        logger.debug(f"Transcribed ({len(segments)} segments): '{text[:100]}'")

        return text

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        if not self.model:
            logger.error("Transcriber not initialized")
            return None
        try:
            return self._run_transcription(audio_path)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def transcribe_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe WAV audio from bytes (no temp file)."""
        if not self.model:
            logger.error("Transcriber not initialized")
            return None
        try:
            audio = _wav_to_float32(audio_bytes)
            return self._run_transcription(audio)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
