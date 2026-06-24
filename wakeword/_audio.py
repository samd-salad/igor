"""Audio preprocessing helpers — load_wav, normalize_peak, trim_trailing_silence,
left_pad_or_trim. Used by the training data pipeline."""
from __future__ import annotations
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
    """Load a 16kHz mono WAV as int16. Raises on mismatched sr/channels."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
        if ch != 1:
            raise ValueError(f"Expected mono, got {ch} channels")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).copy()


def normalize_peak(audio: np.ndarray, target_peak: int = 16000) -> np.ndarray:
    """Scale audio so its peak amplitude equals target_peak.
    Digital silence (peak < 10) is left untouched."""
    peak = int(np.max(np.abs(audio)))
    if peak < 10:
        return audio
    gain = min(target_peak / peak, 1000.0)
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def trim_trailing_silence(audio: np.ndarray, threshold: int = 250,
                          keep_tail_samples: int = 1600) -> np.ndarray:
    """Strip samples after the last energetic sample, keeping a short tail
    so the final syllable isn't clipped."""
    abs_audio = np.abs(audio)
    nonzero = np.where(abs_audio > threshold)[0]
    if len(nonzero) == 0:
        return audio
    last_real = nonzero[-1]
    end = min(len(audio), last_real + keep_tail_samples)
    return audio[:end]


def left_pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Left-pad with silence so content lives at the END of the clip."""
    if len(audio) >= length:
        return audio[-length:]
    return np.pad(audio, (length - len(audio), 0)).astype(audio.dtype)
