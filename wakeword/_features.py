"""Feature extraction wrapping pyopen_wakeword. Single source of truth for
embedding rate and shape."""
from __future__ import annotations
import numpy as np
from pyopen_wakeword import OpenWakeWordFeatures

# Empirically measured: 128 frames per 3-second clip = 42.67 Hz.
# pyopen_wakeword's MELS_PER_SECOND/EMB_STEP nominal arithmetic (97/8 ≈
# 12.125) does NOT reflect the actual emission rate because
# process_streaming uses sliding-window overlap. Trust the measurement.
EMB_RATE_HZ = 42.67
FEATURE_DIM = 96
CHUNK_BYTES = 320  # 10ms @ 16kHz, 16-bit mono


def frames_per_seconds(seconds: float) -> int:
    return int(round(seconds * EMB_RATE_HZ))


def embed_clip(audio_int16: np.ndarray) -> np.ndarray:
    """Stream a single int16 clip through pyopen_wakeword features.
    Returns (T, 96) where T ≈ duration_seconds × 42.67."""
    feats = OpenWakeWordFeatures.from_builtin()
    feats.reset()
    raw = audio_int16.astype(np.int16).tobytes()
    out = []
    for i in range(0, len(raw) - CHUNK_BYTES + 1, CHUNK_BYTES):
        for f in feats.process_streaming(raw[i:i+CHUNK_BYTES]):
            out.append(f.squeeze())
    if not out:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(out).astype(np.float32)
