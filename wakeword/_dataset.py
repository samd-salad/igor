"""Window construction for OWW classifier training. Reference-aligned:
ONE positive window per clip, anchored to the END of the clip with optional
±jitter_ms timing variation. Multi-window labeling per clip (as the old
code did) labels leading silence as positive and causes runtime
phantom-fires."""
from __future__ import annotations
import numpy as np

from wakeword._features import EMB_RATE_HZ

WINDOW_FRAMES = 16


def build_positive_window(clip_emb: np.ndarray, jitter_ms: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Take one (16, 96) window from clip_emb whose end falls within the
    last `jitter_ms` ms of the clip. Mirrors dscripka's reference notebook:
    'aligning the positive clips with background data such that the end of
    the input window aligns with the end of the positive clip'."""
    if clip_emb.shape[0] < WINDOW_FRAMES:
        raise ValueError(
            f"clip too short for a {WINDOW_FRAMES}-frame window "
            f"(got {clip_emb.shape[0]} frames)"
        )
    n_frames = clip_emb.shape[0]
    jitter_frames = max(0, int(round(jitter_ms / 1000.0 * EMB_RATE_HZ)))
    # End anchored at last frame; jitter shifts END backwards by 0..jitter_frames
    end_offset = int(rng.integers(0, jitter_frames + 1)) if jitter_frames else 0
    end_idx = n_frames - end_offset
    start_idx = end_idx - WINDOW_FRAMES
    if start_idx < 0:
        start_idx, end_idx = 0, WINDOW_FRAMES
    return clip_emb[start_idx:end_idx].astype(np.float32)
