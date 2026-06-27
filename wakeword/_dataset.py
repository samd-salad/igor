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


def split_indices(n: int, holdout_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically partition indices 0..n-1 into (train_idx, holdout_idx).

    File-level split (not window-level): callers pass a FILE count so an
    individual clip's training windows can never leak into the holdout
    evaluation set. With same `seed`, the split is byte-for-byte stable
    across runs — same model, same holdout, comparable gate metrics.

    holdout_frac must be in (0.0, 1.0) exclusive. Size of holdout =
    round(n * holdout_frac); train gets the remainder.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(
            f"holdout_frac must be in (0.0, 1.0), got {holdout_frac}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    holdout_size = int(round(n * holdout_frac))
    return perm[holdout_size:], perm[:holdout_size]


def build_negative_windows(clips_emb: np.ndarray, stride: int = 1) -> np.ndarray:
    """Slide 16-frame windows over each (T, 96) clip in clips_emb.
    Returns (N, 16, 96). Clips shorter than 16 frames are skipped.
    stride > 1 reduces window count proportionally (use to keep memory in check
    when the real-negative pool is large)."""
    out = []
    for clip in clips_emb:
        if clip.shape[0] < WINDOW_FRAMES:
            continue
        max_start = clip.shape[0] - WINDOW_FRAMES
        for start in range(0, max_start + 1, stride):
            out.append(clip[start:start + WINDOW_FRAMES])
    if not out:
        return np.empty((0, WINDOW_FRAMES, 96), dtype=np.float32)
    return np.stack(out).astype(np.float32)
