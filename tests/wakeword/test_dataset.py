import numpy as np
import pytest

from wakeword._dataset import (
    build_positive_window, WINDOW_FRAMES,
)


def test_window_frames_constant_is_16():
    assert WINDOW_FRAMES == 16


def test_positive_window_shape_is_16_by_96():
    # 128 frames of fake embeddings (3-second clip at 42.67 Hz)
    clip_emb = np.arange(128 * 96, dtype=np.float32).reshape(128, 96)
    rng = np.random.default_rng(0)
    win = build_positive_window(clip_emb, jitter_ms=0, rng=rng)
    assert win.shape == (16, 96)


def test_positive_window_aligns_to_end_with_zero_jitter():
    clip_emb = np.arange(128 * 96, dtype=np.float32).reshape(128, 96)
    rng = np.random.default_rng(0)
    win = build_positive_window(clip_emb, jitter_ms=0, rng=rng)
    # With jitter=0, the window should be the LAST 16 frames
    assert np.array_equal(win, clip_emb[-16:])


def test_positive_window_with_jitter_stays_in_bounds():
    clip_emb = np.arange(128 * 96, dtype=np.float32).reshape(128, 96)
    rng = np.random.default_rng(0)
    for _ in range(50):
        win = build_positive_window(clip_emb, jitter_ms=200, rng=rng)
        assert win.shape == (16, 96)


def test_positive_window_jitter_actually_shifts_window():
    clip_emb = np.arange(128 * 96, dtype=np.float32).reshape(128, 96)
    rng = np.random.default_rng(42)
    starts_seen = set()
    for _ in range(50):
        win = build_positive_window(clip_emb, jitter_ms=200, rng=rng)
        # Identify start frame from first element
        starts_seen.add(int(win[0, 0]) // 96)
    # Should have observed at least 2 different starts due to jitter
    assert len(starts_seen) >= 2


def test_positive_window_short_clip_raises():
    short = np.zeros((10, 96), dtype=np.float32)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too short"):
        build_positive_window(short, jitter_ms=0, rng=rng)
