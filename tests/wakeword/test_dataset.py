import numpy as np
import pytest

from wakeword._dataset import (
    build_positive_window, build_negative_windows, split_indices,
    WINDOW_FRAMES,
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


def test_build_negative_windows_slides_over_clips():
    # 2 clips, each 20 frames of (96-dim) embeddings
    clips_emb = np.zeros((2, 20, 96), dtype=np.float32)
    clips_emb[0, :, 0] = np.arange(20)   # marker
    clips_emb[1, :, 0] = np.arange(100, 120)
    wins = build_negative_windows(clips_emb, stride=1)
    # 2 clips × (20-16+1) = 10 windows
    assert wins.shape == (10, 16, 96)


def test_build_negative_windows_respects_stride():
    clips_emb = np.zeros((1, 20, 96), dtype=np.float32)
    wins = build_negative_windows(clips_emb, stride=2)
    # (20-16)//2 + 1 = 3 windows
    assert wins.shape == (3, 16, 96)


def test_build_negative_windows_skips_too_short_clips():
    clips_emb = np.zeros((2, 10, 96), dtype=np.float32)
    wins = build_negative_windows(clips_emb, stride=1)
    assert wins.shape == (0, 16, 96)


def test_split_indices_is_disjoint_and_covers_all():
    train, holdout = split_indices(100, holdout_frac=0.2, seed=0)
    assert len(train) == 80
    assert len(holdout) == 20
    assert set(train.tolist()).isdisjoint(set(holdout.tolist()))
    assert set(train.tolist()) | set(holdout.tolist()) == set(range(100))


def test_split_indices_is_deterministic_for_same_seed():
    a_train, a_hold = split_indices(100, holdout_frac=0.2, seed=42)
    b_train, b_hold = split_indices(100, holdout_frac=0.2, seed=42)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_hold, b_hold)


def test_split_indices_changes_with_seed():
    a_hold = split_indices(100, holdout_frac=0.2, seed=0)[1]
    b_hold = split_indices(100, holdout_frac=0.2, seed=1)[1]
    assert not np.array_equal(a_hold, b_hold)


def test_split_indices_rejects_invalid_frac():
    with pytest.raises(ValueError):
        split_indices(100, holdout_frac=0.0, seed=0)
    with pytest.raises(ValueError):
        split_indices(100, holdout_frac=1.0, seed=0)
    with pytest.raises(ValueError):
        split_indices(100, holdout_frac=-0.1, seed=0)


def test_split_indices_rounds_holdout_to_nearest_clip():
    # 4250 clips × 0.2 = 850.0 exact — but verify rounding behavior with awkward sizes
    train, holdout = split_indices(7, holdout_frac=0.2, seed=0)
    # round(7 * 0.2) = round(1.4) = 1
    assert len(holdout) == 1
    assert len(train) == 6
