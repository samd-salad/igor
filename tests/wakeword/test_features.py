import importlib.util

import numpy as np
import pytest

from wakeword._features import embed_clip, frames_per_seconds, EMB_RATE_HZ

requires_pyopen_wakeword = pytest.mark.skipif(
    importlib.util.find_spec("pyopen_wakeword") is None,
    reason="pyopen_wakeword not installed (training-only dep)",
)


def test_emb_rate_hz_matches_empirical_measurement():
    # Empirically: 128 frames for 3 seconds = 42.67 Hz
    assert 42.0 <= EMB_RATE_HZ <= 43.0


@requires_pyopen_wakeword
def test_embed_clip_produces_expected_frame_count_for_3_seconds():
    silence_3s = np.zeros(16000 * 3, dtype=np.int16)
    emb = embed_clip(silence_3s)
    # 3 seconds × 42.67 Hz ≈ 128 frames
    assert 124 <= emb.shape[0] <= 132
    assert emb.shape[1] == 96


def test_frames_per_seconds_helper():
    assert 124 <= frames_per_seconds(3.0) <= 132
    assert 40 <= frames_per_seconds(1.0) <= 44
