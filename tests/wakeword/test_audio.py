import numpy as np

from wakeword._audio import (
    load_wav, normalize_peak, trim_trailing_silence, left_pad_or_trim,
)


def test_normalize_peak_scales_to_target():
    audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 1000).astype(np.int16)
    out = normalize_peak(audio, target_peak=16000)
    assert 15500 <= np.max(np.abs(out)) <= 16500


def test_normalize_peak_leaves_digital_silence_alone():
    audio = np.zeros(16000, dtype=np.int16)
    out = normalize_peak(audio)
    assert np.array_equal(out, audio)


def test_trim_trailing_silence_removes_silent_tail():
    audio = np.zeros(16000, dtype=np.int16)
    audio[:8000] = 1000  # signal in first 0.5s, silence after
    out = trim_trailing_silence(audio, threshold=250, keep_tail_samples=1600)
    # Should keep signal + 100ms tail = ~9600 samples
    assert 8000 < len(out) < 12000


def test_left_pad_or_trim_pads_when_short():
    audio = np.ones(8000, dtype=np.int16)
    out = left_pad_or_trim(audio, 16000)
    assert len(out) == 16000
    assert np.all(out[:8000] == 0)   # silence at start
    assert np.all(out[8000:] == 1)   # signal at end
