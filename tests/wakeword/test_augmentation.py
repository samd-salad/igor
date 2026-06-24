import numpy as np
import pytest

from wakeword._augmentation import (
    mix_with_background, apply_rir, generate_synthetic_rirs, random_snr_db,
)


def test_random_snr_db_within_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        snr = random_snr_db(rng, low=0.0, high=15.0)
        assert 0.0 <= snr <= 15.0


def test_mix_with_background_returns_int16_same_length():
    rng = np.random.default_rng(0)
    pos = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 8000).astype(np.int16)
    bg = (rng.normal(0, 1000, 16000)).astype(np.int16)
    mixed = mix_with_background(pos, bg, snr_db=10.0, rng=rng)
    assert mixed.shape == pos.shape
    assert mixed.dtype == np.int16


def test_mix_with_background_high_snr_preserves_positive_loudness():
    """At 50 dB SNR, the mix should be ~indistinguishable from the positive."""
    rng = np.random.default_rng(0)
    pos = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 8000).astype(np.int16)
    bg = (rng.normal(0, 1000, 16000)).astype(np.int16)
    mixed = mix_with_background(pos, bg, snr_db=50.0, rng=rng)
    # RMS should be very close to positive's RMS
    rms_pos = np.sqrt(np.mean(pos.astype(np.float64) ** 2))
    rms_mix = np.sqrt(np.mean(mixed.astype(np.float64) ** 2))
    assert 0.9 < rms_mix / rms_pos < 1.1


def test_mix_with_background_low_snr_lifts_floor():
    """At 0 dB SNR, the background contributes equally — output is louder."""
    rng = np.random.default_rng(0)
    pos = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 8000).astype(np.int16)
    bg = (rng.normal(0, 8000, 16000)).astype(np.int16)
    mixed_high = mix_with_background(pos, bg, snr_db=50.0, rng=rng)
    mixed_low = mix_with_background(pos, bg, snr_db=0.0, rng=rng)
    rms_high = np.sqrt(np.mean(mixed_high.astype(np.float64) ** 2))
    rms_low = np.sqrt(np.mean(mixed_low.astype(np.float64) ** 2))
    assert rms_low > rms_high


def test_generate_synthetic_rirs_returns_n_arrays():
    rng = np.random.default_rng(0)
    rirs = generate_synthetic_rirs(5, rng)
    assert len(rirs) == 5
    for r in rirs:
        assert r.ndim == 1
        assert r.dtype == np.float32 or r.dtype == np.float64
        # Sane length: 50ms to 1s @ 16kHz
        assert 800 <= len(r) <= 16000


def test_apply_rir_preserves_length():
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 1000, 16000)).astype(np.int16)
    rir = generate_synthetic_rirs(1, rng)[0]
    out = apply_rir(audio, rir)
    assert out.shape == audio.shape
    assert out.dtype == np.int16


def test_apply_rir_does_not_silence_signal():
    rng = np.random.default_rng(0)
    audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 8000).astype(np.int16)
    rir = generate_synthetic_rirs(1, rng)[0]
    out = apply_rir(audio, rir)
    # Output should still have audible content
    assert np.max(np.abs(out)) > 1000
