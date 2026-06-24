"""Audio augmentation: SNR-controlled background mixing + RIR convolution.

These are the canonical noise-robustness levers in dscripka's OWW training
notebooks. SNR mixing teaches the model to recognize the wake word under
varied background loudness; RIR convolution adds room-acoustic variability
so the model generalizes beyond the recording environment."""
from __future__ import annotations
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve


def random_snr_db(rng: np.random.Generator,
                  low: float = 0.0, high: float = 15.0) -> float:
    return float(rng.uniform(low, high))


def _rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def mix_with_background(positive: np.ndarray, background: np.ndarray,
                        snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix positive at +snr_db relative to background. Output length matches
    positive. Background is rolled to a random start so we don't always
    mix with the same chunk."""
    if len(background) < len(positive):
        background = np.tile(background, int(np.ceil(len(positive) / max(len(background), 1))))
    # Random start in background
    if len(background) > len(positive):
        start = int(rng.integers(0, len(background) - len(positive) + 1))
        bg_slice = background[start:start + len(positive)]
    else:
        bg_slice = background

    rms_pos = _rms(positive)
    rms_bg = _rms(bg_slice)
    if rms_bg < 1e-6 or rms_pos < 1e-6:
        return positive.astype(np.int16)

    # gain to apply to bg so that rms_pos/(scale*rms_bg) = 10^(snr_db/20)
    scale = (rms_pos / rms_bg) / (10 ** (snr_db / 20))
    bg_scaled = bg_slice.astype(np.float64) * scale
    mixed = positive.astype(np.float64) + bg_scaled
    return np.clip(mixed, -32768, 32767).astype(np.int16)


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve audio with rir and trim back to original length.
    Output is loudness-matched to the input via peak normalization."""
    in_peak = int(np.max(np.abs(audio))) if len(audio) else 0
    if in_peak == 0:
        return audio.astype(np.int16)
    convolved = fftconvolve(audio.astype(np.float64), rir.astype(np.float64), mode="full")
    convolved = convolved[:len(audio)]
    out_peak = float(np.max(np.abs(convolved)))
    if out_peak < 1e-6:
        return audio.astype(np.int16)
    gain = in_peak / out_peak
    return np.clip(convolved * gain, -32768, 32767).astype(np.int16)


def generate_synthetic_rirs(n: int, rng: np.random.Generator,
                            sample_rate: int = 16000) -> list[np.ndarray]:
    """Generate n RIRs by simulating random rectangular rooms with
    pyroomacoustics. Rooms range from small (3m³) to medium (50m³)."""
    rirs: list[np.ndarray] = []
    for _ in range(n):
        # Random room dimensions
        length = float(rng.uniform(3, 8))
        width  = float(rng.uniform(3, 6))
        height = float(rng.uniform(2.4, 3.5))
        # Random absorption — low = reverberant, high = dead
        absorption = float(rng.uniform(0.2, 0.6))
        max_order = int(rng.integers(2, 8))

        room = pra.ShoeBox(
            [length, width, height],
            fs=sample_rate,
            materials=pra.Material(absorption),
            max_order=max_order,
        )
        # Random source + mic positions inside the room
        src = [float(rng.uniform(0.5, d - 0.5)) for d in (length, width, height)]
        mic = [float(rng.uniform(0.5, d - 0.5)) for d in (length, width, height)]
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        rir = room.rir[0][0]
        rir = np.asarray(rir, dtype=np.float32)
        # Cap RIR length to 1s
        if len(rir) > sample_rate:
            rir = rir[:sample_rate]
        rirs.append(rir)
    return rirs
