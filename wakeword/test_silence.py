#!/usr/bin/env python3
"""Offline silence/noise smoke test for the trained TFLite wake-word model.

A healthy custom OWW model produces max score < 0.5 on:
  - pure zeros (digital silence)
  - low-amplitude Gaussian noise (mic floor)
  - 60 Hz / 300 Hz tones (electrical hum, fan noise)
  - looped 1-2 kHz tones

Fails the test (exit 1) if any test produces 3 consecutive windows above
0.5 — that's a runtime trigger condition.

Run from repo root:
    .venv-wake/Scripts/python.exe wakeword/test_silence.py
    .venv-wake/Scripts/python.exe wakeword/test_silence.py --model igor_v0.3.tflite
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Fallback for older TensorFlow versions
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Add repo root to path so we can import wakeword from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakeword._features import embed_clip

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "igor_v0.3.tflite"
SAMPLE_RATE = 16000
DURATION_S = 3
THRESHOLD = 0.5
TRIGGER_LEVEL = 3


def _generate_tests() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    N = SAMPLE_RATE * DURATION_S
    t = np.arange(N) / SAMPLE_RATE
    return {
        "pure_zeros":      np.zeros(N, dtype=np.int16),
        "mic_floor_quiet": rng.normal(0, 30, N).astype(np.int16),
        "mic_floor_mid":   rng.normal(0, 200, N).astype(np.int16),
        "mic_floor_loud":  rng.normal(0, 800, N).astype(np.int16),
        "fridge_hum_60":   (np.sin(2*np.pi*60*t) * 200).astype(np.int16),
        "fan_hum_120":     (np.sin(2*np.pi*120*t) * 300).astype(np.int16),
        "fan_tone_300":    (np.sin(2*np.pi*300*t) * 200).astype(np.int16),
        "high_pitch_1k":   (np.sin(2*np.pi*1000*t) * 200).astype(np.int16),
        "high_pitch_2k":   (np.sin(2*np.pi*2000*t) * 200).astype(np.int16),
    }


def _score(interp, in_d, out_d, audio: np.ndarray) -> np.ndarray:
    emb = embed_clip(audio)
    if emb.shape[0] < 16:
        return np.array([])
    scores = []
    for start in range(emb.shape[0] - 16 + 1):
        win = emb[start:start+16][None, :, :].astype(np.float32)
        interp.set_tensor(in_d["index"], win)
        interp.invoke()
        scores.append(float(interp.get_tensor(out_d["index"])[0, 0]))
    return np.array(scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Path to the TFLite model")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--trigger-level", type=int, default=TRIGGER_LEVEL)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    tests = _generate_tests()
    failures = []
    print(f"{'test':<20}  max   mean  >th  3-trig")
    for name, audio in tests.items():
        scores = _score(interp, in_d, out_d, audio)
        triple = any(
            (scores[i:i+args.trigger_level] > args.threshold).all()
            for i in range(len(scores) - args.trigger_level + 1)
        ) if len(scores) >= args.trigger_level else False
        n_above = int((scores > args.threshold).sum())
        max_s = float(scores.max()) if len(scores) else 0.0
        mean_s = float(scores.mean()) if len(scores) else 0.0
        marker = "FAIL" if triple else "ok"
        print(f"{name:<20}  {max_s:.3f}  {mean_s:.3f}  {n_above:3d}  {triple!s:<5} {marker}")
        if triple:
            failures.append(name)

    if failures:
        print(f"\nFAILED on {len(failures)} test(s): {', '.join(failures)}")
        return 1
    print("\nAll silence/noise tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
