#!/usr/bin/env python3
"""Run the trained TFLite model directly against positive/negative samples.

If positives score >>0 and negatives score ~0, the model itself is fine and
the problem is on the deployment side. If positives score low, the TFLite
export broke something.

Run from repo root:
    .onnx2tf-venv/Scripts/python.exe wakeword/test_tflite.py
"""
import random
import sys
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf
from openwakeword.utils import AudioFeatures

ROOT = Path(__file__).parent
TFLITE_PATH = ROOT / "models" / "igor_v0.1.tflite"
POSITIVE_DIR = ROOT / "samples" / "positive"
NEGATIVE_DIR = ROOT / "samples" / "negative"
SAMPLE_RATE = 16000
N_TEST = 30  # how many of each to test


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            return None
        n = wf.getnframes()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # Peak-normalize like training does
    peak = max(abs(audio.max()), abs(audio.min()))
    if peak > 0:
        audio = audio * (0.5 / peak)
    return (audio * 32768.0).astype(np.int16)


def main():
    if not TFLITE_PATH.exists():
        print(f"TFLite not found: {TFLITE_PATH}")
        sys.exit(1)

    print(f"Loading {TFLITE_PATH.name}...")
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    print(f"  Input shape: {in_detail['shape']}  dtype: {in_detail['dtype']}")
    print(f"  Output shape: {out_detail['shape']}  dtype: {out_detail['dtype']}")

    print("\nLoading OWW backbone (melspec + embeddings)...")
    feat = AudioFeatures(sr=SAMPLE_RATE, ncpu=4, inference_framework="onnx")

    def score_clip(audio_int16: np.ndarray) -> float:
        # Match training: pad/trim to 3 sec, then embed
        target = SAMPLE_RATE * 3
        if len(audio_int16) < target:
            audio_int16 = np.pad(audio_int16, (0, target - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:target]
        emb = feat.embed_clips(np.array([audio_int16]), batch_size=1)  # (1, 28, 96)

        # The wakeword model takes a sliding 16-frame window.
        # Score across all valid windows and return the max.
        n_frames = emb.shape[1]
        best = 0.0
        for start in range(n_frames - 16 + 1):
            window = emb[:, start:start+16, :].astype(np.float32)  # (1, 16, 96)
            interp.set_tensor(in_detail['index'], window)
            interp.invoke()
            score = float(interp.get_tensor(out_detail['index'])[0, 0])
            best = max(best, score)
        return best

    pos_files = sorted(POSITIVE_DIR.glob("*.wav"))
    neg_files = sorted(NEGATIVE_DIR.glob("*.wav"))
    random.seed(0)
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    pos_files = pos_files[:N_TEST]
    neg_files = neg_files[:N_TEST]

    print(f"\nScoring {len(pos_files)} positive + {len(neg_files)} negative clips...")
    pos_scores, neg_scores = [], []
    for p in pos_files:
        audio = load_wav(p)
        if audio is None:
            continue
        pos_scores.append(score_clip(audio))
    for p in neg_files:
        audio = load_wav(p)
        if audio is None:
            continue
        neg_scores.append(score_clip(audio))

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    print(f"\nPositives ({len(pos_arr)}):  mean={pos_arr.mean():.3f}  min={pos_arr.min():.3f}  max={pos_arr.max():.3f}")
    print(f"Negatives ({len(neg_arr)}):  mean={neg_arr.mean():.3f}  min={neg_arr.min():.3f}  max={neg_arr.max():.3f}")
    print(f"\nPositive scores: {sorted(pos_arr.round(3).tolist())}")
    print(f"Negative scores (top 10): {sorted(neg_arr.round(3).tolist(), reverse=True)[:10]}")

    if pos_arr.mean() > 0.5 and neg_arr.max() < 0.5:
        print("\n✓ Model looks healthy. Problem is likely on the Pi side (audio path / threshold).")
    elif pos_arr.mean() < 0.3:
        print("\n✗ Positives scoring low — TFLite export likely broke the model. Try the input shape (1,16,96) instead.")
    else:
        print("\n? Mixed results. Inspect the score distribution above.")


if __name__ == "__main__":
    main()
