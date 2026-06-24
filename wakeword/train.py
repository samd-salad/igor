#!/usr/bin/env python3
"""Train an OpenWakeWord ONNX model from recorded positive samples.

Workflow:
  1. Pi:   python wakeword/record_samples.py   → wakeword/samples/positive/*.wav
  2. PC:   scp or rsync samples to PC
  3. PC:   python wakeword/train.py
  4. Pi:   scp wakeword/models/igor_v0.1.tflite to Pi's ~/wyoming-openwakeword/custom-models/

Dependencies (install once on PC):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install openwakeword onnx2tf tensorflow tf-keras onnxruntime

How it works:
  - OWW's frozen backbone (melspectrogram + embedding models, auto-downloaded ~50 MB)
    converts audio into 96-dimensional embedding vectors.
  - A small 2-layer PyTorch DNN learns to classify those embeddings as
    wake word (1) vs not (0).
  - Exported to ONNX; the output node name becomes the key in model.predict().
  - Also exported to TFLite (fixed batch=1) for wyoming-openwakeword.

Negative samples:
  This script uses synthetic negatives (noise, silence, sine tones) which work
  reasonably well for quiet environments. For fewer false positives in noisy
  conditions, record real negative audio (background speech, TV, music) into
  wakeword/samples/negative/ and re-train.
"""

import sys
from pathlib import Path

import numpy as np

from wakeword._features import embed_clip
from wakeword._audio import (
    load_wav, normalize_peak, trim_trailing_silence,
    left_pad_or_trim,
)
from wakeword._augmentation import (
    apply_rir, mix_with_background, random_snr_db, generate_synthetic_rirs,
)
from wakeword._dataset import (
    build_positive_window, build_negative_windows, WINDOW_FRAMES,
)

ROOT        = Path(__file__).parent.parent
POSITIVE_DIR = ROOT / "wakeword" / "samples" / "positive"
NEGATIVE_DIR = ROOT / "wakeword" / "samples" / "negative"  # optional real negatives
OUTPUT_DIR  = ROOT / "wakeword" / "models"
MODEL_NAME  = "igor"   # becomes the key in model.predict() results

SAMPLE_RATE  = 16000
CLIP_SAMPLES = SAMPLE_RATE * 3  # 3-second clips (pad/trim all audio to this)
N_EPOCHS     = 50   # capped to prevent sigmoid saturation; v0.2 at 200 collapsed to hard 0/1
BATCH_SIZE   = 64
LAYER_DIM    = 64   # increased from 32 — more capacity to discriminate real-world audio
DROPOUT      = 0.4  # bumped from 0.2 to force softer decisions and improve generalization


# ---------------------------------------------------------------------------
# Backbone model download
# ---------------------------------------------------------------------------

_BACKBONE_URLS = {
    "melspectrogram.onnx":   "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx":  "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
}


def _download_backbone_models():
    """Download OWW ONNX backbone models if not present."""
    import openwakeword
    import urllib.request
    resources_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    resources_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _BACKBONE_URLS.items():
        dest = resources_dir / filename
        if dest.exists():
            continue
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Right-pad with silence (for negatives — silence at end is fine)."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(audio.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check dependencies
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install pyopen_wakeword")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load positive clips from both real recordings and Piper synthesis.
    POSITIVE_SYNTH_DIR = ROOT / "wakeword" / "samples" / "positive_synth"
    all_pos_files = sorted(list(POSITIVE_DIR.glob("*.wav")))
    if POSITIVE_SYNTH_DIR.exists():
        all_pos_files += sorted(POSITIVE_SYNTH_DIR.glob("*.wav"))
    if not all_pos_files:
        print(f"No positive WAVs found in {POSITIVE_DIR}/ or {POSITIVE_SYNTH_DIR}/.")
        sys.exit(1)
    print(f"\nLoading {len(all_pos_files)} positive clips "
          f"(real + synthetic), trim+normalize+left-pad to {CLIP_SAMPLES//SAMPLE_RATE}s...")

    pos_audio_raw = []
    for path in all_pos_files:
        try:
            raw = load_wav(path)
            trimmed = trim_trailing_silence(raw)
            audio = normalize_peak(left_pad_or_trim(trimmed, CLIP_SAMPLES))
            pos_audio_raw.append(audio)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    if not pos_audio_raw:
        print("No valid positive clips loaded.")
        sys.exit(1)
    pos_audio_raw = np.stack(pos_audio_raw)
    print(f"  Loaded {len(pos_audio_raw)} positive clips.")

    # ----- Augmentation -----
    # RIR convolution (50% probability per positive) — adds room acoustics.
    # SNR-mix with a random real-negative background (75% probability) at
    # 0-15 dB SNR — teaches the model to recognize the word over backgrounds.
    print("Generating synthetic RIRs (30 rooms)...")
    aug_rng = np.random.default_rng(123)
    rirs = generate_synthetic_rirs(30, aug_rng)
    print(f"  {len(rirs)} RIRs generated.")

    # Need backgrounds for SNR mixing — load negatives now so we can mix
    if NEGATIVE_DIR.exists():
        neg_paths = sorted(NEGATIVE_DIR.glob("*.wav"))
    else:
        neg_paths = []
    if len(neg_paths) < 100:
        print(f"ERROR: need >= 100 real negative clips in {NEGATIVE_DIR}/, "
              f"found {len(neg_paths)}.")
        sys.exit(1)
    print(f"Loading {len(neg_paths)} real negative clips for backgrounds and training...")
    neg_audio_list = []
    for path in neg_paths:
        try:
            audio = normalize_peak(pad_or_trim(load_wav(path), CLIP_SAMPLES))
            neg_audio_list.append(audio)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    neg_audio = np.stack(neg_audio_list) if neg_audio_list else np.empty(
        (0, CLIP_SAMPLES), dtype=np.int16
    )
    print(f"  {len(neg_audio)} negatives loaded.")

    # Apply augmentation
    pos_audio = []
    for clip in pos_audio_raw:
        x = clip
        if aug_rng.random() < 0.5:
            rir = rirs[aug_rng.integers(0, len(rirs))]
            x = apply_rir(x, rir)
        if aug_rng.random() < 0.75 and len(neg_audio) > 0:
            bg = neg_audio[aug_rng.integers(0, len(neg_audio))]
            snr = random_snr_db(aug_rng, low=0.0, high=15.0)
            x = mix_with_background(x, bg, snr_db=snr, rng=aug_rng)
        pos_audio.append(x)
    pos_audio = np.stack(pos_audio)
    n_pos = len(pos_audio)
    print(f"Augmented positives: {n_pos}")

    # ----- Embed and window -----
    print("\nExtracting embeddings via pyopen_wakeword features...")
    pos_emb = np.stack([embed_clip(c) for c in pos_audio])  # (N, T, 96)
    neg_emb = np.stack([embed_clip(c) for c in neg_audio])  # (M, T, 96)
    print(f"  Positive emb: {pos_emb.shape}  Negative emb: {neg_emb.shape}")

    # ONE positive window per clip with ±200ms jitter (reference-aligned).
    win_rng = np.random.default_rng(7)
    pos_wins_list = []
    pos_win_file_idx = []
    n_too_short = 0
    for i, clip_emb in enumerate(pos_emb):
        try:
            win = build_positive_window(clip_emb, jitter_ms=200.0, rng=win_rng)
            pos_wins_list.append(win)
            pos_win_file_idx.append(i)
        except ValueError:
            n_too_short += 1
    if n_too_short:
        print(f"  Skipped {n_too_short} clip(s) too short for a {WINDOW_FRAMES}-frame window")
    pos_wins = np.stack(pos_wins_list).astype(np.float32) if pos_wins_list else \
        np.empty((0, WINDOW_FRAMES, 96), dtype=np.float32)

    # All windows of all negatives (stride=1 — heaviest learning signal)
    neg_wins = build_negative_windows(neg_emb, stride=1)
    print(f"  Windows — positive: {len(pos_wins)}  negative: {len(neg_wins)}")

    # ------------------------------------------------------------------
    # Prepare training tensors
    # ------------------------------------------------------------------
    X = np.vstack([pos_wins, neg_wins])
    y = np.array([1.0] * len(pos_wins) + [0.0] * len(neg_wins), dtype=np.float32)
    # Track which positive file each sample came from (negatives get -1)
    file_idx = np.array(pos_win_file_idx + [-1] * len(neg_wins))

    perm = np.random.default_rng(0).permutation(len(X))
    X, y, file_idx = X[perm], y[perm], file_idx[perm]

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)

    # ------------------------------------------------------------------
    # Define model: Flatten → FC → LN → ReLU → FC → LN → ReLU → FC → Sigmoid
    # Input: (batch, 16, 96)  Output: (batch, 1)
    # ------------------------------------------------------------------
    torch.manual_seed(42)  # Deterministic weight init for reproducible training
    model = nn.Sequential(
        nn.Flatten(),                               # → (batch, 1536)
        nn.Linear(16 * 96, LAYER_DIM),
        nn.LayerNorm(LAYER_DIM), nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(LAYER_DIM, LAYER_DIM),
        nn.LayerNorm(LAYER_DIM), nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(LAYER_DIM, 1),
        # No Sigmoid here — BCEWithLogitsLoss takes raw logits
    )
    # Inference model wraps with Sigmoid for ONNX export (expects 0-1 output)
    inference_model = nn.Sequential(model, nn.Sigmoid())

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Cosine annealing: LR decays from 1e-3 to 1e-5 over all epochs.
    # Lets the model explore broadly early, then fine-tune decision boundaries.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )
    # Compensate for class imbalance: weight positive examples proportionally
    # so the model doesn't just learn to always predict negative.
    pos_weight = torch.tensor([len(neg_wins) / max(len(pos_wins), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining ({N_EPOCHS} epochs, {len(X)} samples, cosine LR 1e-3 -> 1e-5)...")
    for epoch in range(N_EPOCHS):
        model.train()
        perm_t = torch.randperm(len(X_t))
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, len(X_t), BATCH_SIZE):
            idx = perm_t[i:i + BATCH_SIZE]
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  loss={epoch_loss/n_batches:.4f}")

    # ------------------------------------------------------------------
    # Evaluate on training set
    # ------------------------------------------------------------------
    inference_model.eval()
    with torch.no_grad():
        all_preds = inference_model(X_t).squeeze().numpy()

    pos_scores = all_preds[y == 1.0]
    neg_scores = all_preds[y == 0.0]
    print(f"\nTraining scores:")
    print(f"  Positive  mean={pos_scores.mean():.3f}  min={pos_scores.min():.3f}")
    print(f"  Negative  mean={neg_scores.mean():.3f}  max={neg_scores.max():.3f}")

    # Per-file positive score breakdown (find low-scoring positives)
    pos_mask = y == 1.0
    pos_file_idx_arr = file_idx[pos_mask]
    low_pos_files = []
    for file_i in sorted(set(pos_file_idx_arr)):
        mask = pos_file_idx_arr == file_i
        file_scores = pos_scores[mask]
        file_min = float(file_scores.min())
        if file_min < 0.3:
            low_pos_files.append((file_min, float(file_scores.max()), Path(all_pos_files[file_i]).name))
    if low_pos_files:
        low_pos_files.sort()
        print(f"\n  Low-scoring positive files (min < 0.3):")
        for fmin, fmax, fname in low_pos_files:
            print(f"    min={fmin:.3f}  max={fmax:.3f}  {fname}")

    if pos_scores.mean() < 0.7:
        print("  Warning: positive scores are low — consider recording more samples.")
    if neg_scores.max() > 0.5:
        print("  Warning: some negatives score high — consider adding real negative recordings.")
        # Identify which negative FILES are scoring high (per-clip max score)
        if NEGATIVE_DIR.exists():
            neg_files = sorted(NEGATIVE_DIR.glob("*.wav"))
            if neg_files:
                print("\n  Top-scoring negative files (possible contamination):")
                for path in neg_files:
                    try:
                        audio = pad_or_trim(load_wav(path), CLIP_SAMPLES)
                        emb = np.stack([embed_clip(audio)])  # (1, T, 96)
                        wins = build_negative_windows(emb, stride=1)
                        with torch.no_grad():
                            scores = inference_model(torch.from_numpy(wins)).squeeze().numpy()
                        peak = float(np.atleast_1d(scores).max())
                        if peak > 0.5:
                            print(f"    {peak:.3f}  {path.name}")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Export to ONNX (kept for backward compatibility / inspection)
    # The output node name (output_names) becomes the key in model.predict().
    # ------------------------------------------------------------------
    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    dummy_input = torch.rand(1, 16, 96)

    torch.onnx.export(
        inference_model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=[MODEL_NAME],
        dynamic_axes={"input": {0: "batch"}, MODEL_NAME: {0: "batch"}},
        dynamo=False,  # use legacy exporter (avoids emoji crash on Windows cp1252)
    )
    print(f"\nONNX saved: {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB)")

    # ------------------------------------------------------------------
    # Export to TFLite (required by wyoming-openwakeword)
    # ------------------------------------------------------------------
    # wyoming-openwakeword only loads .tflite (not .onnx). Filename convention
    # is `<name>_v<version>.tflite`. The TFLite must have a *fixed* batch
    # dimension of 1 — dynamic batch loads silently but inference returns
    # nothing, a known issue with the TFLite runtime for dynamic shapes.
    #
    # Re-export ONNX with fixed batch=1, then convert with onnx2tf.
    tflite_path = OUTPUT_DIR / f"{MODEL_NAME}_v0.1.tflite"
    _export_tflite(inference_model, dummy_input, tflite_path)

    print(f"\nDeploy to the Pi5 wyoming-satellite host:")
    print(f"  scp {tflite_path} samda@10.0.30.5:/tmp/")
    print(f"  # On the Pi5:")
    print(f"  cp /tmp/{tflite_path.name} ~/wyoming-openwakeword/custom-models/")
    print(f"  sudo systemctl restart wyoming-openwakeword wyoming-satellite")
    print(f"\nTune --threshold / --trigger-level in the wyoming-openwakeword service")
    print(f"file if you get false positives or misses. Threshold 0.5 with trigger-level 1")
    print(f"is a reasonable starting point.")
    print(f"More real negatives in wakeword/samples/negative/ will help with accuracy.")


def _export_tflite(inference_model, dummy_input, tflite_path: Path) -> None:
    """Convert the PyTorch inference model to TFLite via ONNX + onnx2tf.

    Forces batch=1 on the intermediate ONNX to avoid a known dynamic-batch
    inference failure in wyoming-openwakeword's TFLite runtime.
    """
    try:
        import subprocess
        import shutil
        import sys
        import tempfile
        import torch
    except ImportError as e:
        print(f"  Skipping TFLite export: {e}")
        return

    # Check onnx2tf is importable as a module (works regardless of PATH).
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print("\n  TFLite export skipped: onnx2tf not installed.")
        print("  Install with: pip install onnx2tf tensorflow tf-keras onnxruntime")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixed_onnx = tmp_path / "fixed_batch.onnx"

        # Re-export ONNX with FIXED batch=1 (no dynamic_axes).
        # onnx2tf --batch-size 1 alone doesn't always force input signature fixed.
        torch.onnx.export(
            inference_model,
            dummy_input,
            str(fixed_onnx),
            opset_version=18,
            input_names=["input"],
            output_names=[MODEL_NAME],
            # No dynamic_axes — batch is fixed at 1.
            dynamo=False,
        )

        # Convert with onnx2tf. -b 1 forces batch=1.
        # -kat input keeps the input tensor in its original (1, 16, 96) shape
        # — without it, onnx2tf transposes to NHWC (1, 96, 16) and
        # wyoming-openwakeword feeds frames in the wrong order, producing
        # ~zero scores on real audio.
        result = subprocess.run(
            [sys.executable, "-m", "onnx2tf",
             "-i", str(fixed_onnx), "-o", str(tmp_path / "out"),
             "-b", "1", "-kat", "input"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print(f"  onnx2tf failed (exit {result.returncode}):")
            print("--- stdout ---")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print("--- stderr ---")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            return

        # onnx2tf produces several variants — float32 is the one we want.
        candidates = sorted((tmp_path / "out").glob("*_float32.tflite"))
        if not candidates:
            # Fallback: any .tflite
            candidates = sorted((tmp_path / "out").glob("*.tflite"))
        if not candidates:
            print(f"  No .tflite files produced in {tmp_path / 'out'}")
            return

        shutil.copy(candidates[0], tflite_path)
        print(f"TFLite saved: {tflite_path}  ({tflite_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
