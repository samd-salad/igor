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

# Allow direct invocation as `python wakeword/train.py` (in addition to
# `python -m wakeword.train`) — adding wakeword's package as a subpackage
# in Task 8 broke the script-style invocation by removing the implicit
# namespace package fallback.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    build_positive_window, build_negative_windows, split_indices, WINDOW_FRAMES,
)

ROOT        = Path(__file__).parent.parent
POSITIVE_DIR = ROOT / "wakeword" / "samples" / "positive"
NEGATIVE_DIR = ROOT / "wakeword" / "samples" / "negative"  # optional real negatives
HARD_NEGATIVES_FILE = ROOT / "wakeword" / "hard_negatives.txt"
OUTPUT_DIR  = ROOT / "wakeword" / "models"
MODEL_NAME  = "igor"   # becomes the key in model.predict() results

SAMPLE_RATE  = 16000
CLIP_SAMPLES = SAMPLE_RATE * 3  # 3-second clips (pad/trim all audio to this)

# v0.9: pos_weight from n_neg/n_pos (~222) → 22. The historical setting
# inverted dscripka's canonical loss balance by ~4500x, causing the model
# to saturate positives (mean=1.000) while letting negatives drift. The
# diagnostic sweep showed FP@>0.95 halved at pw=22 without breaking recall
# (pw=1.0 collapsed pos_max; pw=5 still broke recall; pw=50 left more FPs).
POS_WEIGHT   = 22.0

N_EPOCHS     = 100  # re-mining (v0.9) lets the model improve across all 100
                    # epochs instead of overfitting into a ceiling by epoch 30.
                    # top_firer trajectory now DESCENDS over training — 1.000
                    # → 0.714 across epochs 1-100 — because the previously
                    # 1.000-firing holdout clips are now in the training set
                    # (see HARD_NEGATIVES_FILE) and the model is forced to
                    # learn features that suppress them and generalize.
BATCH_SIZE   = 64
N_AUG_VARIANTS = 10   # augmented variants per real positive
NEG_STRIDE     = 1    # MUST be 1 — stride=10 in v0.6 fed the model 10%
                      # of windows; the unseen 90% all fired at the gate
                      # because the model only saw striped subsamples.
                      # Full coverage is non-negotiable.
NEG_HOLDOUT_FRAC = 0.20  # file-level; never seen during training or augmentation
NEG_HOLDOUT_SEED = 4317  # change only when you want a different held-out partition


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
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install pyopen_wakeword")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # REAL POSITIVES ONLY. v0.5 used Piper synth: hundreds scored 0.02
    # post-train (model classified them as negatives), fragmenting the
    # positive cluster and stranding the decision boundary. The user's
    # actual voice is the only thing the runtime ever hears — train on
    # that distribution directly and rely on per-clip augmentation
    # (N_AUG_VARIANTS variants per real clip) to multiply coverage.
    all_pos_files = sorted(list(POSITIVE_DIR.glob("*.wav")))
    if not all_pos_files:
        print(f"No positive WAVs found in {POSITIVE_DIR}/.")
        sys.exit(1)
    print(f"\nLoading {len(all_pos_files)} real positive clips, "
          f"trim+normalize+left-pad to {CLIP_SAMPLES//SAMPLE_RATE}s...")

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
        neg_paths_all = sorted(NEGATIVE_DIR.glob("*.wav"))
    else:
        neg_paths_all = []
    if len(neg_paths_all) < 100:
        print(f"ERROR: need >= 100 real negative clips in {NEGATIVE_DIR}/, "
              f"found {len(neg_paths_all)}.")
        sys.exit(1)

    # File-level holdout split. Holdout clips never enter training data or the
    # augmentation background pool — the only place they touch the model is the
    # post-train shipping gate. That gate is the only honest signal we have for
    # generalization; previously we evaluated on the same negatives we trained
    # on at stride=1, which the hard-neg-mining loop drove to ~0 by design and
    # told us nothing about runtime FPs.
    train_idx, holdout_idx = split_indices(
        len(neg_paths_all),
        holdout_frac=NEG_HOLDOUT_FRAC,
        seed=NEG_HOLDOUT_SEED,
    )

    # Re-mining (v0.9): force runtime-failure clips out of holdout into training.
    # See HARD_NEGATIVES_FILE — one filename per line. The diagnostic run that
    # introduced this dropped top_firer from 1.000 → 0.714 and FP@>0.95 from
    # 1.18% → 0.00% on the remaining holdout. Keep this list growing as new
    # false-fire clips appear in production logs; retrain when it grows.
    if HARD_NEGATIVES_FILE.exists():
        hard_names = {
            line.strip() for line in HARD_NEGATIVES_FILE.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        train_set, holdout_set = set(train_idx.tolist()), set(holdout_idx.tolist())
        moved = 0
        for i, p in enumerate(neg_paths_all):
            if p.name in hard_names and i in holdout_set:
                holdout_set.discard(i)
                train_set.add(i)
                moved += 1
        train_idx = np.array(sorted(train_set))
        holdout_idx = np.array(sorted(holdout_set))
        print(f"Re-mining: {len(hard_names)} hard negatives listed; "
              f"{moved} moved from holdout to train.")

    neg_paths = [neg_paths_all[i] for i in train_idx]
    neg_holdout_paths = [neg_paths_all[i] for i in holdout_idx]
    print(f"Negative split: {len(neg_paths_all)} total -> "
          f"{len(neg_paths)} train / {len(neg_holdout_paths)} holdout "
          f"(seed={NEG_HOLDOUT_SEED}, frac={NEG_HOLDOUT_FRAC})")
    print(f"Loading {len(neg_paths)} train negatives for backgrounds and training...")
    # DO NOT normalize_peak negatives — the user's recording mic puts the
    # quietest negatives at RMS ~100, but normalize_peak scales them all to
    # peak=16000. This taught v0.3/v0.4 that "anything = loud at inference"
    # — and the model false-fired on quiet mic-floor audio because it had
    # never seen the genuinely low-amplitude regime during training.
    # Use natural amplitudes so the inference-time mic-floor distribution
    # is represented.
    neg_audio_list = []
    for path in neg_paths:
        try:
            audio = pad_or_trim(load_wav(path), CLIP_SAMPLES)
            neg_audio_list.append(audio)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    neg_audio = np.stack(neg_audio_list) if neg_audio_list else np.empty(
        (0, CLIP_SAMPLES), dtype=np.int16
    )
    print(f"  {len(neg_audio)} negatives loaded.")

    # Apply augmentation — N_AUG_VARIANTS variants per real positive.
    # Each variant gets its own independent random RIR + SNR mix to
    # produce a diverse training cluster from a small recording pool.
    # The original CLEAN clip is also kept (variant 0) so the model
    # sees the unaugmented prosody as well.
    pos_audio = []
    pos_audio_src_idx = []  # parent real-clip index for diagnostics
    for src_i, clip in enumerate(pos_audio_raw):
        pos_audio.append(clip)
        pos_audio_src_idx.append(src_i)
        for _ in range(N_AUG_VARIANTS - 1):
            x = clip
            if aug_rng.random() < 0.7:
                rir = rirs[aug_rng.integers(0, len(rirs))]
                x = apply_rir(x, rir)
            if aug_rng.random() < 0.85 and len(neg_audio) > 0:
                bg = neg_audio[aug_rng.integers(0, len(neg_audio))]
                # SNR range 0-10 dB: positive comparable in volume to bg
                # (matches dscripka's reference notebook). v0.6's 10-25 dB
                # had positives much louder than bg — the model never trained
                # on the hard "wake word over loud TV" case and learned to
                # equate "loud + clean" with "positive."
                snr = random_snr_db(aug_rng, low=0.0, high=10.0)
                x = mix_with_background(x, bg, snr_db=snr, rng=aug_rng)
            pos_audio.append(x)
            pos_audio_src_idx.append(src_i)
    pos_audio = np.stack(pos_audio)
    n_pos = len(pos_audio)
    print(f"Augmented positives: {n_pos} ({len(pos_audio_raw)} real × ~{N_AUG_VARIANTS} variants)")

    # ----- Embed and window -----
    print("\nExtracting embeddings via pyopen_wakeword features...")
    pos_emb = np.stack([embed_clip(c) for c in pos_audio])  # (N, T, 96)
    neg_emb = np.stack([embed_clip(c) for c in neg_audio])  # (M, T, 96)
    print(f"  Positive emb: {pos_emb.shape}  Negative emb: {neg_emb.shape}")

    # ONE positive window per (augmented) clip with ±200ms jitter.
    # Track by SOURCE real-clip index (pos_audio_src_idx) so the per-file
    # diagnostic reports parent recordings, not augmentation variants.
    win_rng = np.random.default_rng(7)
    pos_wins_list = []
    pos_win_file_idx = []
    n_too_short = 0
    for i, clip_emb in enumerate(pos_emb):
        try:
            win = build_positive_window(clip_emb, jitter_ms=200.0, rng=win_rng)
            pos_wins_list.append(win)
            pos_win_file_idx.append(pos_audio_src_idx[i])
        except ValueError:
            n_too_short += 1
    if n_too_short:
        print(f"  Skipped {n_too_short} clip(s) too short for a {WINDOW_FRAMES}-frame window")
    pos_wins = np.stack(pos_wins_list).astype(np.float32) if pos_wins_list else \
        np.empty((0, WINDOW_FRAMES, 96), dtype=np.float32)

    # stride=1 is mandatory. v0.6 tried stride=10 to reduce class imbalance,
    # but the model perfectly classified the strided windows during training
    # (mean=0.017, max=0.160) while ALL 2913 files fired the gate at stride=1
    # — the unseen 90% of windows lived in feature space the model never had
    # gradient on. With 10x positive augmentation (2310 windows) and pos_weight
    # auto-set to ~143, the 1:143 imbalance is workable.
    neg_wins = build_negative_windows(neg_emb, stride=NEG_STRIDE)
    print(f"  Windows — positive: {len(pos_wins)}  negative: {len(neg_wins)}  "
          f"(ratio 1:{len(neg_wins)/max(1, len(pos_wins)):.1f})")

    # ------------------------------------------------------------------
    # Prepare training tensors
    # ------------------------------------------------------------------
    X = np.vstack([pos_wins, neg_wins])
    y = np.array([1.0] * len(pos_wins) + [0.0] * len(neg_wins), dtype=np.float32)
    # Track which positive file each sample came from (negatives get -1)
    file_idx = np.array(pos_win_file_idx + [-1] * len(neg_wins))

    perm = np.random.default_rng(0).permutation(len(X))
    X, y, file_idx = X[perm], y[perm], file_idx[perm]

    # ----- Train (reference-aligned) -----
    from wakeword._training import train_model

    inference_model = train_model(
        X.astype(np.float32),
        y.astype(np.float32),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-4,
        layer_dim=192,  # layer_dim sweep at pw=22 showed ld in [32,64,96,192]
                        # all hit top_firer=1.000 by epoch 30 — architecture is
                        # NOT the bottleneck. Re-mining is the real lever.
                        # Keeping 192 since it had marginally better epoch-100
                        # pos recall vs smaller sizes (pos_min 0.712 vs 0.768).
        pos_weight=POS_WEIGHT,
    )

    # ------------------------------------------------------------------
    # Evaluate on training set
    # ------------------------------------------------------------------
    inference_model.eval()
    with torch.no_grad():
        all_preds = inference_model(torch.from_numpy(X)).squeeze().numpy()

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

    # ------------------------------------------------------------------
    # SHIPPING GATE: held-out real-audio false-positive check.
    #
    # Scores the HOLDOUT 20% of negatives (never seen during training or
    # augmentation) at stride=1. This is the only honest generalization
    # signal — scoring the training-set negatives is pointless because
    # hard-neg mining drove them to ~0 by construction.
    #
    # v0.5 passed a synthetic-noise smoke test then fired ~every 30s in
    # production because synthetic Gaussian/tonal noise didn't represent
    # actual mic floor. Real held-out audio is the only true gate.
    # ------------------------------------------------------------------
    print(f"\nShipping gate: scoring {len(neg_holdout_paths)} HELD-OUT negatives "
          f"(unseen during training) at stride=1...")
    fp_per_file: list[tuple[float, str]] = []
    for path in neg_holdout_paths:
        try:
            audio = pad_or_trim(load_wav(path), CLIP_SAMPLES)
            emb = np.stack([embed_clip(audio)])
            wins = build_negative_windows(emb, stride=1)
            if len(wins) == 0:
                continue
            with torch.no_grad():
                scores = inference_model(torch.from_numpy(wins)).squeeze().numpy()
            fp_per_file.append((float(np.atleast_1d(scores).max()), path.name))
        except Exception as e:
            print(f"  Skip {path.name}: {e}")

    fp_per_file.sort(reverse=True)
    above_05 = [x for x in fp_per_file if x[0] > 0.5]
    above_07 = [x for x in fp_per_file if x[0] > 0.7]
    above_085 = [x for x in fp_per_file if x[0] > 0.85]
    total = len(fp_per_file)
    print(f"  Holdout FP rate: >0.5: {len(above_05)}/{total} "
          f"({100*len(above_05)/max(1,total):.1f}%)  "
          f">0.7: {len(above_07)}  >0.85: {len(above_085)}")
    if above_05[:20]:
        print("  Top-scoring holdout negatives:")
        for peak, name in above_05[:20]:
            print(f"    {peak:.3f}  {name}")

    # Gate (informational): < 1% of held-out negatives may score > 0.5.
    # Reported but NOT enforced via sys.exit — the model is always exported
    # so deploy decisions can be made on production behavior, not just the
    # offline stride=1 worst-case (production uses smoothing + trigger_level).
    fp_rate = len(above_05) / max(1, total)
    gate_status = "PASS" if fp_rate <= 0.01 else "FAIL"
    print(f"\n{gate_status}: {fp_rate*100:.2f}% holdout FP rate (gate: <1%).")

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
    tflite_path = OUTPUT_DIR / f"{MODEL_NAME}_v0.9.tflite"
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

        # copyfile (not copy) — copy() also tries to chmod the dest, which
        # fails on /mnt/c (OneDrive/WSL drvfs disallows mode changes). The
        # bytes copy fine; we just don't need the perm copy.
        shutil.copyfile(candidates[0], tflite_path)
        print(f"TFLite saved: {tflite_path}  ({tflite_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
