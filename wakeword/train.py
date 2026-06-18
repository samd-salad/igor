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
import wave
from pathlib import Path

import numpy as np

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

def load_wav(path: Path) -> np.ndarray:
    """Load a 16 kHz mono WAV file as int16 numpy array (raw PCM)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected 16 kHz, got {sr} Hz")
        if ch != 1:
            raise ValueError(f"Expected mono, got {ch} channels")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).copy()


def normalize_peak(audio: np.ndarray, target_peak: int = 16000) -> np.ndarray:
    """Normalize audio to a consistent peak amplitude.

    Makes the model amplitude-invariant so it works across mics with
    different gain levels (e.g. Pi mic at peak ~20000 vs PC BlackShark at ~3000).
    """
    peak = int(np.max(np.abs(audio)))
    if peak < 10:
        return audio  # Digital silence — don't amplify
    gain = min(target_peak / peak, 1000.0)
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Right-pad with silence (for negatives — silence at end is fine)."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(audio.dtype)


def left_pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Left-pad with silence (for positives — keeps wake word at the END of
    the clip, matching how wyoming-openwakeword sees rolling buffer at runtime:
    the most-recent N frames are the inference window. If the wake word is at
    the start of training clips and we right-pad, the model learns 'silence
    after wake word = positive' which causes phantom fires on cold start)."""
    if len(audio) >= length:
        return audio[-length:]
    return np.pad(audio, (length - len(audio), 0)).astype(audio.dtype)


def trim_trailing_silence(audio: np.ndarray, threshold: int = 250,
                          keep_tail_samples: int = 1600) -> np.ndarray:
    """Strip silence after the last energetic sample.

    record_samples.py records 2 s; the user says "Igor" early then leaves
    silence at the tail. Trimming that silence before left-padding ensures
    the wake word actually lands at the end of the 3-sec training clip, so
    "last K windows" labeling captures the wake word.

    Keeps `keep_tail_samples` (~100 ms) after the last energetic sample so
    the final syllable doesn't get clipped.
    """
    abs_audio = np.abs(audio)
    nonzero = np.where(abs_audio > threshold)[0]
    if len(nonzero) == 0:
        return audio  # all silence — leave as-is (will be discarded later)
    last_real = nonzero[-1]
    end = min(len(audio), last_real + keep_tail_samples)
    return audio[:end]


def generate_synthetic_negatives(n: int, clip_length: int) -> np.ndarray:
    """
    Generate synthetic negative clips: Gaussian noise, silence, and sine tones.
    These are sufficient for quiet environments; for noisy rooms record real negatives.
    """
    rng = np.random.default_rng(42)
    clips = []
    third = n // 3

    def to_int16(f: np.ndarray) -> np.ndarray:
        return np.clip(f * 32767, -32768, 32767).astype(np.int16)

    # White noise at varied amplitudes
    for amp in np.linspace(0.005, 0.35, third):
        c = rng.normal(0, float(amp), clip_length).astype(np.float32)
        clips.append(to_int16(c))

    # Near-silence (just noise floor)
    for _ in range(third):
        c = rng.normal(0, 0.001, clip_length).astype(np.float32)
        clips.append(to_int16(c))

    # Sine tones
    remainder = n - 2 * third
    for freq in np.linspace(80, 5000, remainder):
        t = np.arange(clip_length, dtype=np.float32) / SAMPLE_RATE
        amp = float(rng.uniform(0.05, 0.3))
        clips.append(to_int16(amp * np.sin(2 * np.pi * freq * t)))

    return np.stack(clips[:n])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wav_files = sorted(POSITIVE_DIR.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {POSITIVE_DIR}/")
        print("Record samples first: python record_samples.py  (on the Pi)")
        sys.exit(1)

    print(f"Found {len(wav_files)} positive samples.")
    if len(wav_files) < 50:
        print(f"  Warning: {len(wav_files)} is low — aim for 100+ for reliable detection.")

    # Check dependencies
    try:
        import torch
        import torch.nn as nn
        from pyopen_wakeword import OpenWakeWordFeatures
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install pyopen_wakeword")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load positive clips — LEFT-PADDED so the wake word lands at the end
    # of the 3-sec clip (matches wyoming-openwakeword's rolling-buffer
    # inference: it sees the most recent ~370ms as the classifier window).
    # Right-padding here is what trained the "silence-prefix = positive"
    # bug that caused phantom fires at session start.
    # ------------------------------------------------------------------
    print("\nLoading positive clips (trim trailing silence, peak-normalize, left-pad)...")
    pos_audio = []
    for path in wav_files:
        try:
            raw = load_wav(path)
            trimmed = trim_trailing_silence(raw)
            audio = normalize_peak(left_pad_or_trim(trimmed, CLIP_SAMPLES))
            pos_audio.append(audio)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")

    if not pos_audio:
        print("No valid clips loaded. Check that files are 16 kHz mono WAV.")
        sys.exit(1)

    pos_audio = np.stack(pos_audio)   # (N, samples)
    n_pos = len(pos_audio)

    # ------------------------------------------------------------------
    # Load or generate negative clips
    # ------------------------------------------------------------------
    neg_audio_list = []

    # Real negatives from wakeword/samples/negative/ (optional)
    if NEGATIVE_DIR.exists():
        neg_files = sorted(NEGATIVE_DIR.glob("*.wav"))
        if neg_files:
            print(f"Loading {len(neg_files)} real negative clips from {NEGATIVE_DIR.name}/...")
            for path in neg_files:
                try:
                    audio = normalize_peak(pad_or_trim(load_wav(path), CLIP_SAMPLES))
                    neg_audio_list.append(audio)
                except Exception as e:
                    print(f"  Skipping {path.name}: {e}")

    # Synthetic negatives to reach at least 3× positive count
    n_synthetic = max(0, n_pos * 3 - len(neg_audio_list))
    if n_synthetic > 0:
        print(f"Generating {n_synthetic} synthetic negative clips (noise/silence/tones)...")
        synthetic = generate_synthetic_negatives(n_synthetic, CLIP_SAMPLES)
        neg_audio_list.append(synthetic)

    neg_audio = np.vstack(neg_audio_list) if neg_audio_list else np.empty((0, CLIP_SAMPLES), dtype=np.float32)
    n_neg = len(neg_audio)
    print(f"  Total negatives: {n_neg}")

    # ------------------------------------------------------------------
    # Extract embeddings via pyopen_wakeword's feature pipeline
    # ------------------------------------------------------------------
    # IMPORTANT: We use pyopen_wakeword here (not the original openwakeword
    # library) because wyoming-openwakeword switched to it in mid-2025. The
    # two libraries emit features at different rates (~43Hz vs ~9.3Hz) and
    # with different distributions. A model trained against the wrong pipeline
    # saturates to ~1.0 on production cold-start audio, causing
    # "Detected at 0" phantom fires every session.
    print("\nExtracting embeddings via pyopen_wakeword features...")
    feats = OpenWakeWordFeatures.from_builtin()
    CHUNK_BYTES = 320  # 10 ms @ 16 kHz, 16-bit mono

    def embed_clip(audio_int16: np.ndarray) -> np.ndarray:
        """Stream a single clip through pyopen_wakeword features.
        Returns (T, 96) where T depends on clip length (~43 frames per second)."""
        feats.reset()
        raw = audio_int16.astype(np.int16).tobytes()
        out = []
        for i in range(0, len(raw) - CHUNK_BYTES + 1, CHUNK_BYTES):
            for f in feats.process_streaming(raw[i:i+CHUNK_BYTES]):
                out.append(f.squeeze())  # (96,)
        if not out:
            return np.empty((0, 96), dtype=np.float32)
        return np.stack(out).astype(np.float32)

    pos_emb = np.stack([embed_clip(c) for c in pos_audio])  # (N, T, 96)
    neg_emb = np.stack([embed_clip(c) for c in neg_audio])  # (M, T, 96)

    n_frames = pos_emb.shape[1]
    print(f"  Positive: {pos_emb.shape}  Negative: {neg_emb.shape}")
    print(f"  Frame rate: {n_frames / 3.0:.1f} features/sec ({3000/n_frames:.1f} ms hop)")

    # Slice into 16-frame windows (OWW inference window size).
    WINDOW = 16
    # For positives, take only the LAST N windows of each clip. After trimming
    # trailing silence + left-padding, the wake word lives at the end of the
    # 3-sec clip; the last N windows are the ones whose 374 ms span actually
    # overlaps it. Labeling earlier windows (silence/zero-padding) positive
    # is what causes phantom fire-at-0 at session start in production.
    POSITIVE_WINDOWS_PER_CLIP = 25  # ~580 ms of end-of-clip context

    def make_windows(emb: np.ndarray) -> np.ndarray:
        """(clips, frames, 96) → (windows, 16, 96). All windows of all clips."""
        wins = []
        for clip in emb:
            for start in range(len(clip) - WINDOW + 1):
                wins.append(clip[start:start + WINDOW])
        return np.stack(wins).astype(np.float32)

    # --- Extract positive windows (only end-of-clip windows) ---
    pos_wins_list = []
    pos_win_file_idx = []
    n_too_short = 0
    for i, clip in enumerate(pos_emb):
        max_start = len(clip) - WINDOW
        if max_start < 0:
            n_too_short += 1
            continue
        # Last K windows whose start index is in [max_start - K + 1, max_start]
        first = max(0, max_start - POSITIVE_WINDOWS_PER_CLIP + 1)
        for start in range(first, max_start + 1):
            pos_wins_list.append(clip[start:start + WINDOW])
            pos_win_file_idx.append(i)
    if n_too_short:
        print(f"  Skipped {n_too_short} positive clip(s) too short for a {WINDOW}-frame window")

    pos_wins = np.stack(pos_wins_list).astype(np.float32) if pos_wins_list else np.empty((0, WINDOW, 96), dtype=np.float32)
    neg_wins = make_windows(neg_emb)
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
            low_pos_files.append((file_min, float(file_scores.max()), Path(wav_files[file_i]).name))
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
                        wins = make_windows(emb)
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
