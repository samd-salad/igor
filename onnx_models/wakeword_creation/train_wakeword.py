#!/usr/bin/env python3
"""Train an OpenWakeWord ONNX model from recorded positive samples.

Workflow:
  1. Pi:   python record_samples.py           → wakeword_samples/positive/*.wav
  2. PC:   scp or rsync samples to PC
  3. PC:   python onnx_models/wakeword_creation/train_wakeword.py
  4. Pi:   scp oww_models/igor.onnx to Pi's oww_models/

Dependencies (install once on PC):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install openwakeword

How it works:
  - OWW's frozen backbone (melspectrogram + embedding models, auto-downloaded ~50 MB)
    converts audio into 96-dimensional embedding vectors.
  - A small 2-layer PyTorch DNN learns to classify those embeddings as
    wake word (1) vs not (0).
  - Exported to ONNX; the output node name becomes the key in model.predict().

Negative samples:
  This script uses synthetic negatives (noise, silence, sine tones) which work
  reasonably well for quiet environments. For fewer false positives in noisy
  conditions, record real negative audio (background speech, TV, music) into
  wakeword_samples/negative/ and re-train.
"""

import sys
import wave
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent.parent
POSITIVE_DIR = ROOT / "wakeword_samples" / "positive"
NEGATIVE_DIR = ROOT / "wakeword_samples" / "negative"  # optional real negatives
OUTPUT_DIR  = ROOT / "oww_models"
MODEL_NAME  = "igor"   # becomes the key in model.predict() results

SAMPLE_RATE  = 16000
CLIP_SAMPLES = SAMPLE_RATE * 3  # 3-second clips (pad/trim all audio to this)
N_EPOCHS     = 200
BATCH_SIZE   = 64
LAYER_DIM    = 64   # increased from 32 — more capacity to discriminate real-world audio


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
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(audio.dtype)


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
        from openwakeword.utils import AudioFeatures
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install openwakeword")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load positive clips
    # ------------------------------------------------------------------
    print("\nLoading positive clips (peak-normalized to 16000)...")
    pos_audio = []
    for path in wav_files:
        try:
            audio = normalize_peak(pad_or_trim(load_wav(path), CLIP_SAMPLES))
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

    # Real negatives from wakeword_samples/negative/ (optional)
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
    # Extract embeddings via OWW frozen backbone
    # ------------------------------------------------------------------
    # print("\nExtracting embeddings (downloading base models ~50 MB on first run)...")
    # _download_backbone_models()
    feat = AudioFeatures(sr=SAMPLE_RATE, ncpu=4, inference_framework="onnx")

    pos_emb = feat.embed_clips(pos_audio, batch_size=64)   # (N, 16, 96)
    neg_emb = feat.embed_clips(neg_audio, batch_size=64)   # (M, 16, 96)

    n_frames = pos_emb.shape[1]
    print(f"  Positive: {pos_emb.shape}  Negative: {neg_emb.shape}")

    # Slice into 16-frame windows (OWW inference window size).
    # A clip of N frames yields (N - 16 + 1) windows — free data augmentation.
    WINDOW = 16
    # Cosine similarity threshold: auto clips below this are skipped (no wake word found)
    SIMILARITY_THRESHOLD = 0.7
    # How many windows to extract around the peak similarity position
    PEAK_CONTEXT = 3  # windows ending at peak-1, peak, peak+1

    def make_windows(emb: np.ndarray) -> np.ndarray:
        """(clips, frames, 96) → (windows, 16, 96)"""
        wins = []
        for clip in emb:
            for start in range(len(clip) - WINDOW + 1):
                wins.append(clip[start:start + WINDOW])
        return np.stack(wins).astype(np.float32)

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two flattened vectors."""
        a_flat, b_flat = a.flatten(), b.flatten()
        denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a_flat, b_flat) / denom)

    # --- Build template embedding from reference samples (sample_*.wav) ---
    # These are manually recorded, tightly trimmed — wake word present throughout.
    ref_indices = [i for i, p in enumerate(wav_files) if Path(p).name.startswith("sample_")]
    auto_indices = [i for i, p in enumerate(wav_files)
                    if Path(p).name.startswith(("auto_", "dup_auto_", "pc_auto_"))]
    other_indices = [i for i in range(len(wav_files)) if i not in ref_indices and i not in auto_indices]

    if not ref_indices:
        print("  Warning: no sample_*.wav reference files found — falling back to all-windows for auto clips")
        template_emb = None
    else:
        # Average the last WINDOW frames of each reference clip (right-aligned, where OWW expects the keyword)
        ref_windows = []
        for i in ref_indices:
            clip = pos_emb[i]
            # Use the last 16-frame window (right-aligned keyword)
            ref_windows.append(clip[-WINDOW:])
        template_emb = np.mean(ref_windows, axis=0)  # (16, 96) average template
        print(f"  Built template from {len(ref_indices)} reference samples (sample_*.wav)")

    # --- Extract positive windows ---
    pos_wins_list = []
    pos_win_file_idx = []  # track which file each window came from
    n_auto_matched = 0
    n_auto_skipped = 0

    for i, (clip, path) in enumerate(zip(pos_emb, wav_files)):
        n = len(clip)
        max_start = n - WINDOW
        if max_start < 0:
            continue
        name = Path(path).name
        is_auto = name.startswith(("auto_", "dup_auto_", "pc_auto_"))

        if is_auto and template_emb is not None:
            # Slide window across clip, find peak cosine similarity to template
            best_sim = -1.0
            best_end = max_start  # fallback: last window
            for start in range(max_start + 1):
                window = clip[start:start + WINDOW]
                sim = _cosine_sim(window, template_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_end = start  # start index of best window

            if best_sim < SIMILARITY_THRESHOLD:
                n_auto_skipped += 1
                continue  # No region matches the wake word — skip this clip

            n_auto_matched += 1
            # Extract PEAK_CONTEXT windows centered on the best match
            for offset in range(-PEAK_CONTEXT // 2, PEAK_CONTEXT // 2 + 1):
                s = best_end + offset
                if 0 <= s <= max_start:
                    pos_wins_list.append(clip[s:s + WINDOW])
                    pos_win_file_idx.append(i)
        else:
            # Reference samples or other files: wake word present throughout, use all windows
            for start in range(max_start + 1):
                pos_wins_list.append(clip[start:start + WINDOW])
                pos_win_file_idx.append(i)

    if auto_indices:
        print(f"  Auto clips: {n_auto_matched} matched (sim>={SIMILARITY_THRESHOLD}), {n_auto_skipped} skipped")

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
        nn.Dropout(0.2),
        nn.Linear(LAYER_DIM, LAYER_DIM),
        nn.LayerNorm(LAYER_DIM), nn.ReLU(),
        nn.Dropout(0.2),
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
                feat2 = AudioFeatures(sr=SAMPLE_RATE, ncpu=1, inference_framework="onnx")
                for path in neg_files:
                    try:
                        audio = pad_or_trim(load_wav(path), CLIP_SAMPLES)
                        emb = feat2.embed_clips(np.stack([audio]), batch_size=1)
                        wins = make_windows(emb)
                        with torch.no_grad():
                            scores = inference_model(torch.from_numpy(wins)).squeeze().numpy()
                        peak = float(np.atleast_1d(scores).max())
                        if peak > 0.5:
                            print(f"    {peak:.3f}  {path.name}")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Export to ONNX
    # The output node name (output_names) becomes the key in model.predict().
    # ------------------------------------------------------------------
    output_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    dummy_input = torch.rand(1, 16, 96)

    torch.onnx.export(
        inference_model,
        dummy_input,
        str(output_path),
        opset_version=18,
        input_names=["input"],
        output_names=[MODEL_NAME],
        dynamic_axes={"input": {0: "batch"}, MODEL_NAME: {0: "batch"}},
        dynamo=False,  # use legacy exporter (avoids emoji crash on Windows cp1252)
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"\nModel saved: {output_path}  ({size_kb:.0f} KB)")
    print(f"\nDeploy to Pi:")
    print(f"  scp {output_path} pi@<PI_IP>:~/smart_assistant/oww_models/")
    print(f"\nThen restart the client:")
    print(f"  .venv/bin/python -m client.main")
    print(f"\nTune OWW_THRESHOLD in client/config.py if you get false positives or misses.")
    print(f"More real negatives in wakeword_samples/negative/ will help with accuracy.")


if __name__ == "__main__":
    main()
