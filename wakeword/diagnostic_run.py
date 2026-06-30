#!/usr/bin/env python3
"""Diagnostic: train 100 epochs with v0.7 hyperparameters, snapshot every 10,
score the holdout at each snapshot, print a curve.

Answers two questions in one run:
  1. Is there a sweet-spot epoch where v0.7's setup actually generalizes?
  2. Or is the holdout FP rate stuck above 1% regardless of epoch —
     evidence that the architecture (not training dynamics) is the bottleneck.

Uses the same data prep + augmentation + windowing as train.py. The only
delta: pre-embeds the holdout ONCE upfront so per-snapshot scoring is fast
(FC inference on cached embeddings, ~1s per snapshot vs ~3min if we
re-embedded each time).

Run from repo root:
    /home/samda/.venvs/igor/bin/python -u wakeword/diagnostic_run.py
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from wakeword._audio import (
    load_wav, normalize_peak, trim_trailing_silence, left_pad_or_trim,
)
from wakeword._augmentation import (
    apply_rir, mix_with_background, random_snr_db, generate_synthetic_rirs,
)
from wakeword._dataset import (
    build_positive_window, build_negative_windows, split_indices, WINDOW_FRAMES,
)
from wakeword._features import embed_clip
from wakeword._training import train_model, WakewordModel


POSITIVE_DIR = REPO / "wakeword" / "samples" / "positive"
NEGATIVE_DIR = REPO / "wakeword" / "samples" / "negative"

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 3

# Match train.py / v0.7 exactly so the curve answers the right question
N_EPOCHS = 100
N_AUG_VARIANTS = 10
NEG_STRIDE = 1
NEG_HOLDOUT_FRAC = 0.20
NEG_HOLDOUT_SEED = 4317
LAYER_DIM = 192
BATCH_SIZE = 64
LR = 1e-4

CHECKPOINT_EVERY = 10
HOLDOUT_THRESHOLDS = (0.5, 0.7, 0.85, 0.95, 0.99)

EXPERIMENTS_FILE = REPO / "wakeword" / "EXPERIMENTS.md"


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(audio.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic train + holdout-score curve")
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="BCE pos_weight. Canonical=1.0, historical=n_neg/n_pos (~222)")
    p.add_argument("--layer-dim", type=int, default=LAYER_DIM,
                   help="FC classifier hidden width. Canonical=32, ours has been 192.")
    p.add_argument("--label", type=str, default=None,
                   help="Human-readable label for this run")
    p.add_argument("--hard-negatives-file", type=str, default=None,
                   help="Path to a text file listing negative WAV filenames "
                        "(one per line, e.g. 'neg_4023.wav') to FORCE into "
                        "training. Files are removed from the holdout if they "
                        "would otherwise have landed there. Used for re-mining: "
                        "take the clips a prior model false-fired on and check "
                        "whether retraining can suppress them without spawning "
                        "new ceiling firers elsewhere.")
    return p.parse_args()


def append_experiment_row(label: str, pos_weight: float, layer_dim: int,
                          n_pos_windows: int, n_neg_windows: int,
                          n_holdout: int, curve_rows: list[dict]) -> None:
    """Append one experiment section to EXPERIMENTS.md so results persist
    outside conversation context. curve_rows: list of dicts with keys
    epoch, pos_min, pos_mean, pos_max, fp_counts (list aligned to
    HOLDOUT_THRESHOLDS), top_firer."""
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # NEW VERDICT (2026-06-29): false fires hurt more than missed wakes —
    # optimize for lowest top_firer (neg_max) subject to pos_max >= 0.9
    # (i.e. AT LEAST ONE positive variant still fires confidently).
    qualifying = [r for r in curve_rows if r["pos_max"] >= 0.9]
    if qualifying:
        best = min(qualifying, key=lambda r: r["top_firer"])
        verdict = (f"BEST epoch {best['epoch']}: "
                   f"top_firer={best['top_firer']:.3f}, "
                   f"pos_max={best['pos_max']:.3f}, "
                   f"FP@>0.95 = {best['fp_counts'][3]}/{n_holdout} "
                   f"({100 * best['fp_counts'][3] / n_holdout:.2f}%)")
    else:
        verdict = "POSITIVE COLLAPSE — no epoch achieved pos_max >= 0.9"

    header_cells = ["epoch", "pos_min", "pos_mean", "pos_max"] + \
                   [f">{t:.2f}" for t in HOLDOUT_THRESHOLDS] + ["top_firer"]
    md_header = "| " + " | ".join(header_cells) + " |"
    md_sep = "|" + "|".join(["---"] * len(header_cells)) + "|"
    md_rows = []
    for r in curve_rows:
        fps = [f"{c}/{n_holdout}" for c in r["fp_counts"]]
        md_rows.append("| " + " | ".join([
            str(r["epoch"]),
            f"{r['pos_min']:.3f}",
            f"{r['pos_mean']:.3f}",
            f"{r['pos_max']:.3f}",
            *fps,
            f"{r['top_firer']:.3f}",
        ]) + " |")

    section = (
        f"\n## {label} — {timestamp}\n\n"
        f"**Hyperparams:** pos_weight={pos_weight}, "
        f"neg_w 1.0→8.0, hard_neg low=0.001, layer_dim={layer_dim}, "
        f"epochs={N_EPOCHS}, batch={BATCH_SIZE}, lr={LR}\n\n"
        f"**Data:** {n_pos_windows} pos windows, {n_neg_windows} neg windows, "
        f"{n_holdout} holdout clips\n\n"
        f"**Verdict:** {verdict}\n\n"
        + md_header + "\n" + md_sep + "\n" + "\n".join(md_rows) + "\n"
    )

    # Create file with a header if it doesn't exist yet
    if not EXPERIMENTS_FILE.exists():
        EXPERIMENTS_FILE.write_text(
            "# Wake-word training experiments\n\n"
            "Each section below is one run of `wakeword/diagnostic_run.py`. "
            "Sections are appended in chronological order. "
            "The **Verdict** picks the epoch with the lowest holdout FP at "
            "the runtime threshold (>0.95) *among epochs where positive recall "
            "is intact* (pos_min ≥ 0.5).\n"
        )
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(section)


def main() -> None:
    args = parse_args()
    label = args.label or f"pw{args.pos_weight} ld{args.layer_dim}"

    # ---- Load + augment positives (mirrors train.py) ----
    pos_files = sorted(POSITIVE_DIR.glob("*.wav"))
    print(f"Loading {len(pos_files)} positives...")
    pos_audio_raw = []
    for p in pos_files:
        raw = load_wav(p)
        trimmed = trim_trailing_silence(raw)
        pos_audio_raw.append(normalize_peak(left_pad_or_trim(trimmed, CLIP_SAMPLES)))
    pos_audio_raw = np.stack(pos_audio_raw)

    print("Generating 30 synthetic RIRs...")
    aug_rng = np.random.default_rng(123)
    rirs = generate_synthetic_rirs(30, aug_rng)

    # ---- Load + split negatives ----
    neg_paths_all = sorted(NEGATIVE_DIR.glob("*.wav"))
    train_idx, holdout_idx = split_indices(
        len(neg_paths_all), holdout_frac=NEG_HOLDOUT_FRAC, seed=NEG_HOLDOUT_SEED,
    )

    # Re-mining: forcibly move hard-negative filenames into training, out of
    # holdout. Tests whether the model can suppress specific failure cases.
    forced_into_train: set[str] = set()
    if args.hard_negatives_file:
        forced_into_train = {
            line.strip() for line in Path(args.hard_negatives_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        train_set, holdout_set = set(train_idx.tolist()), set(holdout_idx.tolist())
        moved = 0
        for i, p in enumerate(neg_paths_all):
            if p.name in forced_into_train and i in holdout_set:
                holdout_set.discard(i)
                train_set.add(i)
                moved += 1
        train_idx = np.array(sorted(train_set))
        holdout_idx = np.array(sorted(holdout_set))
        print(f"Re-mining: forced {len(forced_into_train)} filenames into train; "
              f"{moved} were in holdout and got moved")

    neg_paths = [neg_paths_all[i] for i in train_idx]
    neg_holdout_paths = [neg_paths_all[i] for i in holdout_idx]
    print(f"Negatives: {len(neg_paths_all)} total -> "
          f"{len(neg_paths)} train / {len(neg_holdout_paths)} holdout")

    print(f"Loading {len(neg_paths)} train negatives...")
    neg_audio_list = []
    for p in neg_paths:
        neg_audio_list.append(pad_or_trim(load_wav(p), CLIP_SAMPLES))
    neg_audio = np.stack(neg_audio_list)

    # ---- Augment positives (same as train.py) ----
    pos_audio = []
    for clip in pos_audio_raw:
        pos_audio.append(clip)
        for _ in range(N_AUG_VARIANTS - 1):
            x = clip
            if aug_rng.random() < 0.7:
                rir = rirs[aug_rng.integers(0, len(rirs))]
                x = apply_rir(x, rir)
            if aug_rng.random() < 0.85 and len(neg_audio) > 0:
                bg = neg_audio[aug_rng.integers(0, len(neg_audio))]
                x = mix_with_background(
                    x, bg, snr_db=random_snr_db(aug_rng, low=0.0, high=10.0),
                    rng=aug_rng,
                )
            pos_audio.append(x)
    pos_audio = np.stack(pos_audio)
    print(f"Augmented positives: {len(pos_audio)}")

    # ---- Embed train data ----
    print(f"Embedding {len(pos_audio)} positives + {len(neg_audio)} negatives...")
    pos_emb = np.stack([embed_clip(c) for c in pos_audio])
    neg_emb = np.stack([embed_clip(c) for c in neg_audio])

    win_rng = np.random.default_rng(7)
    pos_wins = np.stack([
        build_positive_window(e, jitter_ms=200.0, rng=win_rng) for e in pos_emb
    ]).astype(np.float32)
    neg_wins = build_negative_windows(neg_emb, stride=NEG_STRIDE)
    print(f"Windows — positive: {len(pos_wins)}  negative: {len(neg_wins)}")

    X = np.vstack([pos_wins, neg_wins]).astype(np.float32)
    y = np.array([1.0] * len(pos_wins) + [0.0] * len(neg_wins), dtype=np.float32)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]

    # ---- Pre-embed holdout (one (N, 16, 96) array per file) ----
    print(f"Pre-embedding {len(neg_holdout_paths)} holdout clips (one-time)...")
    holdout_wins: list[np.ndarray] = []
    for p in neg_holdout_paths:
        audio = pad_or_trim(load_wav(p), CLIP_SAMPLES)
        emb = np.stack([embed_clip(audio)])
        wins = build_negative_windows(emb, stride=1)
        holdout_wins.append(wins.astype(np.float32))

    # ---- Train with snapshot callback ----
    snapshots: dict[int, dict] = {}

    def snapshot(epoch: int, model: WakewordModel) -> None:
        if epoch % CHECKPOINT_EVERY == 0:
            snapshots[epoch] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            print(f"    [snapshot @ epoch {epoch}]")

    print(f"\nTraining {N_EPOCHS} epochs ({label}), snapshotting every {CHECKPOINT_EVERY}...")
    _ = train_model(
        X, y, epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        layer_dim=args.layer_dim, on_epoch_end=snapshot, pos_weight=args.pos_weight,
    )

    # ---- Score each snapshot ----
    pos_idx = np.where(y == 1.0)[0]
    pos_X = torch.from_numpy(X[pos_idx])

    print("\n" + "=" * 100)
    print(f"Diagnostic curve [{label}]: holdout FP per checkpoint")
    print(f"  hyperparams: epochs={N_EPOCHS}, neg_w 1.0→8.0, hard_neg low=0.001, "
          f"layer_dim={args.layer_dim}, pos_weight={args.pos_weight}")
    print(f"  data:        {len(pos_wins)} pos windows, {len(neg_wins)} neg windows "
          f"(1:{len(neg_wins)/len(pos_wins):.1f}), {len(neg_holdout_paths)} holdout clips")
    print("=" * 100)
    header = (
        f"{'epoch':>5}  "
        f"{'pos_min':>8}  {'pos_mean':>8}  " +
        "  ".join(f">{t:.2f}".rjust(8) for t in HOLDOUT_THRESHOLDS) +
        "    top-firer"
    )
    print(header)
    print("-" * len(header))

    curve_rows: list[dict] = []
    for epoch in sorted(snapshots):
        m = WakewordModel(layer_dim=args.layer_dim, inference=True)
        m.load_state_dict(snapshots[epoch])
        m.eval()
        with torch.no_grad():
            pos_scores = m(pos_X).squeeze().numpy()

            holdout_peaks = []
            for wins in holdout_wins:
                if len(wins) == 0:
                    continue
                s = m(torch.from_numpy(wins)).squeeze().numpy()
                holdout_peaks.append(float(np.atleast_1d(s).max()))

        peaks = np.array(holdout_peaks)
        total = len(peaks)
        fp_counts = [int((peaks > t).sum()) for t in HOLDOUT_THRESHOLDS]
        max_peak = float(peaks.max()) if total else 0.0

        cells = "  ".join(f"{c:4d}/{total}".rjust(8) for c in fp_counts)
        print(
            f"{epoch:5d}  "
            f"{float(pos_scores.min()):8.3f}  {float(pos_scores.mean()):8.3f}  "
            f"max_pos={float(pos_scores.max()):.3f}  "
            f"{cells}    top_firer={max_peak:.3f}"
        )

        curve_rows.append({
            "epoch": epoch,
            "pos_min": float(pos_scores.min()),
            "pos_mean": float(pos_scores.mean()),
            "pos_max": float(pos_scores.max()),
            "fp_counts": fp_counts,
            "top_firer": max_peak,
        })

    print("=" * 100)

    append_experiment_row(
        label=label,
        pos_weight=args.pos_weight,
        layer_dim=args.layer_dim,
        n_pos_windows=len(pos_wins),
        n_neg_windows=len(neg_wins),
        n_holdout=len(neg_holdout_paths),
        curve_rows=curve_rows,
    )
    print(f"\nAppended results to {EXPERIMENTS_FILE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
