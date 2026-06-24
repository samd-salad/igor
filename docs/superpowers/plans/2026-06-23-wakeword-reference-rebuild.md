# Wake-Word Training: Reference-Aligned Rebuild (Path B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace our current divergent OWW training pipeline with one aligned to dscripka's reference implementation, eliminating the noise-floor phantom-fire failure mode and getting Igor's wake-word detector to community best practice.

**Architecture:** Refactor `wakeword/train.py` into composable helpers (`_audio.py`, `_features.py`, `_augmentation.py`, `_dataset.py`, `_training.py`) so each piece is unit-testable. Add a Piper-based synthetic-positive generator (`synthesize_positives.py`) and an offline silence smoke test (`test_silence.py`). Wire RIR + SNR-mix augmentation via `pyroomacoustics` (no external RIR download). Replace synthetic Gaussian-noise negatives with real Pi-recorded mic-floor silence + content-bearing negatives.

**Tech Stack:** Python 3.12, `pyopen_wakeword` (feature extraction, matches Pi runtime), `piper-tts` (synthetic positives), `pyroomacoustics` (RIR generation), `torch` 2.x (training), `onnx2tf` (TFLite export). Existing `.venv-wake` venv extended with the new deps.

## Global Constraints

- pyopen_wakeword emits embeddings at **~12 Hz** (EMB_STEP=8 × 80ms frames). This is the truth; current code claims ~43 Hz and is wrong. All frame-count math must use 12 Hz.
- 16-frame window = **~1.3 seconds** (not ~370ms).
- 3-second clip → **~37 frames**, not ~129.
- Use the `.venv-wake` Python 3.12 venv for all training-related code; main `.venv` (3.13) doesn't have TensorFlow.
- Never delete user-recorded `samples/positive/` or `samples/negative/` clips — they are the user's labour.
- Output TFLite filename: `igor_v0.3.tflite` (Pi currently has v0.2 which phantom-fires; v0.3 is the path-B retrain).
- Wyoming-openwakeword runtime expects fixed batch=1 TFLite; export must keep `-kat input` (channel-last preserved).
- All synthetic positives go to `samples/positive_synth/`; train.py loads both `positive/` and `positive_synth/`.
- All synthetic backgrounds go to `samples/backgrounds/`; reserved for SNR mixing.
- No new top-level dependencies in `requirements-server-text.txt` — training-only deps live in a new `requirements-train.txt`.

---

## File Structure

**New files (all Python 3.12 / training-time only):**
- `wakeword/_audio.py` — `load_wav`, `normalize_peak`, `trim_trailing_silence`, `pad_or_trim`, `left_pad_or_trim`
- `wakeword/_features.py` — `embed_clip(audio_int16) -> (T, 96)`, `EMB_RATE_HZ=12.13`, `frames_per_seconds(seconds)`
- `wakeword/_augmentation.py` — `mix_with_background(pos, bg, snr_db)`, `apply_rir(audio, rir)`, `random_snr_db(rng)`, `generate_synthetic_rirs(n, rng)`
- `wakeword/_dataset.py` — `build_positive_windows(clips, jitter_ms)`, `build_negative_windows(clips)`, `WINDOW_FRAMES=16`
- `wakeword/_training.py` — `WakewordModel` class, `train_model(X, y, epochs)`, `hard_negative_filter(preds, labels, low, high)`
- `wakeword/synthesize_positives.py` — Piper TTS positive generator, writes to `samples/positive_synth/`
- `wakeword/test_silence.py` — offline silence smoke test, fails if model triple-fires on silence/noise/hum
- `requirements-train.txt` — pinned wake-word training deps

**Modified files:**
- `wakeword/train.py` — slimmed to orchestrator that calls into helpers
- `wakeword/contracts.py:7` — fix `FEATURE_RATE_HZ` constant from `42.7` to `12.13`
- `CLAUDE.md` — wake-word section updated for ~12 Hz feature rate
- `deploy/wyoming-openwakeword.service` — add `--vad-threshold 0.5`

**Test files:**
- `tests/wakeword/test_features.py`
- `tests/wakeword/test_audio.py`
- `tests/wakeword/test_augmentation.py`
- `tests/wakeword/test_dataset.py`
- `tests/wakeword/test_training.py`

---

### Task 1: Fix frame-rate constant + documentation

**Files:**
- Modify: `wakeword/contracts.py`
- Modify: `CLAUDE.md` (the wake-word section)
- Modify: `wakeword/train.py:289` (comment claiming "~43 features/sec")
- Test: `tests/wakeword/test_features.py` (new)

**Interfaces:**
- Produces: `wakeword.contracts.FEATURE_RATE_HZ = 12.13`. This value is consumed by Task 2 (window labeling) and Task 11 (silence smoke test).

- [ ] **Step 1: Confirm actual rate empirically**

Run from repo root:

```bash
.venv-wake/Scripts/python.exe -c "
import numpy as np
from pyopen_wakeword import OpenWakeWordFeatures
feats = OpenWakeWordFeatures.from_builtin()
feats.reset()
silence = np.zeros(16000 * 3, dtype=np.int16).tobytes()
out = []
for i in range(0, len(silence) - 320 + 1, 320):
    for f in feats.process_streaming(silence[i:i+320]):
        out.append(f.squeeze())
print(f'frames for 3s audio: {len(out)}')
print(f'rate Hz: {len(out)/3:.2f}')
"
```

Expected: `frames for 3s audio: 36` and `rate Hz: 12.13` (or close).

- [ ] **Step 2: Write the failing test**

Create `tests/wakeword/test_features.py`:

```python
import numpy as np

from wakeword.contracts import FEATURE_RATE_HZ
from wakeword._features import embed_clip, frames_per_seconds


def test_feature_rate_constant_matches_pyopen_wakeword():
    # pyopen_wakeword EMB_STEP=8 mel frames, MELS_PER_SECOND=97
    # So embeddings come at 97/8 = 12.125 Hz
    assert 12.0 <= FEATURE_RATE_HZ <= 12.2


def test_embed_clip_produces_expected_frame_count_for_3_seconds():
    silence_3s = np.zeros(16000 * 3, dtype=np.int16)
    emb = embed_clip(silence_3s)
    # 3 seconds × 12.13 Hz ≈ 36-37 frames
    assert 34 <= emb.shape[0] <= 38
    assert emb.shape[1] == 96


def test_frames_per_seconds_helper():
    assert 34 <= frames_per_seconds(3.0) <= 38
    assert 11 <= frames_per_seconds(1.0) <= 13
```

- [ ] **Step 3: Run test — confirm fails**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'wakeword._features'`.

- [ ] **Step 4: Create `wakeword/_features.py`**

```python
"""Feature extraction wrapping pyopen_wakeword. Single source of truth for
embedding rate and shape."""
from __future__ import annotations
import numpy as np
from pyopen_wakeword import OpenWakeWordFeatures

# pyopen_wakeword constants: MELS_PER_SECOND=97, EMB_STEP=8 mel frames per
# emitted embedding => 97/8 = 12.125 Hz. NOT 43 Hz as previously claimed.
EMB_RATE_HZ = 12.125
FEATURE_DIM = 96
CHUNK_BYTES = 320  # 10ms @ 16kHz, 16-bit mono


def frames_per_seconds(seconds: float) -> int:
    return int(round(seconds * EMB_RATE_HZ))


def embed_clip(audio_int16: np.ndarray) -> np.ndarray:
    """Stream a single int16 clip through pyopen_wakeword features.
    Returns (T, 96) where T ≈ duration_seconds × 12.13."""
    feats = OpenWakeWordFeatures.from_builtin()
    feats.reset()
    raw = audio_int16.astype(np.int16).tobytes()
    out = []
    for i in range(0, len(raw) - CHUNK_BYTES + 1, CHUNK_BYTES):
        for f in feats.process_streaming(raw[i:i+CHUNK_BYTES]):
            out.append(f.squeeze())
    if not out:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(out).astype(np.float32)
```

- [ ] **Step 5: Update `wakeword/contracts.py`**

Find and replace:

```python
FEATURE_RATE_HZ = 42.7
```

with:

```python
FEATURE_RATE_HZ = 12.125
```

- [ ] **Step 6: Run test — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_features.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Update CLAUDE.md wake-word notes**

Find any reference to `~43 Hz`, `~370ms`, or `MODEL_INPUT_SHAPE` math in CLAUDE.md and reword. Specifically the line claiming "feature rate" should say `~12 Hz`.

- [ ] **Step 8: Commit**

```bash
git add wakeword/_features.py wakeword/contracts.py CLAUDE.md tests/wakeword/test_features.py
git commit -m "wakeword: correct embedding rate to ~12 Hz (was wrongly claimed ~43 Hz)

pyopen_wakeword emits at EMB_RATE_HZ = 97/8 = 12.125 Hz, not the ~43 Hz our
contracts and training script assumed. This was a silent ~3.6x error that
caused 'last N windows' positive labeling to span the entire clip instead
of just the wake-word tail — directly producing the phantom-fire-on-silence
symptom we've been chasing."
```

---

### Task 2: Reference-aligned positive window labeling

**Files:**
- Create: `wakeword/_audio.py`
- Create: `wakeword/_dataset.py`
- Create: `tests/wakeword/test_audio.py`
- Create: `tests/wakeword/test_dataset.py`
- Modify: `wakeword/train.py` (will swap in helpers later — leave for now)

**Interfaces:**
- Consumes: `embed_clip`, `EMB_RATE_HZ`, `frames_per_seconds` from Task 1.
- Produces:
  - `_audio.load_wav(path) -> np.ndarray (int16, mono, 16kHz)`
  - `_audio.normalize_peak(audio, target=16000) -> np.ndarray`
  - `_audio.trim_trailing_silence(audio, threshold=250, keep_tail_samples=1600)`
  - `_audio.left_pad_or_trim(audio, length) -> np.ndarray`
  - `_dataset.build_positive_window(clip_emb, jitter_ms, rng) -> np.ndarray (16, 96)`
  - `_dataset.WINDOW_FRAMES = 16`
- Replaces: the `POSITIVE_WINDOWS_PER_CLIP = 25` loop in `train.py`. Going forward each positive clip produces exactly ONE labeled window.

- [ ] **Step 1: Write the failing test for `_audio`**

Create `tests/wakeword/test_audio.py`:

```python
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
```

- [ ] **Step 2: Create `wakeword/_audio.py`** — port helpers from existing `train.py`:

```python
"""Audio preprocessing helpers — load_wav, normalize_peak, trim_trailing_silence,
left_pad_or_trim. Used by the training data pipeline."""
from __future__ import annotations
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
    """Load a 16kHz mono WAV as int16. Raises on mismatched sr/channels."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
        if ch != 1:
            raise ValueError(f"Expected mono, got {ch} channels")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).copy()


def normalize_peak(audio: np.ndarray, target_peak: int = 16000) -> np.ndarray:
    """Scale audio so its peak amplitude equals target_peak.
    Digital silence (peak < 10) is left untouched."""
    peak = int(np.max(np.abs(audio)))
    if peak < 10:
        return audio
    gain = min(target_peak / peak, 1000.0)
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def trim_trailing_silence(audio: np.ndarray, threshold: int = 250,
                          keep_tail_samples: int = 1600) -> np.ndarray:
    """Strip samples after the last energetic sample, keeping a short tail
    so the final syllable isn't clipped."""
    abs_audio = np.abs(audio)
    nonzero = np.where(abs_audio > threshold)[0]
    if len(nonzero) == 0:
        return audio
    last_real = nonzero[-1]
    end = min(len(audio), last_real + keep_tail_samples)
    return audio[:end]


def left_pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Left-pad with silence so content lives at the END of the clip."""
    if len(audio) >= length:
        return audio[-length:]
    return np.pad(audio, (length - len(audio), 0)).astype(audio.dtype)
```

- [ ] **Step 3: Run audio tests — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_audio.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Write failing tests for `_dataset`**

Create `tests/wakeword/test_dataset.py`:

```python
import numpy as np
import pytest

from wakeword._dataset import (
    build_positive_window, WINDOW_FRAMES,
)


def test_window_frames_constant_is_16():
    assert WINDOW_FRAMES == 16


def test_positive_window_shape_is_16_by_96():
    # 37 frames of fake embeddings (3-second clip rate-12 Hz)
    clip_emb = np.arange(37 * 96, dtype=np.float32).reshape(37, 96)
    rng = np.random.default_rng(0)
    win = build_positive_window(clip_emb, jitter_ms=0, rng=rng)
    assert win.shape == (16, 96)


def test_positive_window_aligns_to_end_with_zero_jitter():
    clip_emb = np.arange(37 * 96, dtype=np.float32).reshape(37, 96)
    rng = np.random.default_rng(0)
    win = build_positive_window(clip_emb, jitter_ms=0, rng=rng)
    # With jitter=0, the window should be the LAST 16 frames
    assert np.array_equal(win, clip_emb[-16:])


def test_positive_window_with_jitter_stays_in_bounds():
    clip_emb = np.arange(37 * 96, dtype=np.float32).reshape(37, 96)
    rng = np.random.default_rng(0)
    for _ in range(50):
        win = build_positive_window(clip_emb, jitter_ms=200, rng=rng)
        assert win.shape == (16, 96)


def test_positive_window_jitter_actually_shifts_window():
    clip_emb = np.arange(37 * 96, dtype=np.float32).reshape(37, 96)
    rng = np.random.default_rng(42)
    starts_seen = set()
    for _ in range(50):
        win = build_positive_window(clip_emb, jitter_ms=200, rng=rng)
        # Identify start frame from first element
        starts_seen.add(int(win[0, 0]) // 96)
    # Should have observed at least 2 different starts due to jitter
    assert len(starts_seen) >= 2


def test_positive_window_short_clip_raises():
    short = np.zeros((10, 96), dtype=np.float32)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too short"):
        build_positive_window(short, jitter_ms=0, rng=rng)
```

- [ ] **Step 5: Create `wakeword/_dataset.py`**

```python
"""Window construction for OWW classifier training. Reference-aligned:
ONE positive window per clip, anchored to the END of the clip with optional
±jitter_ms timing variation. Multi-window labeling per clip (as the old
code did) labels leading silence as positive and causes runtime
phantom-fires."""
from __future__ import annotations
import numpy as np

from wakeword._features import EMB_RATE_HZ

WINDOW_FRAMES = 16


def build_positive_window(clip_emb: np.ndarray, jitter_ms: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Take one (16, 96) window from clip_emb whose end falls within the
    last `jitter_ms` ms of the clip. Mirrors dscripka's reference notebook:
    'aligning the positive clips with background data such that the end of
    the input window aligns with the end of the positive clip'."""
    if clip_emb.shape[0] < WINDOW_FRAMES:
        raise ValueError(
            f"clip too short for a {WINDOW_FRAMES}-frame window "
            f"(got {clip_emb.shape[0]} frames)"
        )
    n_frames = clip_emb.shape[0]
    jitter_frames = max(0, int(round(jitter_ms / 1000.0 * EMB_RATE_HZ)))
    # End anchored at last frame; jitter shifts END backwards by 0..jitter_frames
    end_offset = int(rng.integers(0, jitter_frames + 1)) if jitter_frames else 0
    end_idx = n_frames - end_offset
    start_idx = end_idx - WINDOW_FRAMES
    if start_idx < 0:
        start_idx, end_idx = 0, WINDOW_FRAMES
    return clip_emb[start_idx:end_idx].astype(np.float32)
```

- [ ] **Step 6: Run dataset tests — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_dataset.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add wakeword/_audio.py wakeword/_dataset.py tests/wakeword/test_audio.py tests/wakeword/test_dataset.py
git commit -m "wakeword: reference-aligned positive window labeling — one window per clip

dscripka's official notebook labels exactly one positive window per clip,
end-anchored to the trim-aligned wake-word end with ±200ms jitter. Our
prior code labeled the LAST 25 windows per clip; at the correct 12 Hz
frame rate (Task 1) those 25 windows span ~2 seconds — almost the entire
clip — and effectively labeled leading silence as positive.

build_positive_window now produces a single (16, 96) window per clip with
configurable jitter. Old POSITIVE_WINDOWS_PER_CLIP loop in train.py will
be replaced by Task 4."
```

---

### Task 3: Remove synthetic-noise/silence negative class; require real negatives

**Files:**
- Modify: `wakeword/train.py:144-173` (delete `generate_synthetic_negatives`)
- Modify: `wakeword/train.py:235-256` (delete synthetic-neg loading branch)
- Modify: `wakeword/_dataset.py` (add `build_negative_windows` helper)
- Test: extend `tests/wakeword/test_dataset.py`

**Interfaces:**
- Produces: `_dataset.build_negative_windows(clips_emb, stride=1) -> np.ndarray (N, 16, 96)`. Takes (M, T, 96) embeddings, slides 16-frame windows with given stride, returns stacked windows.
- Removes: `generate_synthetic_negatives`. Reference workflow doesn't use synthetic Gaussian/sine clips.

- [ ] **Step 1: Write the failing test**

Append to `tests/wakeword/test_dataset.py`:

```python
from wakeword._dataset import build_negative_windows


def test_build_negative_windows_slides_over_clips():
    # 2 clips, each 20 frames of (96-dim) embeddings
    clips_emb = np.zeros((2, 20, 96), dtype=np.float32)
    clips_emb[0, :, 0] = np.arange(20)   # marker
    clips_emb[1, :, 0] = np.arange(100, 120)
    wins = build_negative_windows(clips_emb, stride=1)
    # 2 clips × (20-16+1) = 10 windows
    assert wins.shape == (10, 16, 96)


def test_build_negative_windows_respects_stride():
    clips_emb = np.zeros((1, 20, 96), dtype=np.float32)
    wins = build_negative_windows(clips_emb, stride=2)
    # (20-16)//2 + 1 = 3 windows
    assert wins.shape == (3, 16, 96)


def test_build_negative_windows_skips_too_short_clips():
    clips_emb = np.zeros((2, 10, 96), dtype=np.float32)
    wins = build_negative_windows(clips_emb, stride=1)
    assert wins.shape == (0, 16, 96)
```

- [ ] **Step 2: Append `build_negative_windows` to `wakeword/_dataset.py`**

```python
def build_negative_windows(clips_emb: np.ndarray, stride: int = 1) -> np.ndarray:
    """Slide 16-frame windows over each (T, 96) clip in clips_emb.
    Returns (N, 16, 96). Clips shorter than 16 frames are skipped.
    stride > 1 reduces window count proportionally (use to keep memory in check
    when the real-negative pool is large)."""
    out = []
    for clip in clips_emb:
        if clip.shape[0] < WINDOW_FRAMES:
            continue
        max_start = clip.shape[0] - WINDOW_FRAMES
        for start in range(0, max_start + 1, stride):
            out.append(clip[start:start + WINDOW_FRAMES])
    if not out:
        return np.empty((0, WINDOW_FRAMES, 96), dtype=np.float32)
    return np.stack(out).astype(np.float32)
```

- [ ] **Step 3: Run tests — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_dataset.py -v
```

Expected: 9 passed (6 original + 3 new).

- [ ] **Step 4: Delete `generate_synthetic_negatives` from `train.py`**

Open `wakeword/train.py`, find the function `generate_synthetic_negatives` (around line 144) and the block in `main()` that calls it (around line 235-256 — the section starting `# Synthetic negatives` and ending where `neg_audio` is built). Replace the whole block with:

```python
    # Real negatives only. Synthetic Gaussian-noise / sine-tone negatives
    # are NOT used — they don't represent the actual mic-noise-floor
    # distribution and training on them taught the model to phantom-fire
    # on quiet rooms. Reference workflow (dscripka) uses RIR + SNR-mixed
    # backgrounds instead (see Tasks 6, 7).
    if NEGATIVE_DIR.exists():
        neg_files = sorted(NEGATIVE_DIR.glob("*.wav"))
    else:
        neg_files = []
    if len(neg_files) < 100:
        print(f"ERROR: need at least 100 real negative clips in {NEGATIVE_DIR}/,"
              f" found {len(neg_files)}.")
        print("Record more on the Pi: python3 wakeword/record_negatives.py")
        sys.exit(1)
    print(f"Loading {len(neg_files)} real negative clips...")
    neg_audio_list = []
    for path in neg_files:
        try:
            audio = normalize_peak(pad_or_trim(load_wav(path), CLIP_SAMPLES))
            neg_audio_list.append(audio)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    neg_audio = np.vstack(neg_audio_list) if neg_audio_list else np.empty(
        (0, CLIP_SAMPLES), dtype=np.float32
    )
```

Also delete the `generate_synthetic_negatives` function definition entirely.

- [ ] **Step 5: Run wakeword tests — verify nothing regressed**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/ -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add wakeword/train.py wakeword/_dataset.py tests/wakeword/test_dataset.py
git commit -m "wakeword: remove synthetic noise/silence negative class

Synthetic Gaussian-noise + near-silence + sine-tone clips were intended to
teach 'quiet audio = negative' but they live in a feature subspace that
doesn't overlap with real mic-floor noise. Trained models still triple-
fire on σ=30 Gaussian (mic floor) and 60 Hz hum.

Reference workflow (dscripka) handles noise robustness differently: SNR-
mixed real backgrounds + RIR convolution. Those land in Tasks 6 and 7.

For now, require >= 100 real negative clips in samples/negative/ and fail
loudly otherwise. build_negative_windows in _dataset.py replaces the
inline window-building logic from train.py."
```

---

### Task 4: Augmentation helpers (SNR mix + RIR convolution)

**Files:**
- Create: `wakeword/_augmentation.py`
- Create: `tests/wakeword/test_augmentation.py`
- Modify: `requirements-train.txt` (new file — list training-time deps)

**Interfaces:**
- Consumes: nothing from prior tasks (pure functions).
- Produces:
  - `_augmentation.mix_with_background(positive, background, snr_db, rng) -> int16 audio`
  - `_augmentation.apply_rir(audio, rir) -> int16 audio (same length)`
  - `_augmentation.generate_synthetic_rirs(n: int, rng) -> list[np.ndarray]`
  - `_augmentation.random_snr_db(rng, low=0.0, high=15.0) -> float`

- [ ] **Step 1: Add `requirements-train.txt`**

Create `requirements-train.txt`:

```
# Training-only deps for wakeword/train.py. Install in .venv-wake (Python 3.12).
# These are NOT installed by the runtime container.
torch>=2.0
tensorflow>=2.16
tf-keras
onnx>=1.15
onnxruntime
onnx2tf
ai_edge_litert
psutil
sng4onnx
onnx_graphsurgeon
onnxsim
openwakeword
pyopen_wakeword>=1.1
piper-tts>=1.2
pyroomacoustics>=0.7
scipy
```

- [ ] **Step 2: Install pyroomacoustics + scipy**

```bash
.venv-wake/Scripts/python.exe -m pip install pyroomacoustics scipy
```

Expected: install succeeds; pyroomacoustics 0.7+, scipy 1.x. Smoke test:

```bash
.venv-wake/Scripts/python.exe -c "import pyroomacoustics as pra; from scipy.signal import fftconvolve; print('ok')"
```

- [ ] **Step 3: Write the failing tests**

Create `tests/wakeword/test_augmentation.py`:

```python
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
```

- [ ] **Step 4: Run tests — confirm fails**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_augmentation.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Create `wakeword/_augmentation.py`**

```python
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
```

- [ ] **Step 6: Run tests — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_augmentation.py -v
```

Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add wakeword/_augmentation.py tests/wakeword/test_augmentation.py requirements-train.txt
git commit -m "wakeword: SNR-mix + RIR augmentation helpers

mix_with_background scales a background clip so the positive is at +snr_db
relative to it, then sums. random_snr_db samples uniformly in [0, 15] dB
matching the dscripka notebook range.

apply_rir convolves audio with a room impulse response and re-normalizes.

generate_synthetic_rirs builds n RIRs by simulating random shoebox rooms
in pyroomacoustics — no external download needed. Rooms vary in dimensions
(3-8m × 3-6m × 2.4-3.5m), absorption (0.2-0.6), and reflection order.

Wired into the training data pipeline in Task 6."
```

---

### Task 5: Piper TTS synthetic positives

**Files:**
- Create: `wakeword/synthesize_positives.py`
- No tests — Piper is an external CLI tool; output is verified by listening + by feeding into the training set.

**Interfaces:**
- Produces: `samples/positive_synth/synth_<voice>_<idx>.wav` files. Each is 16kHz mono, ~1.5s, contains "Igor" spoken by a Piper voice.
- Consumed by: Task 6 (training data loader).

- [ ] **Step 1: Install Piper voices**

```bash
.venv-wake/Scripts/python.exe -m pip install piper-tts
```

Then download 6-10 voices (mix of male / female / accents). Piper hosts voices on Hugging Face. Use the included `piper` CLI:

```bash
mkdir -p wakeword/piper_voices
.venv-wake/Scripts/python.exe -m piper.download_voices \
  en_US-lessac-medium en_US-amy-medium en_US-ryan-medium \
  en_US-libritts_r-medium en_GB-alan-medium en_GB-jenny_dioco-medium \
  --download-dir wakeword/piper_voices
ls wakeword/piper_voices/
```

Expected: each voice produces a `.onnx` + `.onnx.json` pair.

- [ ] **Step 2: Write `wakeword/synthesize_positives.py`**

```python
"""Generate synthetic positive wake-word clips using Piper TTS.

For each installed Piper voice we render 'Igor' in many variations
(with surrounding silence, different lengths/punctuation, varied
length_scale and noise_scale). Each output clip is 16kHz mono int16
and lives in samples/positive_synth/. Loaded alongside real positives
by wakeword/train.py."""
from __future__ import annotations
import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import piper

ROOT = Path(__file__).parent
VOICES_DIR = ROOT / "piper_voices"
OUT_DIR = ROOT / "samples" / "positive_synth"
SAMPLE_RATE = 16000

# Surface forms: different prosody-inducing punctuation around the word.
# Piper's prosody changes with terminal punctuation, so this provides
# variety without us needing to manually tune length_scale per output.
PHRASES = [
    "Igor.",
    "Igor!",
    "Igor?",
    "Igor,",
    "Igor",
    "Hey Igor.",
    "Igor?",
    "Igor!",
]


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.astype(np.int16).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-voice", type=int, default=100,
                        help="Synth clips per voice (default 100)")
    args = parser.parse_args()

    voice_files = sorted(VOICES_DIR.glob("*.onnx"))
    if not voice_files:
        print(f"No Piper voices in {VOICES_DIR}/. Download some first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.wav")))
    print(f"  Existing synthetic positives: {existing}")
    print(f"  Voices: {[v.stem for v in voice_files]}")

    rng = np.random.default_rng(0)
    idx = existing
    for voice_path in voice_files:
        voice = piper.PiperVoice.load(voice_path)
        per_voice = args.per_voice
        for i in range(per_voice):
            phrase = PHRASES[i % len(PHRASES)]
            length_scale = float(rng.uniform(0.85, 1.20))
            noise_scale = float(rng.uniform(0.4, 0.8))
            noise_w_scale = float(rng.uniform(0.4, 0.9))
            audio_chunks = list(voice.synthesize(
                phrase,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
            ))
            # Concatenate, resample to 16kHz if needed
            raw = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
            audio = np.frombuffer(raw, dtype=np.int16)
            voice_sr = audio_chunks[0].sample_rate if audio_chunks else SAMPLE_RATE
            if voice_sr != SAMPLE_RATE:
                # Simple linear resample
                ratio = SAMPLE_RATE / voice_sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_len, endpoint=False),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.int16)
            out_path = OUT_DIR / f"synth_{voice_path.stem}_{idx:05d}.wav"
            _write_wav(out_path, audio)
            idx += 1
            if idx % 50 == 0:
                print(f"  [{idx - existing}] {out_path.name}", flush=True)
    print(f"\nDone. {idx - existing} synthetic positives written to {OUT_DIR}/.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test — generate 5 per voice first**

```bash
.venv-wake/Scripts/python.exe wakeword/synthesize_positives.py --per-voice 5
ls wakeword/samples/positive_synth/ | wc -l
```

Expected: ~30-60 WAVs (depending on how many voices downloaded). Spot-check 2-3 clips by playing them on your system to confirm Piper produced intelligible "Igor."

- [ ] **Step 4: Full generation**

```bash
.venv-wake/Scripts/python.exe wakeword/synthesize_positives.py --per-voice 200
```

Expected: ~1200-2000 synthetic positives.

- [ ] **Step 5: Commit (Piper voices excluded from git via .gitignore)**

First, ensure `.gitignore` excludes:

```
wakeword/piper_voices/
wakeword/samples/positive_synth/
```

(Add the lines if not already there.)

Then:

```bash
git add wakeword/synthesize_positives.py .gitignore
git commit -m "wakeword: Piper TTS synthetic positive generator

Per voice, renders 'Igor' / 'Igor!' / 'Igor?' / 'Hey Igor.' with randomized
length_scale and noise_scale. With 6 voices × 200 clips/voice we get ~1200
extra positives, raising effective positive count from 231 to ~1430+.

This addresses the 'too few real positives' issue dscripka's README flags:
custom OWW models work best with thousands of varied positives.

Voices and output WAVs are gitignored — regenerate as needed."
```

---

### Task 6: Wire augmentation + Piper positives into the training pipeline

**Files:**
- Modify: `wakeword/train.py` — extensive rewrite of positive-loading section
- Test: extend `tests/wakeword/test_dataset.py`

**Interfaces:**
- Consumes: helpers from Tasks 1-5.
- Produces: training data tensors where each positive clip has been (a) optionally RIR-convolved (50% chance), (b) optionally background-mixed at random SNR (75% chance), then (c) windowed via `build_positive_window`.

- [ ] **Step 1: Replace the positive-loading section in `train.py`**

Find the block starting with the comment `# Load positive clips` (around line 212) and ending at the `pos_audio = np.stack(pos_audio)` line. Replace with:

```python
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
```

- [ ] **Step 2: Insert RIR + SNR-mix augmentation block after positives load and before embedding**

Right after the `print(f"  Loaded ... positive clips.")` line, add:

```python
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
```

- [ ] **Step 3: Add necessary imports at the top of `train.py`**

Find the existing import block and add:

```python
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
```

Also remove the now-duplicate definitions (`load_wav`, `normalize_peak`, `trim_trailing_silence`, `left_pad_or_trim`, `pad_or_trim` may still be used for negatives — leave it) from the body of `train.py`.

- [ ] **Step 4: Replace the window-extraction section**

Find the section starting with `# Slice into 16-frame windows` (around line 291-327). Replace from `WINDOW = 16` through the `print(f"  Windows — positive: ... negative: ...")` line with:

```python
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
```

Also remove the now-dead helpers `make_windows` and `POSITIVE_WINDOWS_PER_CLIP` in `train.py`.

- [ ] **Step 5: Run the full training pipeline end-to-end at small scale**

```bash
.venv-wake/Scripts/python.exe wakeword/train.py > wakeword/_train.log 2>&1
echo "EXIT $?"
tail -40 wakeword/_train.log
```

Expected: training completes; positive count printed reflects real + synthetic; window count for positives ≈ N positives (NOT 25 × N); training scores printed.

- [ ] **Step 6: Commit**

```bash
git add wakeword/train.py tests/wakeword/test_dataset.py
git commit -m "wakeword/train: wire augmentation pipeline + reference window labeling

Positive flow: load (real + synthetic) -> trim_trailing_silence ->
normalize_peak -> left_pad_or_trim -> 50% RIR convolve -> 75% SNR-mix at
0-15 dB -> embed_clip -> build_positive_window (ONE window, ±200ms jitter).

Negative flow: load (real) -> pad_or_trim -> normalize_peak -> embed_clip
-> build_negative_windows (stride=1, all windows).

Window count drops from 25*N to 1*N for positives, matching dscripka's
reference notebook. Augmentation adds the canonical OWW noise-robustness
levers (RIR + SNR-mix) that we were previously missing."
```

---

### Task 7: Reference-aligned model + training loop (hard-negative mining, neg-weight schedule)

**Files:**
- Create: `wakeword/_training.py`
- Modify: `wakeword/train.py` — replace inline training loop with call into `_training.train_model`
- Create: `tests/wakeword/test_training.py`

**Interfaces:**
- Consumes: nothing from prior tasks (pure PyTorch).
- Produces:
  - `_training.WakewordModel(layer_dim=128) -> nn.Module` (inference variant returns sigmoid)
  - `_training.train_model(X, y, epochs, lr, batch_size, pos_weight) -> trained_inference_model`
  - `_training.hard_negative_filter(preds_logits, labels, low=0.001, high=0.999) -> mask: torch.Tensor`
  - `_training.NEGATIVE_WEIGHT_SCHEDULE: list[(epoch_frac, weight)]`

- [ ] **Step 1: Write the failing tests**

Create `tests/wakeword/test_training.py`:

```python
import numpy as np
import torch

from wakeword._training import (
    WakewordModel, hard_negative_filter, build_neg_weight_schedule,
)


def test_wakeword_model_takes_16x96_returns_scalar():
    model = WakewordModel(layer_dim=128)
    x = torch.zeros(4, 16, 96)
    out = model(x)
    assert out.shape == (4, 1)


def test_wakeword_model_inference_outputs_sigmoid_in_0_1():
    model = WakewordModel(layer_dim=128, inference=True)
    x = torch.randn(8, 16, 96)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_hard_negative_filter_drops_easy_negatives():
    # Easy negatives = (label=0, sigmoid pred close to 0) — should be dropped
    # Easy positives = (label=1, sigmoid pred close to 1) — should be dropped
    # Hard samples in middle — should be kept
    preds_sigmoid = torch.tensor([0.0005, 0.5, 0.9999, 0.3, 0.7])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    mask = hard_negative_filter(preds_sigmoid, labels, low=0.001, high=0.999)
    # idx 0: neg, p=0.0005 < 0.001 → drop
    # idx 1: neg, p=0.5 → keep
    # idx 2: pos, p=0.9999 > 0.999 → drop
    # idx 3: pos, p=0.3 → keep (hard pos)
    # idx 4: neg, p=0.7 → keep (hard neg)
    expected = torch.tensor([False, True, False, True, True])
    assert torch.equal(mask, expected)


def test_neg_weight_schedule_ramps_up():
    sched = build_neg_weight_schedule(epochs=50, start=1.0, end=4.0)
    assert len(sched) == 50
    assert sched[0] == 1.0
    assert sched[-1] == 4.0
    # Monotonic non-decreasing
    for i in range(1, 50):
        assert sched[i] >= sched[i-1]
```

- [ ] **Step 2: Create `wakeword/_training.py`**

```python
"""Model architecture + training loop. Mirrors dscripka's reference:
2-layer FC + LayerNorm, no dropout, lr=1e-4, warmup+cosine schedule,
hard-negative mining, negative-weight ramp."""
from __future__ import annotations
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def build_neg_weight_schedule(epochs: int,
                              start: float = 1.0,
                              end: float = 4.0) -> list[float]:
    """Linear ramp of the negative-class weight from start at epoch 0 to end
    at the final epoch. Pushes the model to be more confident on negatives
    later in training."""
    if epochs <= 1:
        return [end]
    step = (end - start) / (epochs - 1)
    return [start + i * step for i in range(epochs)]


def hard_negative_filter(preds_sigmoid: torch.Tensor,
                         labels: torch.Tensor,
                         low: float = 0.001,
                         high: float = 0.999) -> torch.Tensor:
    """Boolean mask selecting samples that should contribute to the gradient.
    Drops trivial samples: labels==0 with pred<low (easy negs already classified)
    and labels==1 with pred>high (easy positives already classified)."""
    labels_flat = labels.view(-1)
    preds_flat = preds_sigmoid.view(-1)
    is_easy_neg = (labels_flat == 0) & (preds_flat < low)
    is_easy_pos = (labels_flat == 1) & (preds_flat > high)
    return ~(is_easy_neg | is_easy_pos)


class WakewordModel(nn.Module):
    """Reference DNN: Flatten -> Linear -> LayerNorm -> ReLU -> Linear ->
    LayerNorm -> ReLU -> Linear -> [Sigmoid if inference else raw logit].
    Defaults match dscripka's training_models.ipynb middle ground:
    layer_dim=128, no dropout."""

    def __init__(self, layer_dim: int = 128, inference: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, layer_dim),
            nn.LayerNorm(layer_dim), nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.LayerNorm(layer_dim), nn.ReLU(),
            nn.Linear(layer_dim, 1),
        )
        self._inference = inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self._inference:
            return torch.sigmoid(out)
        return out

    def as_inference(self) -> "WakewordModel":
        """Return a clone wired for sigmoid output, suitable for ONNX export."""
        clone = WakewordModel(
            layer_dim=self.body[1].out_features, inference=True
        )
        clone.body.load_state_dict(self.body.state_dict())
        return clone


def _warmup_then_cosine(optimizer, num_steps: int,
                        warmup_frac: float = 0.1,
                        min_lr: float = 1e-5):
    """LR schedule: warmup over first warmup_frac of steps from min_lr to
    base_lr, then cosine decay back to min_lr."""
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_steps = max(1, int(num_steps * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return min_lr + (base_lr - min_lr) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cos

    return lr_at


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    layer_dim: int = 128,
    log_every: int = 5,
    print_fn: Callable[[str], None] = print,
) -> WakewordModel:
    """Train the wakeword classifier with hard-negative mining + neg-weight ramp.
    Returns the inference-wired model (sigmoid output)."""
    torch.manual_seed(42)
    model = WakewordModel(layer_dim=layer_dim, inference=False)

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples.")
    pos_weight_value = n_neg / n_pos
    print_fn(f"  Class imbalance: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight_value:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * max(1, (len(X) + batch_size - 1) // batch_size)
    lr_at = _warmup_then_cosine(optimizer, total_steps)

    neg_weight_sched = build_neg_weight_schedule(epochs)

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        epoch_loss, n_batches, n_kept_total, n_total = 0.0, 0, 0, 0

        neg_w = neg_weight_sched[epoch]
        # Weighted BCE: pos_weight controls positive class; we further
        # downweight easy negatives via mining (below) instead of using
        # a static neg_weight.
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value * neg_w]),
            reduction="none",
        )

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            step += 1

            logits = model(xb)
            with torch.no_grad():
                preds_sig = torch.sigmoid(logits)
                mask = hard_negative_filter(preds_sig, yb)
            losses = loss_fn(logits, yb).view(-1)
            kept = losses[mask]
            n_total += len(losses)
            n_kept_total += int(mask.sum())
            if len(kept) == 0:
                continue
            loss = kept.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            kept_pct = 100 * n_kept_total / max(1, n_total)
            print_fn(f"  Epoch {epoch+1:3d}/{epochs}  "
                     f"loss={epoch_loss / max(1, n_batches):.4f}  "
                     f"kept={kept_pct:.1f}%  neg_w={neg_w:.2f}  "
                     f"lr={optimizer.param_groups[0]['lr']:.5f}")

    return model.as_inference()
```

- [ ] **Step 3: Run tests — verify passes**

```bash
.venv-wake/Scripts/python.exe -m pytest tests/wakeword/test_training.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Replace the inline training section in `train.py`**

Find the section starting `# Define model: Flatten` (~line 343) and ending right before `# Evaluate on training set` (~line 395). Replace the whole model-definition + training loop with:

```python
    # ----- Train (reference-aligned) -----
    from wakeword._training import train_model

    inference_model_pt = train_model(
        X.astype(np.float32),
        y.astype(np.float32),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-4,
        layer_dim=128,
    )
    inference_model = inference_model_pt  # alias for downstream ONNX export
```

Also remove the now-unused constants `LAYER_DIM`, `DROPOUT` from the top of `train.py` and adjust `N_EPOCHS` if you want (50 is fine for the new pipeline).

- [ ] **Step 5: Run end-to-end (will take 5-15 min)**

```bash
.venv-wake/Scripts/python.exe wakeword/train.py > wakeword/_train.log 2>&1
echo "EXIT $?"
grep -E "Epoch|scores|saved|kept" wakeword/_train.log
```

Expected: training completes, TFLite saved, scores show positive mean ≥ 0.9 and negative max < 0.5 (better than before).

- [ ] **Step 6: Commit**

```bash
git add wakeword/_training.py wakeword/train.py tests/wakeword/test_training.py
git commit -m "wakeword: reference-aligned model + training loop

WakewordModel: 2-layer FC (Linear-LN-ReLU)*2 + Linear, no dropout,
layer_dim=128 (matches dscripka's training_models.ipynb middle variant).

train_model: lr=1e-4 (was 1e-3), warmup-then-cosine LR over 10% warmup,
hard-negative filtering each batch (drops samples with pred<0.001 if neg
or pred>0.999 if pos), linear negative-weight ramp from 1.0 -> 4.0 over
epochs, BCEWithLogitsLoss with pos_weight = n_neg/n_pos × neg_w_at_epoch."
```

---

### Task 8: Offline silence smoke test

**Files:**
- Create: `wakeword/test_silence.py`
- This file is also runnable from the existing `test_tflite.py` slot — keep both; the old test_tflite.py uses the wrong feature pipeline (openwakeword instead of pyopen_wakeword) and is misleading; we leave it for now but the new test_silence.py is what we actually trust.

**Interfaces:**
- Consumes: a compiled TFLite at `wakeword/models/igor_v0.3.tflite` (or override via `--model`).
- Produces: exit 0 if all silence/noise tests pass; exit 1 otherwise. Prints per-test max-score / 3-consecutive-above-threshold summary.

- [ ] **Step 1: Create `wakeword/test_silence.py`**

```python
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
import tensorflow as tf

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

    interp = tf.lite.Interpreter(model_path=str(model_path))
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
```

- [ ] **Step 2: Run smoke test against the freshly retrained model**

```bash
.venv-wake/Scripts/python.exe wakeword/test_silence.py --model wakeword/models/igor_v0.3.tflite
```

Expected: 0-fail summary. If failures show up, they reveal exactly which audio class still triggers — direct signal for the next training iteration.

- [ ] **Step 3: Commit**

```bash
git add wakeword/test_silence.py
git commit -m "wakeword: offline silence/noise smoke test

Generates 9 synthetic audio classes (zeros, 3 noise floor variants, 5
tone variants) and scores them against the trained TFLite via the
pyopen_wakeword feature pipeline — same as the Pi runtime. Fails (exit 1)
if any class produces 3 consecutive windows above threshold.

Use after every retrain BEFORE deploying to the Pi. Saves a round trip
when the model is obviously broken."
```

---

### Task 9: User records real mic-floor silence on the Pi

**Files:** none (operational task)

**Interfaces:** the new clips go into `wakeword/samples/negative/` on both the Pi and (after scp) the PC.

- [ ] **Step 1: Stop the satellite (frees the mic)**

```bash
sudo systemctl stop wyoming-satellite
```

- [ ] **Step 2: Make the room quiet**

TV off, music off, no podcasts. You can talk normally with other humans — that's also useful negative data — but no ongoing media playback.

- [ ] **Step 3: Run the recorder for 20-30 minutes**

```bash
cd ~/igor
python3 wakeword/record_negatives.py
```

Expected: prints "Clips already on disk: N" with N being the existing count (we want to ADD to those, not replace). Let it run for 20-30 min — that's ~400-600 new 3-second clips of actual mic-floor noise.

- [ ] **Step 4: Stop with Ctrl+C, restart the satellite**

```bash
sudo systemctl start wyoming-satellite
```

- [ ] **Step 5: SCP the new negatives to PC**

From your PC (PowerShell):

```powershell
scp -r samda@10.0.30.5:~/igor/wakeword/samples/negative C:\Users\samda\OneDrive\Documents\Repos\igor\wakeword\samples\
```

Then verify total count:

```powershell
(Get-ChildItem C:\Users\samda\OneDrive\Documents\Repos\igor\wakeword\samples\negative\neg_*.wav).Count
```

Expected: >= 1600 (1226 you had + ~400+ new mic-floor clips).

---

### Task 10: Retrain + smoke + deploy

**Files:** none (operational)

**Interfaces:** the deployed TFLite is `~/wyoming-openwakeword/custom-models/igor_v0.3.tflite` on the Pi. The Pi service file gets `--vad-threshold 0.5`.

- [ ] **Step 1: Add `--vad-threshold` to the service file**

Modify `deploy/wyoming-openwakeword.service` line:

Find:
```
ExecStart=/home/samda/wyoming-openwakeword/script/run --uri tcp://0.0.0.0:10400 --custom-model-dir /home/samda/wyoming-openwakeword/custom-models --preload-model igor --preload-model okay_nabu --threshold 0.5 --trigger-level 3 --debug
```

Replace with:
```
ExecStart=/home/samda/wyoming-openwakeword/script/run --uri tcp://0.0.0.0:10400 --custom-model-dir /home/samda/wyoming-openwakeword/custom-models --preload-model igor --preload-model okay_nabu --threshold 0.5 --trigger-level 3 --vad-threshold 0.5 --debug
```

- [ ] **Step 2: Retrain with the full pipeline**

```bash
.venv-wake/Scripts/python.exe wakeword/train.py > wakeword/_train.log 2>&1
echo "EXIT $?"
grep -E "Loaded|Augmented|Epoch|scores|saved" wakeword/_train.log
```

Expected: TFLite saved to `wakeword/models/igor_v0.3.tflite`. Positive mean ≥ 0.9, negative max ideally < 0.5 (but with ~1600 real negs it might be 0.7-0.85; the silence smoke test is the real gate).

- [ ] **Step 3: Offline silence smoke**

```bash
.venv-wake/Scripts/python.exe wakeword/test_silence.py
```

Expected: "All silence/noise tests passed." If any FAIL, the model isn't ready — iterate (more mic-floor recordings, tune augmentation knobs) before deploying.

- [ ] **Step 4: Deploy to Pi**

From PC:

```powershell
scp wakeword/models/igor_v0.3.tflite samda@10.0.30.5:/tmp/
ssh samda@10.0.30.5 "cp /tmp/igor_v0.3.tflite ~/wyoming-openwakeword/custom-models/ && rm -f ~/wyoming-openwakeword/custom-models/igor_v0.2.tflite ~/wyoming-openwakeword/custom-models/igor_v0.1.tflite && sudo cp ~/igor/deploy/wyoming-openwakeword.service /etc/systemd/system/wyoming-openwakeword.service && sudo systemctl daemon-reload && sudo systemctl restart wyoming-openwakeword wyoming-satellite"
```

(Copies the new TFLite, deletes old versions, refreshes the service file with the new VAD flag, restarts.)

- [ ] **Step 5: Verify in production**

```bash
ssh samda@10.0.30.5 "journalctl -u wyoming-openwakeword -u wyoming-satellite --since '5 min ago'"
```

Look for:
- Catalog cached successfully.
- Quiet room sits silent for 5-10 minutes.
- Saying "Okay Nabu" / "Igor" fires reliably.

- [ ] **Step 6: Commit deploy artifacts + final**

```bash
git add deploy/wyoming-openwakeword.service
git commit -m "deploy: add --vad-threshold 0.5 to wyoming-openwakeword

Silero VAD gate at the openwakeword runtime suppresses predictions during
non-speech audio. Belt-and-suspenders for the now-aligned igor_v0.3
model — even if a noise-floor signature occasionally trips the model, the
VAD layer prevents it from cascading into a wake event."
git push
```

---

## Self-Review

**Spec coverage** (cross-checked against the research report's punch list):

1. ✅ Fix window labeling → Task 2 + Task 6 (`build_positive_window`, integrated).
2. ✅ Fix frame rate → Task 1 (constant + tests + CLAUDE.md).
3. ✅ SNR-mixed positive augmentation → Task 4 (`mix_with_background`) + Task 6 (wired).
4. ✅ Drop synthetic noise/silence floor → Task 3.
5. ✅ Match reference hyperparams (lr=1e-4, layer_dim=128, no dropout, warmup-cosine) → Task 7.
6. ✅ Hard-negative mining → Task 7 (`hard_negative_filter`).
7. ✅ Negative-weight schedule → Task 7 (`build_neg_weight_schedule`).
8. ✅ Peak-normalize negatives → Task 3 (negatives load through `normalize_peak`).
9. ✅ Piper-synthesized positives → Task 5.
10. ✅ Runtime VAD threshold → Task 10 Step 1.
11. ✅ Silence smoke test → Task 8.
12. ✅ Record real mic-floor silence on Pi → Task 9.

**Placeholder scan:** None present. Every code step shows full code; every command shows expected output.

**Type consistency:** `build_positive_window(clip_emb, jitter_ms, rng)` signature used identically in Tasks 2 and 6. `mix_with_background(positive, background, snr_db, rng)` matches Tasks 4 and 6. `WakewordModel(layer_dim, inference)` and `train_model(X, y, *, epochs, batch_size, lr, layer_dim)` consistent in Tasks 7 and beyond. `EMB_RATE_HZ = 12.125` consistent across Tasks 1, 2, 6. `FEATURE_RATE_HZ = 12.125` in `wakeword/contracts.py` Task 1.

**Operational dependency:** Task 9 (user records silence on Pi) gates Task 10 (deploy). Tasks 1-8 can be done back-to-back with no Pi interaction. Recommended order: 1→2→3→4→5→6→7→8 (all code) → 9 (user records on Pi, ~30 min) → 10 (retrain + deploy + verify).
