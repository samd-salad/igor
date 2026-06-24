"""Generate synthetic positive wake-word clips using Piper TTS.

For each installed Piper voice we render 'Igor' in many variations
(with surrounding silence, different lengths/punctuation, varied
length_scale and noise_scale). Each output clip is 16kHz mono int16
and lives in samples/positive_synth/. Loaded alongside real positives
by wakeword/train.py.

Usage:
    python wakeword/synthesize_positives.py --per-voice 200
"""
from __future__ import annotations
import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import piper
from piper.config import SynthesisConfig

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
    parser = argparse.ArgumentParser(
        description="Generate synthetic positive wake-word clips using Piper TTS."
    )
    parser.add_argument(
        "--per-voice", type=int, default=100,
        help="Synth clips per voice (default 100)"
    )
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
        print(f"\nLoading voice: {voice_path.stem}")
        voice = piper.PiperVoice.load(voice_path)
        per_voice = args.per_voice
        for i in range(per_voice):
            phrase = PHRASES[i % len(PHRASES)]
            syn_config = SynthesisConfig(
                length_scale=float(rng.uniform(0.85, 1.20)),
                noise_scale=float(rng.uniform(0.4, 0.8)),
                noise_w_scale=float(rng.uniform(0.4, 0.9)),
            )
            audio_chunks = list(voice.synthesize(phrase, syn_config=syn_config))
            if not audio_chunks:
                print(f"  WARNING: no audio for phrase={phrase!r}, skipping")
                continue
            # Concatenate int16 bytes from all chunks
            raw = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
            audio = np.frombuffer(raw, dtype=np.int16)
            voice_sr = audio_chunks[0].sample_rate
            if voice_sr != SAMPLE_RATE:
                # Linear resample to 16kHz
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

    total_new = idx - existing
    print(f"\nDone. {total_new} synthetic positives written to {OUT_DIR}/.")


if __name__ == "__main__":
    main()
