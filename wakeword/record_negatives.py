#!/usr/bin/env python3
"""Record negative samples for OpenWakeWord training.

Runs on the Pi using arecord — the same path wyoming-satellite uses
(plughw:CARD=CODEC,DEV=0). No pyaudio, no PortAudio device discovery,
no JACK weirdness.

Stop wyoming-satellite first (it owns the mic), then run this. Audio is
chunked into 3-second WAV clips back-to-back. Press Ctrl+C when done.

Aim for at least 100 clips. More variety = fewer false positives.
Good negative content:
  - Normal conversation (anything except "Igor")
  - TV / music / podcast playing
  - Background noise, fans, kitchen sounds
  - Other people talking
  - Silence / room tone
"""
from __future__ import annotations
import signal
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE  = 16000
CLIP_SECONDS = 3                # Must match CLIP_SAMPLES in train.py
OUTPUT_DIR   = Path(__file__).parent / "samples" / "negative"
ALSA_DEVICE  = "plughw:CARD=CODEC,DEV=0"


def _next_index(out_dir: Path) -> int:
    """Find first unused neg_NNNN.wav slot so we don't overwrite."""
    existing = list(out_dir.glob("neg_*.wav"))
    nums: list[int] = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return (max(nums) + 1) if nums else 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_idx = _next_index(OUTPUT_DIR)
    count = start_idx

    print("=" * 60)
    print("  Negative sample recorder (arecord)")
    print("=" * 60)
    print(f"  Device:               {ALSA_DEVICE}")
    print(f"  Clips already on disk: {start_idx}")
    print(f"  Recording {CLIP_SECONDS}-second clips back-to-back.")
    print(f"  Just play TV / podcast / music / talk. Ctrl+C to stop.")
    print(f"  DO NOT say 'Igor' during recording.")
    print()

    stopped = False

    def _stop(sig, frame):
        nonlocal stopped
        stopped = True
        print("\n  Stopping after current clip...", flush=True)

    signal.signal(signal.SIGINT, _stop)

    while not stopped:
        filepath = OUTPUT_DIR / f"neg_{count:04d}.wav"
        cmd = [
            "arecord",
            "-D", ALSA_DEVICE,
            "-r", str(SAMPLE_RATE),
            "-c", "1",
            "-f", "S16_LE",
            "-d", str(CLIP_SECONDS),
            "-q",
            str(filepath),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"\n  arecord exited {proc.returncode}; aborting.", file=sys.stderr)
            print(f"  Verify the device with: arecord -l", file=sys.stderr)
            sys.exit(proc.returncode)
        count += 1
        print(f"  [{count}] saved {filepath.name}", end="\r", flush=True)

    new_clips = count - start_idx
    print(f"\n\nDone. {new_clips} new clips ({count} total) -> {OUTPUT_DIR}/")
    if count < 100:
        print(f"  Tip: aim for 100+ clips. Run again to record more.")
    print("\nNext (on your PC):")
    print(f"  scp -r samda@10.0.30.5:~/igor/wakeword/samples/negative wakeword/samples/")
    print(f"  python wakeword/train.py")


if __name__ == "__main__":
    main()
