#!/usr/bin/env python3
"""Record wake word samples for OpenWakeWord training.

Run this on the Pi. Aim for 100+ samples across varied conditions.
Transfer the wakeword_samples/ directory to your PC when done, then run train_wakeword.py.
"""
import wave
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 2.0  # seconds — enough time to say the phrase once clearly
OUTPUT_DIR = Path("wakeword_samples/positive")
GOAL = 150

VARIATIONS = [
    ("Normal",          "Say it at your normal conversational pace and volume"),
    ("Slow",            "Say it slowly and deliberately"),
    ("Fast",            "Say it quickly, like you're in a hurry"),
    ("Quiet",           "Say it softly, like you don't want to disturb anyone"),
    ("Loud",            "Say it louder than normal"),
    ("Far away",        "Step back 2–3 metres from the mic"),
    ("Close",           "Lean in close to the mic"),
    ("Tired/flat",      "Say it in a flat, monotone, tired voice"),
    ("Different room",  "Move to a different room or position"),
    ("Background noise","Have the TV or music playing quietly in the background"),
    ("Question tone",   "Say it with a rising, questioning intonation"),
    ("Casual",          "Say it as naturally as possible, like you mean it"),
]


def record_sample(index: int) -> Path:
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )

    print("  [recording]", end=" ", flush=True)
    frames = []
    for _ in range(int(SAMPLE_RATE / 1024 * DURATION)):
        frames.append(stream.read(1024, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    p.terminate()

    filepath = OUTPUT_DIR / f"sample_{index:04d}.wav"
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"saved {filepath.name}")
    return filepath


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUTPUT_DIR.glob("*.wav")))
    count = existing

    print("=" * 60)
    print("  Wake word recorder — 'Doctor Butts'")
    print("=" * 60)
    print(f"  Goal: {GOAL} samples | Existing: {existing}")
    print(f"  Each recording is {DURATION}s — say the phrase once per recording")
    print(f"  Variation helps a lot. Follow the prompts below.\n")

    variation_index = existing % len(VARIATIONS)

    while True:
        style, instruction = VARIATIONS[variation_index % len(VARIATIONS)]
        remaining = max(0, GOAL - count)
        done = count >= GOAL

        header = f"[{count}/{GOAL}]"
        if done:
            header += " (goal reached!)"

        print(f"{header}  Style: {style}")
        print(f"          {instruction}")

        try:
            input("          Press Enter to record, Ctrl+C to stop... ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n\nDone. {count} samples saved to {OUTPUT_DIR}/")
            if count < 50:
                print(f"Warning: {count} samples is low. Aim for {GOAL}+ for good accuracy.")
            print(f"\nNext: rsync wakeword_samples/ to your PC, then run train_wakeword.py")
            break

        record_sample(count)
        count += 1
        variation_index += 1
        print()


if __name__ == "__main__":
    main()
