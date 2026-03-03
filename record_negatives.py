#!/usr/bin/env python3
"""Record negative samples for OpenWakeWord training.

Run on the Pi. Just talk, watch TV, play music — anything that is NOT the wake word.
Audio is automatically split into 3-second clips. Press Ctrl+C when done.

Aim for at least 100 clips. More variety = fewer false positives.
Good negative content:
  - Normal conversation (anything except "Igor")
  - TV / music / podcast playing
  - Background noise, fans, kitchen sounds
  - Other people talking
  - Silence / room tone
"""
import signal
import sys
import wave
from pathlib import Path

SAMPLE_RATE  = 16000
CLIP_SECONDS = 3          # Must match CLIP_SAMPLES in train_wakeword.py
OUTPUT_DIR   = Path("wakeword_samples/negative")
CHUNK        = 1024


def main():
    import pyaudio

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUTPUT_DIR.glob("*.wav")))
    count = existing

    print("=" * 60)
    print("  Negative sample recorder")
    print("=" * 60)
    print(f"  Clips saved: {existing} existing")
    print(f"  Recording {CLIP_SECONDS}-second clips continuously.")
    print(f"  Just talk, play TV/music, or make noise. Press Ctrl+C to stop.\n")
    print(f"  DO NOT say 'Igor' during this recording.\n")

    p = pyaudio.PyAudio()

    # Find USB mic (same logic as client)
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "USB" in info["name"] and info["maxInputChannels"] > 0:
            print(f"  Using device: {info['name']}")
            device_index = i
            break
    if device_index is None:
        print("  No USB microphone found. Using default input device.")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )

    frames_per_clip = int(SAMPLE_RATE / CHUNK * CLIP_SECONDS)
    stopped = False

    def _stop(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)

    print("  Recording... (Ctrl+C to stop)\n")
    try:
        while not stopped:
            frames = []
            for _ in range(frames_per_clip):
                if stopped:
                    break
                frames.append(stream.read(CHUNK, exception_on_overflow=False))

            if not frames:
                break

            filepath = OUTPUT_DIR / f"neg_{count:04d}.wav"
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            count += 1
            print(f"  [{count}] saved {filepath.name}", end="\r", flush=True)

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    new_clips = count - existing
    print(f"\n\nDone. {new_clips} new clips saved ({count} total) → {OUTPUT_DIR}/")
    if count < 100:
        print(f"  Tip: aim for 100+ clips. Run again to record more.")
    print(f"\nNext: copy wakeword_samples/ to PC and retrain:")
    print(f"  scp -r pi@192.168.0.3:~/smart_assistant/wakeword_samples/ wakeword_samples/")
    print(f"  python onnx_models/wakeword_creation/train_wakeword.py")


if __name__ == "__main__":
    main()
