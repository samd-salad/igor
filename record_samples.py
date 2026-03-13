#!/usr/bin/env python3
"""Record wake word samples for OpenWakeWord training.

Run on any machine with a microphone. Aim for 100+ samples across varied conditions.
After recording, run train_wakeword.py on the PC to produce oww_models/igor.onnx.

Usage:
    python record_samples.py                  # use default mic
    python record_samples.py --list-devices   # show available mics
    python record_samples.py --mic 7          # specify mic by index
    python record_samples.py --negative       # record negative samples instead
"""
import argparse
import math
import struct
import threading
import wave
import sys
import time
from pathlib import Path

import pyaudio

SAMPLE_RATE = 16000
DURATION = 2.0  # seconds -- say the phrase once after the beep
POSITIVE_DIR = Path("wakeword_samples/positive")
NEGATIVE_DIR = Path("wakeword_samples/negative")
GOAL = 150

VARIATIONS = [
    ("Normal",          "Say it at your normal conversational pace and volume"),
    ("Slow",            "Say it slowly and deliberately"),
    ("Fast",            "Say it quickly, like you're in a hurry"),
    ("Quiet",           "Say it softly, like you don't want to disturb anyone"),
    ("Loud",            "Say it louder than normal"),
    ("Far away",        "Step back 2-3 metres from the mic"),
    ("Close",           "Lean in close to the mic"),
    ("Tired/flat",      "Say it in a flat, monotone, tired voice"),
    ("Different room",  "Move to a different room or position"),
    ("Background noise","Have the TV or music playing quietly in the background"),
    ("Question tone",   "Say it with a rising, questioning intonation"),
    ("Casual",          "Say it as naturally as possible, like you mean it"),
]

NEG_VARIATIONS = [
    ("Conversation",    "Talk normally about anything -- weather, plans, etc."),
    ("Similar words",   "Say words that sound like Igor -- eager, ego, tiger, figure"),
    ("TV/media",        "Play TV or music at normal volume"),
    ("Silence",         "Just sit quietly -- capture ambient room noise"),
    ("Typing",          "Type on your keyboard or rustle papers"),
    ("Background chat", "Have someone else talking in the background"),
]


def find_default_devices(p: pyaudio.PyAudio) -> tuple:
    """Find BlackShark or system default mic and speaker."""
    mic_idx = spk_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info["name"].lower()
        if mic_idx is None and info["maxInputChannels"] > 0:
            if "blackshark" in name and "chat" in name:
                mic_idx = i
        if spk_idx is None and info["maxOutputChannels"] > 0:
            if "blackshark" in name and "chat" in name:
                spk_idx = i
    if mic_idx is None:
        try:
            mic_idx = p.get_default_input_device_info()["index"]
        except Exception:
            mic_idx = 0
    if spk_idx is None:
        try:
            spk_idx = p.get_default_output_device_info()["index"]
        except Exception:
            spk_idx = 0
    return mic_idx, spk_idx


def list_devices():
    p = pyaudio.PyAudio()
    print("\nInput devices (microphones):")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  {i}: {info['name']}")
    p.terminate()


def beep(p: pyaudio.PyAudio, spk_idx: int, freq: int = 800, duration: float = 0.2):
    """Play a short beep tone via callback (blocking writes don't work on some Windows devices)."""
    rate = 44100
    n = int(rate * duration)
    samples = []
    for i in range(n):
        t = i / rate
        envelope = min(1.0, i / 200, (n - i) / 200)
        value = int(24000 * envelope * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, value))))
    audio_data = b"".join(samples)
    pos = [0]
    done = threading.Event()

    def cb(in_data, frame_count, time_info, status):
        start = pos[0]
        end = start + frame_count * 2
        if end >= len(audio_data):
            data = audio_data[start:] + b"\x00" * (end - len(audio_data))
            pos[0] = end
            done.set()
            return (data, pyaudio.paComplete)
        pos[0] = end
        return (audio_data[start:end], pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate,
                    output=True, output_device_index=spk_idx,
                    stream_callback=cb, frames_per_buffer=512)
    stream.start_stream()
    done.wait(timeout=2)
    while stream.is_active():
        time.sleep(0.01)
    stream.stop_stream()
    stream.close()


def record_sample(p: pyaudio.PyAudio, mic_idx: int, spk_idx: int,
                   filepath: Path, duration: float = DURATION,
                   silent: bool = False):
    if not silent:
        # Start beep, then short pause so user says the word ~0.5s in
        beep(p, spk_idx, freq=800, duration=0.2)
        time.sleep(0.4)

    # Use callback mode — blocking stream.read() doesn't work on some Windows devices
    frames = []
    target_chunks = int(SAMPLE_RATE / 1024 * duration)
    done = threading.Event()

    def callback(in_data, frame_count, time_info, status):
        frames.append(in_data)
        if len(frames) >= target_chunks:
            done.set()
            return (None, pyaudio.paComplete)
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
        input_device_index=mic_idx,
        stream_callback=callback,
    )

    print("  * recording *", end="", flush=True)
    stream.start_stream()
    done.wait(timeout=duration + 1)
    stream.stop_stream()
    stream.close()

    if not silent:
        beep(p, spk_idx, freq=1000, duration=0.15)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"saved {filepath.name}")


def main():
    parser = argparse.ArgumentParser(description="Record wake word samples")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio input devices and exit")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--spk", type=int, default=None,
                        help="Speaker device index")
    parser.add_argument("--negative", action="store_true",
                        help="Record negative (non-wake-word) samples")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help=f"Recording duration in seconds (default {DURATION})")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    is_negative = args.negative
    output_dir = NEGATIVE_DIR if is_negative else POSITIVE_DIR
    variations = NEG_VARIATIONS if is_negative else VARIATIONS
    sample_type = "negative" if is_negative else "positive"

    output_dir.mkdir(parents=True, exist_ok=True)

    p = pyaudio.PyAudio()
    default_mic, default_spk = find_default_devices(p)
    mic_idx = args.mic if args.mic is not None else default_mic
    spk_idx = args.spk if args.spk is not None else default_spk
    mic_name = p.get_device_info_by_index(mic_idx)["name"]
    spk_name = p.get_device_info_by_index(spk_idx)["name"]

    # Count existing samples (all patterns)
    existing = len(list(output_dir.glob("*.wav")))
    count = existing

    # Find next available index
    next_idx = 0
    while (output_dir / f"sample_{next_idx:04d}.wav").exists():
        next_idx += 1

    print("=" * 60)
    print(f"  Wake word recorder -- '{sample_type}' samples")
    print("=" * 60)
    print(f"  Mic:     [{mic_idx}] {mic_name}")
    print(f"  Speaker: [{spk_idx}] {spk_name}")
    print(f"  Goal: {GOAL} samples | Existing: {existing}")
    print(f"  Each recording is {args.duration}s")
    if not is_negative:
        print(f"  Say 'Igor' once per recording")
    else:
        print(f"  Do NOT say 'Igor' -- record background/noise/speech")
    print(f"  Variation helps a lot. Follow the prompts below.\n")

    variation_index = existing % len(variations)

    while True:
        style, instruction = variations[variation_index % len(variations)]
        done = count >= GOAL

        header = f"[{count}/{GOAL}]"
        if done:
            header += " (goal reached!)"

        print(f"{header}  Style: {style}")
        print(f"          {instruction}")

        if is_negative:
            # Auto-record negatives — just capture ambient audio continuously
            if count == 0:
                print("\n  Auto-recording negatives (Ctrl+C to stop)...\n")
        else:
            try:
                input("          Press Enter to record, Ctrl+C to stop... ")
            except (KeyboardInterrupt, EOFError):
                print(f"\n\nDone. {count} total {sample_type} samples in {output_dir}/")
                if count < 50:
                    print(f"Warning: {count} samples is low. Aim for {GOAL}+ for good accuracy.")
                print(f"\nNext: python onnx_models/wakeword_creation/train_wakeword.py")
                break

        try:
            filepath = output_dir / f"sample_{next_idx:04d}.wav"
            record_sample(p, mic_idx, spk_idx, filepath, duration=args.duration,
                         silent=is_negative)
            count += 1
            next_idx += 1
            variation_index += 1
            print()
        except KeyboardInterrupt:
            print(f"\n\nDone. {count} total {sample_type} samples in {output_dir}/")
            if is_negative:
                print(f"\nNext: python onnx_models/wakeword_creation/train_wakeword.py")
            break

    p.terminate()


if __name__ == "__main__":
    main()
