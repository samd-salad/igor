#!/usr/bin/env python3
"""CLI tool for enrolling speakers for voice identification."""

import argparse
import sys
import time
import wave
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD
from server.speaker_id import SpeakerIdentifier


def record_audio(duration: float, sample_rate: int = 16000) -> bytes:
    """Record audio from microphone."""
    import pyaudio

    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )

    print(f"Recording for {duration} seconds...")
    frames = []
    for _ in range(int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)


def audio_bytes_to_numpy(audio_bytes: bytes, sample_rate: int = 16000):
    """Convert raw audio bytes to numpy array."""
    import numpy as np
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


ENROLLMENT_VARIATIONS = [
    ("Conversational",   "Chat naturally — tell us what you had for breakfast, or describe your day"),
    ("Read aloud",       "Read something nearby: a label, a message on your phone, anything"),
    ("Different volume", "Speak a bit quieter than usual, like you're in a library"),
    ("Faster pace",      "Talk a little faster than normal, like you're in a hurry to explain something"),
    ("Different topic",  "Count slowly from 1 to 20, filling the time with anything extra"),
    ("Casual/relaxed",   "Describe a room you're in — furniture, colours, whatever comes to mind"),
    ("Expressive",       "Tell a short joke or something funny that happened recently"),
    ("Monotone",         "Recite the alphabet or days of the week in a flat, even tone"),
]


def enroll_interactive(name: str, num_samples: int = 5, duration: float = 5.0):
    """Interactively enroll a speaker with multiple voice samples."""
    identifier = SpeakerIdentifier(SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD)

    print(f"\n{'='*60}")
    print(f"  Speaker enrollment — {name}")
    print(f"{'='*60}")
    print(f"  Samples: {num_samples}  |  Duration: {duration}s each")
    print(f"  Each sample captures your voice characteristics.")
    print(f"  Variety across samples improves recognition accuracy.\n")

    samples = []
    for i in range(num_samples):
        style, instruction = ENROLLMENT_VARIATIONS[i % len(ENROLLMENT_VARIATIONS)]
        print(f"[{i+1}/{num_samples}]  Style: {style}")
        print(f"          {instruction}")
        input("          Press Enter to record... ")
        audio_bytes = record_audio(duration)
        audio = audio_bytes_to_numpy(audio_bytes)
        samples.append(audio)
        print(f"  Sample {i+1} saved.\n")

    print("Processing voice samples...")
    success = identifier.enroll_speaker(name, samples, sample_rate=16000)

    if success:
        print(f"\n✓ Successfully enrolled '{name}'!")
        print(f"  Total enrolled speakers: {identifier.list_speakers()}")
    else:
        print(f"\n✗ Failed to enroll '{name}'. Check logs for details.")

    return success


def enroll_from_files(name: str, audio_files: list[str]):
    """Enroll a speaker from audio files."""
    import soundfile as sf
    import numpy as np

    identifier = SpeakerIdentifier(SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD)

    print(f"\n=== Enrolling speaker: {name} from files ===")

    samples = []
    for filepath in audio_files:
        print(f"Loading: {filepath}")
        try:
            audio, sr = sf.read(filepath)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Resample if needed
            if sr != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            samples.append(audio.astype(np.float32))
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue

    if not samples:
        print("No valid audio files loaded!")
        return False

    print(f"Processing {len(samples)} voice samples...")
    success = identifier.enroll_speaker(name, samples, sample_rate=16000)

    if success:
        print(f"\n✓ Successfully enrolled '{name}'!")
    else:
        print(f"\n✗ Failed to enroll '{name}'.")

    return success


def test_identification():
    """Test speaker identification with a recording."""
    identifier = SpeakerIdentifier(SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD)

    if not identifier.list_speakers():
        print("No speakers enrolled! Enroll someone first.")
        return

    print(f"\n=== Testing Speaker Identification ===")
    print(f"Enrolled speakers: {identifier.list_speakers()}")

    input("\nPress Enter to record a 3-second test sample...")
    audio_bytes = record_audio(3.0)
    audio = audio_bytes_to_numpy(audio_bytes)

    print("Identifying speaker...")
    start = time.perf_counter()
    result = identifier.identify(audio, sample_rate=16000)
    elapsed = time.perf_counter() - start

    print(f"\nResult:")
    print(f"  Speaker: {result.name}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Is known: {result.is_known}")
    print(f"  Time: {elapsed*1000:.1f}ms")


def list_speakers():
    """List all enrolled speakers."""
    identifier = SpeakerIdentifier(SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD)
    speakers = identifier.list_speakers()

    print(f"\n=== Enrolled Speakers ({len(speakers)}) ===")
    if speakers:
        for name in speakers:
            print(f"  - {name}")
    else:
        print("  No speakers enrolled.")


def remove_speaker(name: str):
    """Remove a speaker from the database."""
    identifier = SpeakerIdentifier(SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD)

    if identifier.remove_speaker(name):
        print(f"✓ Removed speaker '{name}'")
    else:
        print(f"✗ Speaker '{name}' not found")


def main():
    parser = argparse.ArgumentParser(description="Speaker enrollment and identification tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("name", help="Speaker's name")
    enroll_parser.add_argument("--files", nargs="+", help="Audio files to use (optional)")
    enroll_parser.add_argument("--samples", type=int, default=5, help="Number of samples to record (default: 5)")
    enroll_parser.add_argument("--duration", type=float, default=5.0, help="Duration of each sample in seconds (default: 5)")

    # Test command
    subparsers.add_parser("test", help="Test speaker identification")

    # List command
    subparsers.add_parser("list", help="List enrolled speakers")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a speaker")
    remove_parser.add_argument("name", help="Speaker's name to remove")

    args = parser.parse_args()

    if args.command == "enroll":
        if args.files:
            enroll_from_files(args.name, args.files)
        else:
            enroll_interactive(args.name, args.samples, args.duration)
    elif args.command == "test":
        test_identification()
    elif args.command == "list":
        list_speakers()
    elif args.command == "remove":
        remove_speaker(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
