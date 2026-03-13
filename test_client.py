#!/usr/bin/env python3
"""Local PC test client — multiple modes for testing Igor pipeline + wake word.

Modes:
    python test_client.py                          # interactive push-to-talk
    python test_client.py --text "what time is it" # text-only, no audio
    python test_client.py --wav sample.wav         # send pre-recorded WAV
    python test_client.py --wakeword               # live wake word scoring
    python test_client.py --wakeword --wav-dir wakeword_samples/positive  # batch score
    python test_client.py --list-devices           # show audio devices

All modes log structured JSONL to data/test_client.jsonl for AI auditing.
Server logs go to data/server.log (read via MCP tail_logs).
"""
import argparse
import base64
import io
import json
import math
import queue
import struct
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import pyaudio
import requests

SERVER_URL = "http://192.168.0.4:8000"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 100  # Peak-based (BlackShark silence=17-27, speech=333+)
SILENCE_DURATION = 1.5
MIN_RECORDING = 0.5
MAX_RECORDING = 15

# Wake word defaults
OWW_CHUNK = 1280  # 80ms frames for OpenWakeWord
OWW_THRESHOLD = 0.7
OWW_TRIGGER_FRAMES = 8

# Audio normalization — required for low-gain mics like BlackShark V3 Pro
# which output RMS ~5-6 even during speech (Razer USB firmware issue).
# Normalizes each chunk so OWW sees proper signal levels.
NORMALIZE_TARGET_PEAK = 16000  # Target peak amplitude after normalization
NORMALIZE_MAX_GAIN = 1000.0    # Cap gain to avoid extreme amplification of near-zero audio
NORMALIZE_FLOOR = 50           # Don't normalize below this peak (silence=17-27, speech=333+)
SPEECH_PEAK_MIN = 100          # Peak above this activates speech window
SPEECH_WINDOW_FRAMES = 12     # Keep speech window open for ~1s after last high-peak frame

LOG_FILE = Path("data/test_client.jsonl")

# ANSI colors
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Logging ──────────────────────────────────────────────────────────────────

def log_event(event: str, **data):
    """Append a JSONL event to the log file."""
    entry = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event}
    entry.update(data)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Audio helpers ────────────────────────────────────────────────────────────

def list_devices():
    p = pyaudio.PyAudio()
    print("\nInput devices (microphones):")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  {i}: {info['name']}")
    print("\nOutput devices (speakers):")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"  {i}: {info['name']}")
    p.terminate()


def find_default_devices(p: pyaudio.PyAudio) -> tuple:
    """Find the first BlackShark or default mic/speaker."""
    mic_idx = None
    spk_idx = None
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


def record_audio(p: pyaudio.PyAudio, mic_idx: int) -> bytes:
    """Record until silence detected. Returns WAV bytes.

    Uses callback mode because blocking stream.read() doesn't work
    on some Windows audio devices (returns instantly without blocking).
    """
    frames = []
    silent_chunks = 0
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
    min_chunks = int(MIN_RECORDING * SAMPLE_RATE / CHUNK)
    max_chunks = int(MAX_RECORDING * SAMPLE_RATE / CHUNK)
    has_speech = False
    done = threading.Event()
    chunk_count = 0

    def callback(in_data, frame_count, time_info, status):
        nonlocal silent_chunks, has_speech, chunk_count
        frames.append(in_data)
        chunk_count += 1

        samples = struct.unpack(f"<{len(in_data)//2}h", in_data)
        peak = max(abs(s) for s in samples)

        if peak > SILENCE_THRESHOLD:
            silent_chunks = 0
            has_speech = True
        else:
            silent_chunks += 1

        if has_speech and silent_chunks >= chunks_for_silence and chunk_count >= min_chunks:
            done.set()
            return (None, pyaudio.paComplete)
        if chunk_count >= max_chunks:
            done.set()
            return (None, pyaudio.paComplete)
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK,
        input_device_index=mic_idx,
        stream_callback=callback,
    )
    stream.start_stream()
    done.wait(timeout=MAX_RECORDING + 1)
    stream.stop_stream()
    stream.close()

    if not has_speech:
        return b""

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def play_audio(p: pyaudio.PyAudio, spk_idx: int, wav_bytes: bytes):
    """Play WAV audio through speakers (callback mode for Windows compat)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        rate = wf.getframerate()

    pos = [0]
    frame_size = sample_width * channels
    done = threading.Event()

    def cb(in_data, frame_count, time_info, status):
        start = pos[0]
        end = start + frame_count * frame_size
        if end >= len(audio_data):
            data = audio_data[start:] + b"\x00" * (end - len(audio_data))
            pos[0] = end
            done.set()
            return (data, pyaudio.paComplete)
        pos[0] = end
        return (audio_data[start:end], pyaudio.paContinue)

    stream = p.open(
        format=p.get_format_from_width(sample_width),
        channels=channels,
        rate=rate,
        output=True,
        output_device_index=spk_idx,
        stream_callback=cb,
        frames_per_buffer=1024,
    )
    stream.start_stream()
    done.wait(timeout=30)
    while stream.is_active():
        time.sleep(0.01)
    stream.stop_stream()
    stream.close()


def normalize_audio(audio, target_peak: int = NORMALIZE_TARGET_PEAK,
                    floor: int = NORMALIZE_FLOOR,
                    max_gain: float = NORMALIZE_MAX_GAIN):
    """Normalize int16 audio chunk to target peak amplitude.

    Only normalizes when peak >= floor. Silence (peak 17-27 on BlackShark)
    passes through unchanged so OWW's mel buffer stays clean. Speech
    (peak 333+) gets boosted to proper levels for the classifier.
    """
    import numpy as np
    peak = int(np.max(np.abs(audio)))
    if peak < floor:
        return audio  # Silence — don't amplify noise floor
    gain = min(target_peak / peak, max_gain)
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def beep(p: pyaudio.PyAudio, spk_idx: int, freq_start: int = 800,
         freq_end: int = None, duration: float = 0.12, vol: float = 0.25):
    """Play a beep tone with optional frequency sweep (matches Pi sox beeps).

    Args:
        freq_start: Starting frequency in Hz
        freq_end:   Ending frequency (None = constant tone)
        duration:   Duration in seconds
        vol:        Volume 0.0-1.0
    """
    if freq_end is None:
        freq_end = freq_start
    rate = 44100
    n = int(rate * duration)
    amplitude = int(32767 * vol)
    samples = []
    for i in range(n):
        t = i / rate
        frac = i / n
        freq = freq_start + (freq_end - freq_start) * frac
        envelope = min(1.0, i / 200, (n - i) / 200)
        value = int(amplitude * envelope * math.sin(2 * math.pi * freq * t))
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

    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate,
                        output=True, output_device_index=spk_idx,
                        stream_callback=cb, frames_per_buffer=512)
        stream.start_stream()
        done.wait(timeout=2)
        while stream.is_active():
            time.sleep(0.01)
        stream.stop_stream()
        stream.close()
    except Exception:
        pass


def beep_start(p, spk_idx):
    """Rising sweep — matches Pi sox 'sine 500:900'."""
    beep(p, spk_idx, freq_start=500, freq_end=900, duration=0.12, vol=0.25)


def beep_end(p, spk_idx):
    """Falling sweep — matches Pi sox 'sine 700:400'."""
    beep(p, spk_idx, freq_start=700, freq_end=400, duration=0.12, vol=0.2)


def beep_error(p, spk_idx):
    """Low warning tone."""
    beep(p, spk_idx, freq_start=200, freq_end=200, duration=0.3, vol=0.2)


# ── Server communication ────────────────────────────────────────────────────

def send_audio(wav_bytes: bytes, server_url: str) -> dict:
    """Send audio to server and return response."""
    audio_b64 = base64.b64encode(wav_bytes).decode()
    resp = requests.post(
        f"{server_url}/api/process_interaction",
        json={
            "audio_base64": audio_b64,
            "wake_word": "igor",
            "timestamp": time.time(),
            "prefer_sonos_output": False,
            "client_id": "test_pc",
            "room_id": "default",
        },
        timeout=30,
    )
    return resp.json()


def send_text(text: str, server_url: str) -> dict:
    """Send text to server (no STT/TTS) and return response."""
    resp = requests.post(
        f"{server_url}/api/text_interaction",
        json={
            "text": text,
            "client_id": "test_pc",
            "room_id": "default",
        },
        timeout=30,
    )
    return resp.json()


def display_result(result: dict):
    """Print server response to console."""
    transcription = result.get("transcription", "")
    response_text = result.get("response_text", "")
    cmds = result.get("commands_executed", [])
    error = result.get("error")
    timings = result.get("timings", {})
    speaker = result.get("speaker")

    if transcription:
        print(f"  {BOLD}You:{RESET} {transcription}")
    if speaker:
        print(f"  {BOLD}Speaker:{RESET} {speaker}")
    if cmds:
        print(f"  {BOLD}Commands:{RESET} {', '.join(cmds)}")
    if error:
        print(f"  {RED}Error: {error}{RESET}")
    print(f"  {BOLD}Igor:{RESET} {response_text}")
    if timings:
        parts = [f"{k}={timings[k]:.0f}ms" for k in ["stt", "llm", "tts", "total"] if k in timings]
        if parts:
            print(f"  {YELLOW}{' | '.join(parts)}{RESET}")


def log_result(result: dict, mode: str):
    """Log server response as JSONL."""
    log_event(
        "response",
        mode=mode,
        transcription=result.get("transcription", ""),
        response_text=result.get("response_text", ""),
        commands=result.get("commands_executed", []),
        error=result.get("error"),
        timings=result.get("timings", {}),
        speaker=result.get("speaker"),
        await_followup=result.get("await_followup", False),
    )


def check_server(server_url: str) -> bool:
    """Check server health, return True if up."""
    try:
        r = requests.get(f"{server_url}/api/health", timeout=3)
        if r.ok:
            print(f"  {GREEN}Server is up{RESET}")
            return True
        print(f"  {RED}Server returned {r.status_code}{RESET}")
    except Exception:
        print(f"  {RED}Server not reachable at {server_url}{RESET}")
    return False


# ── Mode: text ───────────────────────────────────────────────────────────────

def mode_text(args):
    """Send text query, print response, exit."""
    server_url = args.server
    print(f"\n{BOLD}Text mode{RESET} -> {server_url}")
    if not check_server(server_url):
        return

    print(f"  {YELLOW}Sending: {args.text}{RESET}")
    log_event("send_text", text=args.text, server=server_url)

    try:
        result = send_text(args.text, server_url)
    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")
        log_event("error", error=str(e))
        return

    display_result(result)
    log_result(result, mode="text")
    print()


# ── Mode: wav ────────────────────────────────────────────────────────────────

def mode_wav(args):
    """Send a pre-recorded WAV file through the full pipeline."""
    server_url = args.server
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"{RED}File not found: {wav_path}{RESET}")
        return

    print(f"\n{BOLD}WAV mode{RESET} -> {server_url}")
    if not check_server(server_url):
        return

    wav_bytes = wav_path.read_bytes()
    size_kb = len(wav_bytes) / 1024
    print(f"  Sending {wav_path.name} ({size_kb:.0f}KB)...")
    log_event("send_wav", file=str(wav_path), size_kb=round(size_kb, 1), server=server_url)

    try:
        result = send_audio(wav_bytes, server_url)
    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")
        log_event("error", error=str(e))
        return

    display_result(result)
    log_result(result, mode="wav")
    print()


# ── Mode: interactive ────────────────────────────────────────────────────────

def mode_interactive(args):
    """Push-to-talk loop with Enter key."""
    server_url = args.server
    p = pyaudio.PyAudio()

    if args.mic is not None and args.spk is not None:
        mic_idx, spk_idx = args.mic, args.spk
    else:
        mic_idx, spk_idx = find_default_devices(p)

    mic_name = p.get_device_info_by_index(mic_idx)["name"]
    spk_name = p.get_device_info_by_index(spk_idx)["name"]

    print(f"\n{BOLD}Igor Test Client{RESET}")
    print(f"  Server:  {server_url}")
    print(f"  Mic:     [{mic_idx}] {mic_name}")
    print(f"  Speaker: [{spk_idx}] {spk_name}")
    print(f"\n  Press {BOLD}Enter{RESET} to talk, {BOLD}Ctrl+C{RESET} to quit.\n")

    check_server(server_url)
    log_event("session_start", mode="interactive", server=server_url,
              mic=mic_name, spk=spk_name)
    print()

    while True:
        try:
            input(f"{BLUE}[Enter to talk]{RESET} ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        beep_start(p, spk_idx)

        print(f"  {BLUE}Listening...{RESET}", end="", flush=True)
        wav_bytes = record_audio(p, mic_idx)
        if not wav_bytes:
            print(f"\r  {YELLOW}No speech detected{RESET}        ")
            beep_end(p, spk_idx)
            log_event("no_speech")
            continue

        beep_end(p, spk_idx)
        size_kb = len(wav_bytes) / 1024
        print(f"\r  {GREEN}Recorded ({size_kb:.0f}KB){RESET}     ")
        log_event("recorded", size_kb=round(size_kb, 1))

        print(f"  {YELLOW}Processing...{RESET}", end="", flush=True)
        try:
            result = send_audio(wav_bytes, server_url)
        except requests.Timeout:
            print(f"\r  {RED}Server timeout{RESET}              ")
            beep_error(p, spk_idx)
            log_event("error", error="timeout")
            continue
        except Exception as e:
            print(f"\r  {RED}Error: {e}{RESET}              ")
            beep_error(p, spk_idx)
            log_event("error", error=str(e))
            continue

        print("\r", end="")
        display_result(result)
        log_result(result, mode="interactive")

        # Play audio response
        audio_b64 = result.get("audio_base64", "")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                play_audio(p, spk_idx, audio_bytes)
            except Exception as e:
                print(f"  {RED}Playback failed: {e}{RESET}")
                log_event("playback_error", error=str(e))

        # Follow-up
        if result.get("await_followup"):
            print(f"\n  {BLUE}Follow-up expected — listening...{RESET}")
            beep_start(p, spk_idx)
            time.sleep(0.2)
            wav_bytes = record_audio(p, mic_idx)
            if wav_bytes:
                beep_end(p, spk_idx)
                print(f"  {YELLOW}Processing follow-up...{RESET}", end="", flush=True)
                try:
                    result = send_audio(wav_bytes, server_url)
                    print("\r", end="")
                    display_result(result)
                    log_result(result, mode="followup")
                    audio_b64 = result.get("audio_base64", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        play_audio(p, spk_idx, audio_bytes)
                except Exception as e:
                    print(f"\r  {RED}Follow-up failed: {e}{RESET}")
                    log_event("error", error=str(e))

        print()

    log_event("session_end")
    p.terminate()


# ── Mode: wakeword ───────────────────────────────────────────────────────────

def mode_wakeword(args):
    """Wake word detection testing — live mic or batch WAV scoring."""
    try:
        import numpy as np
        from client.wakeword import WakeWordDetector
    except ImportError as e:
        print(f"{RED}Cannot import dependency: {e}{RESET}")
        print("Install: pip install numpy openwakeword onnxruntime")
        return

    model_dir = Path("oww_models")
    model_paths = [str(p) for p in sorted(model_dir.glob("*.onnx"))]
    if not model_paths:
        print(f"{RED}No .onnx models in {model_dir}/{RESET}")
        return

    threshold = args.ww_threshold
    trigger_frames = args.ww_trigger

    print(f"\n{BOLD}Wake Word Test{RESET}")
    print(f"  Models: {', '.join(Path(p).stem for p in model_paths)}")
    print(f"  Threshold: {threshold}  Trigger frames: {trigger_frames}")

    vad_threshold = 0.0 if args.no_vad else 0.5
    detector = WakeWordDetector(model_paths, threshold=threshold,
                                vad_threshold=vad_threshold)

    # Batch mode: score WAV files
    if args.wav_dir:
        wav_dir = Path(args.wav_dir)
        if not wav_dir.exists():
            print(f"{RED}Directory not found: {wav_dir}{RESET}")
            return

        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            print(f"{YELLOW}No WAV files in {wav_dir}{RESET}")
            return

        print(f"  Scoring {len(wav_files)} files from {wav_dir}/\n")
        detected_count = 0

        for wav_path in wav_files:
            with wave.open(str(wav_path), "rb") as wf:
                if wf.getframerate() != 16000 or wf.getnchannels() != 1:
                    print(f"  {YELLOW}SKIP{RESET} {wav_path.name} (not 16kHz mono)")
                    continue
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

            peak_score = 0.0
            consecutive = 0
            triggered = False

            for i in range(0, len(audio) - OWW_CHUNK, OWW_CHUNK):
                chunk = audio[i:i + OWW_CHUNK]
                scores = detector.predict(chunk)
                for name, score in scores.items():
                    peak_score = max(peak_score, score)
                    if score >= threshold:
                        consecutive += 1
                        if consecutive >= trigger_frames:
                            triggered = True
                    else:
                        consecutive = 0

            detector.reset()
            status = f"{GREEN}HIT{RESET}" if triggered else f"{RED}MISS{RESET}"
            print(f"  {status}  peak={peak_score:.3f}  {wav_path.name}")
            log_event("wakeword_file", file=wav_path.name,
                      peak=round(peak_score, 4), triggered=triggered)
            if triggered:
                detected_count += 1

        total = len(wav_files)
        rate = detected_count / total * 100 if total else 0
        print(f"\n  {BOLD}Result: {detected_count}/{total} detected ({rate:.0f}%){RESET}")
        log_event("wakeword_batch", dir=str(wav_dir), total=total,
                  detected=detected_count, rate=round(rate, 1))
        return

    # Live mode: listen on mic
    p = pyaudio.PyAudio()
    if args.mic is not None:
        mic_idx = args.mic
    else:
        mic_idx, _ = find_default_devices(p)
    if args.spk is not None:
        spk_idx = args.spk
    else:
        _, spk_idx = find_default_devices(p)

    mic_name = p.get_device_info_by_index(mic_idx)["name"]
    spk_name = p.get_device_info_by_index(spk_idx)["name"]
    print(f"  Mic:     [{mic_idx}] {mic_name}")
    print(f"  Speaker: [{spk_idx}] {spk_name}")

    # Use a queue-based callback for audio input — blocking stream.read()
    # doesn't work on some Windows devices (returns instantly).
    audio_q = queue.Queue()

    def _audio_callback(in_data, frame_count, time_info, status):
        audio_q.put(in_data)
        return (None, pyaudio.paContinue)

    def open_ww_stream():
        s = p.open(
            rate=SAMPLE_RATE, channels=1, format=FORMAT,
            input=True, frames_per_buffer=OWW_CHUNK,
            input_device_index=mic_idx,
            stream_callback=_audio_callback,
        )
        s.start_stream()
        return s

    stream = open_ww_stream()

    # Warmup — consume frames for 2s
    print("  Warming up OWW (2s)...")
    warmup_end = time.time() + 2.0
    while time.time() < warmup_end:
        try:
            chunk = audio_q.get(timeout=0.2)
            detector.predict(chunk)
        except queue.Empty:
            pass

    pipeline = args.ww_pipeline
    print(f"  {GREEN}Listening...{RESET} Say 'Igor'. "
          f"{'Pipeline enabled.' if pipeline else 'Scores only.'} "
          f"Ctrl+C to stop.\n")
    log_event("wakeword_live_start", mic=mic_name, spk=spk_name,
              threshold=threshold, trigger_frames=trigger_frames,
              pipeline=pipeline)

    consecutive = {}
    speech_countdown = 0  # Sticky window: frames remaining with speech active
    score_log = Path("data/wakeword_scores.csv")
    score_log.parent.mkdir(exist_ok=True)
    csv = open(score_log, "w", encoding="utf-8")
    csv.write("timestamp,model,score,raw_peak,rms,consecutive,speech,triggered\n")

    try:
        while True:
            try:
                audio_bytes = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            raw_peak = int(np.max(np.abs(audio)))

            # Sticky speech window: high peak resets countdown, stays open
            # for SPEECH_WINDOW_FRAMES after last speech frame (~1s).
            # Catches speech onset + model tail (OWW scores stay high
            # for several frames after the word ends).
            was_speech = speech_countdown > 0
            if raw_peak >= SPEECH_PEAK_MIN:
                if not was_speech:
                    # Speech just started — reset consecutive counter so
                    # stale 1.0 scores from silence don't carry over
                    consecutive.clear()
                speech_countdown = SPEECH_WINDOW_FRAMES
            elif speech_countdown > 0:
                speech_countdown -= 1
            has_speech = speech_countdown > 0

            rms = int(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            # Feed raw audio to OWW (matches training data levels).
            # Peak gate handles silence rejection — model handles speech discrimination.
            scores = detector.predict(audio)
            ts = time.strftime("%H:%M:%S")

            for name, score in scores.items():
                triggered = False

                if has_speech and score >= threshold:
                    consecutive[name] = consecutive.get(name, 0) + 1
                    if consecutive[name] >= trigger_frames:
                        triggered = True
                        print(f"  >>> {GREEN}DETECTED: {name}{RESET} "
                              f"(score={score:.3f}, rms={rms})")
                        log_event("wakeword_detected", model=name,
                                  score=round(score, 4), rms=rms)
                        detector.reset()
                        consecutive.clear()
                        speech_countdown = 0

                        # Full pipeline: beep -> record -> send -> play response
                        if pipeline:
                            # Close wakeword stream so record_audio can use the mic
                            stream.stop_stream()
                            stream.close()
                            # Drain the queue
                            while not audio_q.empty():
                                try:
                                    audio_q.get_nowait()
                                except queue.Empty:
                                    break

                            beep_start(p, spk_idx)
                            print(f"  {BLUE}Listening...{RESET}", end="", flush=True)
                            wav_bytes = record_audio(p, mic_idx)

                            if wav_bytes:
                                beep_end(p, spk_idx)
                                size_kb = len(wav_bytes) / 1024
                                print(f"\r  {GREEN}Recorded ({size_kb:.0f}KB){RESET}     ")
                                log_event("recorded", size_kb=round(size_kb, 1))

                                print(f"  {YELLOW}Processing...{RESET}", end="", flush=True)
                                try:
                                    result = send_audio(wav_bytes, args.server)
                                    print("\r", end="")
                                    display_result(result)
                                    log_result(result, mode="wakeword_pipeline")

                                    # Play audio response
                                    audio_b64 = result.get("audio_base64", "")
                                    if audio_b64:
                                        try:
                                            resp_audio = base64.b64decode(audio_b64)
                                            play_audio(p, spk_idx, resp_audio)
                                        except Exception as e:
                                            print(f"  {RED}Playback failed: {e}{RESET}")
                                            log_event("playback_error", error=str(e))

                                    # Follow-up
                                    if result.get("await_followup"):
                                        print(f"\n  {BLUE}Follow-up expected -- listening...{RESET}")
                                        beep_start(p, spk_idx)
                                        time.sleep(0.2)
                                        wav_bytes = record_audio(p, mic_idx)
                                        if wav_bytes:
                                            beep_end(p, spk_idx)
                                            print(f"  {YELLOW}Processing follow-up...{RESET}",
                                                  end="", flush=True)
                                            try:
                                                result = send_audio(wav_bytes, args.server)
                                                print("\r", end="")
                                                display_result(result)
                                                log_result(result, mode="followup")
                                                audio_b64 = result.get("audio_base64", "")
                                                if audio_b64:
                                                    resp_audio = base64.b64decode(audio_b64)
                                                    play_audio(p, spk_idx, resp_audio)
                                            except Exception as e:
                                                print(f"\r  {RED}Follow-up failed: {e}{RESET}")
                                                log_event("error", error=str(e))
                                except Exception as e:
                                    print(f"\r  {RED}Error: {e}{RESET}")
                                    log_event("error", error=str(e))
                                    beep_error(p, spk_idx)
                            else:
                                print(f"\r  {YELLOW}No speech detected{RESET}        ")
                                beep_end(p, spk_idx)
                                log_event("no_speech")

                            # Re-open wakeword stream
                            print(f"\n  {GREEN}Listening...{RESET}\n")
                            try:
                                stream = open_ww_stream()
                                # Brief warmup
                                warmup_end = time.time() + 0.5
                                while time.time() < warmup_end:
                                    try:
                                        chunk = audio_q.get(timeout=0.2)
                                        detector.predict(chunk)
                                    except queue.Empty:
                                        pass
                            except Exception as e:
                                print(f"  {RED}Failed to re-open mic: {e}{RESET}")
                                log_event("error", error=f"stream reopen: {e}")
                                break

                        print()
                else:
                    consecutive[name] = 0

                if score > 0.01 or has_speech:
                    speech_tag = "" if has_speech else f"  {YELLOW}[silence]{RESET}"
                    print(f"  [{ts}] {name}: {score:.4f}  peak={raw_peak}  rms={rms}  "
                          f"consec={consecutive.get(name, 0)}{speech_tag}")

                csv.write(f"{time.time():.3f},{name},{score:.4f},{raw_peak},{rms},"
                          f"{consecutive.get(name, 0)},{has_speech},{triggered}\n")
                csv.flush()

    except KeyboardInterrupt:
        print(f"\n  Scores saved: {score_log}")
        log_event("wakeword_live_end")
    finally:
        stream.close()
        p.terminate()
        csv.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Igor test client — interactive, WAV, text, and wake word modes")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--spk", type=int, default=None,
                        help="Speaker device index")
    parser.add_argument("--server", type=str, default=SERVER_URL,
                        help="Server URL")

    # Mode flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", type=str, metavar="QUERY",
                      help="Send text query (no audio)")
    mode.add_argument("--wav", type=str, metavar="FILE",
                      help="Send a pre-recorded WAV file")
    mode.add_argument("--wakeword", action="store_true",
                      help="Wake word detection testing")

    # Wake word options
    parser.add_argument("--wav-dir", type=str, metavar="DIR",
                        help="Batch score WAV files in directory (with --wakeword)")
    parser.add_argument("--ww-threshold", type=float, default=OWW_THRESHOLD,
                        help=f"Wake word threshold (default {OWW_THRESHOLD})")
    parser.add_argument("--ww-trigger", type=int, default=OWW_TRIGGER_FRAMES,
                        help=f"Consecutive frames to trigger (default {OWW_TRIGGER_FRAMES})")
    parser.add_argument("--ww-pipeline", action="store_true",
                        help="After wake word detection, record and send to server")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable Silero VAD pre-filter (for noisy PC mics)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.text:
        mode_text(args)
    elif args.wav:
        mode_wav(args)
    elif args.wakeword:
        mode_wakeword(args)
    else:
        mode_interactive(args)


if __name__ == "__main__":
    main()
