"""PC voice client — first-class endpoint running on the same machine as the server.

Usage:
    python -m client.pc_main

Requires the server to be running first (python -m server.main).
Registers as CLIENT_ID/ROOM_ID with the server, starts a callback server
for receiving audio/beeps/suppression, and enters the wake word detection loop.
"""
import base64
import io
import logging
import queue
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio
import requests

from client.pc_audio import PCAudio
from client.pc_config import (
    SERVER_URL, CLIENT_HOST, CLIENT_PORT, CLIENT_ID, ROOM_ID,
    SAMPLE_RATE, CHANNELS, CHUNK, OWW_MODELS_DIR,
    OWW_CHUNK, OWW_THRESHOLD, OWW_TRIGGER_FRAMES,
    NORMALIZE_TARGET_PEAK, NORMALIZE_FLOOR,
    SPEECH_PEAK_MIN, SPEECH_TRAIL_FRAMES, DIP_SCORE, MAX_GAP_FRAMES,
    SILENCE_THRESHOLD, SILENCE_DURATION, MIN_RECORDING, MAX_RECORDING,
    OWW_AUTO_SAVE_SAMPLES, OWW_MAX_AUTO_SAMPLES, OWW_SAMPLE_BUFFER_SECONDS,
    WAKE_SAMPLES_DIR, USE_SONOS_OUTPUT, TEMP_WAV,
    MAX_FOLLOWUP_DEPTH, FOLLOWUP_TIMEOUT,
)
from client.suppress import suppress, is_suppressed
from shared.utils import setup_logging

logger = logging.getLogger(__name__)


def normalize_audio(audio: np.ndarray, target_peak: int = NORMALIZE_TARGET_PEAK,
                    floor: int = NORMALIZE_FLOOR) -> np.ndarray:
    """Normalize audio to target peak amplitude for low-gain mics."""
    if isinstance(audio, (bytes, bytearray)):
        audio = np.frombuffer(audio, dtype=np.int16).copy()
    peak = int(np.max(np.abs(audio)))
    if peak < floor:
        return audio
    gain = min(target_peak / peak, 1000.0)
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def normalize_wav(wav_bytes: bytes, target_peak: int = NORMALIZE_TARGET_PEAK) -> bytes:
    """Normalize WAV audio to target peak amplitude."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).copy()
    peak = int(np.max(np.abs(audio)))
    if peak > 0:
        gain = min(target_peak / peak, 1000.0)
        audio = np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(audio.tobytes())
    return out.getvalue()


class PCClient:
    """First-class PC voice client with wake word detection and server integration."""

    def __init__(self):
        self.audio = PCAudio()
        self.server_url = SERVER_URL
        self.callback_url = f"http://{CLIENT_HOST}:{CLIENT_PORT}"
        self.detector = None
        self._audio_q = queue.Queue()
        self._consecutive_errors = 0

    def initialize(self) -> bool:
        """Initialize audio, wake word detector, and register with server."""
        # Load wake word models
        try:
            from openwakeword.model import Model
            onnx_files = list(OWW_MODELS_DIR.glob("*.onnx"))
            if not onnx_files:
                logger.error(f"No .onnx models found in {OWW_MODELS_DIR}")
                return False
            self.detector = Model(
                wakeword_models=[str(f) for f in onnx_files],
                inference_framework="onnx",
            )
            logger.info(f"Wake word models loaded: {[f.stem for f in onnx_files]}")
        except Exception as e:
            logger.error(f"Failed to load wake word models: {e}")
            return False

        # Ensure sample dirs exist
        WAKE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        (WAKE_SAMPLES_DIR.parent / "negative").mkdir(parents=True, exist_ok=True)

        # Start callback server
        from client.pc_server import start_callback_server
        start_callback_server(self.audio, CLIENT_HOST, CLIENT_PORT)
        time.sleep(0.5)  # Let server bind

        # Register with server
        if not self._register():
            logger.warning("Server registration failed — will retry on first interaction")

        return True

    def _register(self) -> bool:
        """Register this client with the server."""
        try:
            resp = requests.post(
                f"{self.server_url}/api/register",
                json={
                    "client_id": CLIENT_ID,
                    "room_id": ROOM_ID,
                    "client_type": "audio",
                    "callback_url": self.callback_url,
                },
                timeout=5,
            )
            if resp.status_code == 200:
                logger.info(f"Registered with server as {CLIENT_ID} (room={ROOM_ID})")
                return True
            else:
                logger.warning(f"Registration failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"Registration failed: {e}")
            return False

    def _send_audio(self, wav_bytes: bytes) -> dict:
        """Send recorded audio to the server for processing."""
        wav_bytes = normalize_wav(wav_bytes)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        resp = requests.post(
            f"{self.server_url}/api/process_interaction",
            json={
                "audio_base64": audio_b64,
                "wake_word": "igor",
                "prefer_sonos_output": USE_SONOS_OUTPUT,
                "client_id": CLIENT_ID,
                "room_id": ROOM_ID,
            },
            timeout=60,
        )
        return resp.json()

    def _record_until_silence(self) -> bytes:
        """Record audio until silence is detected. Returns WAV bytes."""
        frames = []
        silent_chunks = 0
        silent_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
        min_chunks = int(MIN_RECORDING * SAMPLE_RATE / CHUNK)
        max_chunks = int(MAX_RECORDING * SAMPLE_RATE / CHUNK)

        stream = self.audio.open_input_stream()
        try:
            for i in range(max_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                audio = np.frombuffer(data, dtype=np.int16)
                peak = int(np.max(np.abs(audio)))

                if peak < SILENCE_THRESHOLD and i >= min_chunks:
                    silent_chunks += 1
                    if silent_chunks >= silent_limit:
                        break
                else:
                    silent_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()

        # Pack into WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def _save_wake_sample(self, audio_buffer: list):
        """Save auto-detected wake word audio for training."""
        if not OWW_AUTO_SAVE_SAMPLES:
            return
        try:
            WAKE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filepath = WAKE_SAMPLES_DIR / f"pc_auto_{ts}.wav"
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(audio_buffer))
            logger.debug(f"Saved wake sample: {filepath.name}")

            # Rotate old samples
            auto_files = sorted(WAKE_SAMPLES_DIR.glob("pc_auto_*.wav"), key=lambda f: f.stat().st_mtime)
            excess = len(auto_files) - OWW_MAX_AUTO_SAMPLES
            if excess > 0:
                for f in auto_files[:excess]:
                    f.unlink()
        except Exception as e:
            logger.warning(f"Failed to save wake sample: {e}")

    def _process_interaction(self, is_followup: bool = False):
        """Record, send to server, play response. Returns True if follow-up requested."""
        if is_suppressed():
            logger.info("Suppressed — skipping interaction")
            return False

        if not is_followup:
            self.audio.beep_start()
            time.sleep(0.1)

        # Record
        wav_bytes = self._record_until_silence()
        size_kb = len(wav_bytes) // 1024
        logger.info(f"Recorded ({size_kb}KB)")
        self.audio.beep_end()

        # Send to server
        try:
            result = self._send_audio(wav_bytes)
            self._consecutive_errors = 0
        except Exception as e:
            logger.error(f"Server request failed: {e}")
            self.audio.beep_error()
            self._consecutive_errors += 1
            return False

        # Handle response
        transcription = result.get("transcription", "")
        response_text = result.get("response_text", "")
        audio_b64 = result.get("audio_base64", "")
        await_followup = result.get("await_followup", False)

        if transcription:
            logger.info(f"You: {transcription}")
        if response_text:
            logger.info(f"Igor: {response_text}")

        # Play audio response (if not routed to Sonos)
        if audio_b64:
            audio_data = base64.b64decode(audio_b64)
            self.audio.play_wav_bytes(audio_data)

        if result.get("error"):
            logger.error(f"Server error: {result['error']}")
            self.audio.beep_error()
            return False

        return await_followup

    def run(self):
        """Main wake word detection loop."""
        logger.info("Starting wake word detection...")

        # Audio callback → queue
        def _audio_callback(in_data, frame_count, time_info, status):
            self._audio_q.put(in_data)
            return (None, pyaudio.paContinue)

        stream = self.audio.open_input_stream(callback=_audio_callback)
        stream.start_stream()

        # Warmup — let OWW process a few frames
        logger.info("Warming up wake word detector...")
        warmup_end = time.time() + 1.5
        while time.time() < warmup_end:
            try:
                chunk = self._audio_q.get(timeout=0.2)
                self.detector.predict(normalize_audio(np.frombuffer(chunk, dtype=np.int16)))
            except queue.Empty:
                pass

        logger.info(f"Listening (threshold={OWW_THRESHOLD}, trigger={OWW_TRIGGER_FRAMES})")

        # Detection state
        consecutive = {}
        gap_frames = {}
        saw_dip = False
        frames_since_speech = SPEECH_TRAIL_FRAMES + 1
        audio_ring = []  # Ring buffer for wake word sample saving
        ring_size = int(OWW_SAMPLE_BUFFER_SECONDS * SAMPLE_RATE / OWW_CHUNK)

        try:
            while True:
                try:
                    audio_bytes = self._audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if is_suppressed():
                    continue

                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                raw_peak = int(np.max(np.abs(audio)))

                # Track speech energy
                if raw_peak >= SPEECH_PEAK_MIN:
                    frames_since_speech = 0
                else:
                    frames_since_speech += 1
                has_speech = frames_since_speech <= SPEECH_TRAIL_FRAMES

                # Maintain ring buffer for sample saving
                audio_ring.append(audio_bytes)
                if len(audio_ring) > ring_size:
                    audio_ring.pop(0)

                # Feed normalized audio to OWW
                norm_audio = normalize_audio(audio)
                scores = self.detector.predict(norm_audio)

                for name, score in scores.items():
                    # Dip requirement
                    if score < DIP_SCORE:
                        saw_dip = True

                    if score >= OWW_THRESHOLD and has_speech and saw_dip:
                        gap_frames[name] = 0
                        consecutive[name] = consecutive.get(name, 0) + 1

                        if consecutive[name] >= OWW_TRIGGER_FRAMES:
                            # TRIGGERED
                            logger.info(f"Wake word detected: {name} (score={score:.3f})")
                            self.detector.reset()
                            consecutive.clear()
                            gap_frames.clear()
                            saw_dip = False
                            frames_since_speech = SPEECH_TRAIL_FRAMES + 1

                            # Save wake word sample
                            self._save_wake_sample(list(audio_ring))

                            # Process interaction + follow-ups
                            followup = self._process_interaction()
                            depth = 0
                            while followup and depth < MAX_FOLLOWUP_DEPTH:
                                if is_suppressed():
                                    break
                                depth += 1
                                logger.info(f"Follow-up {depth}...")
                                self.audio.beep_start()
                                time.sleep(0.1)
                                followup = self._process_interaction(is_followup=True)

                            # Post-interaction warmup
                            warmup_end = time.time() + 1.0
                            while time.time() < warmup_end:
                                try:
                                    chunk = self._audio_q.get(timeout=0.2)
                                    self.detector.predict(normalize_audio(
                                        np.frombuffer(chunk, dtype=np.int16)
                                    ))
                                except queue.Empty:
                                    pass

                    elif score >= OWW_THRESHOLD and not has_speech:
                        gap_frames[name] = gap_frames.get(name, 0) + 1
                        if gap_frames.get(name, 0) > MAX_GAP_FRAMES:
                            consecutive[name] = 0
                            gap_frames[name] = 0
                    else:
                        consecutive[name] = 0
                        gap_frames[name] = 0

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()


def main():
    setup_logging("pc_client", log_file="data/pc_client.log")
    # Also add console handler to root so all loggers (including __main__) print
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        root.addHandler(ch)
    logger.info("Igor PC Client starting...")

    client = PCClient()
    if not client.initialize():
        logger.error("Failed to initialize PC client")
        sys.exit(1)

    client.run()


if __name__ == "__main__":
    main()
