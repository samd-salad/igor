#!/usr/bin/env python3
"""Main entry point for Raspberry Pi client."""
import sys
import logging
import time
import threading
import wave
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from client.config import (
    SERVER_URL,
    CLIENT_HOST,
    CLIENT_PORT,
    SAMPLE_RATE,
    OWW_MODEL_PATHS,
    OWW_THRESHOLD,
    OWW_TRIGGER_FRAMES,
    OWW_AUTO_SAVE_SAMPLES,
    OWW_SAMPLE_BUFFER_SECONDS,
    WAKE_SAMPLES_DIR,
    TEMP_WAV,
    REQUEST_TIMEOUT,
    FOLLOWUP_TIMEOUT,
    USE_SONOS_OUTPUT,
)
from client.audio import Audio
from client.wakeword import WakeWordDetector
from client.pi_server import run_pi_server
from shared.protocol import PROCESS_INTERACTION_ENDPOINT
from shared.models import ProcessInteractionRequest
from shared.utils import setup_logging, read_wav_file, encode_audio_base64, decode_audio_base64, get_timestamp

logger = setup_logging('client', level=logging.INFO)


class PiClient:
    """Main Pi client for Dr. Butts voice assistant."""

    def __init__(self):
        self.audio = Audio()
        self.wakeword_detector: WakeWordDetector = None
        self.server_url = SERVER_URL
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

    def initialize(self) -> bool:
        """Initialize all client components."""
        logger.info("Initializing Dr. Butts Voice Assistant Client...")

        logger.info("Loading audio system...")
        if not self.audio.initialize():
            logger.error("Failed to initialize audio")
            return False

        logger.info("Loading wake word detection (OpenWakeWord)...")
        try:
            if not OWW_MODEL_PATHS:
                logger.error("No .onnx models found in oww_models/")
                logger.error("Train models with train_wakeword.py then copy .onnx files to oww_models/")
                return False

            self.wakeword_detector = WakeWordDetector(
                model_paths=OWW_MODEL_PATHS,
                threshold=OWW_THRESHOLD,
            )
            keywords = [Path(p).stem.replace("_", " ") for p in OWW_MODEL_PATHS]
            logger.info(f"Wake word detection initialized. Keywords: {keywords}")

        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            return False

        logger.info("All components initialized successfully")
        return True

    def run(self):
        """Main client loop."""
        server_thread = threading.Thread(
            target=run_pi_server,
            args=(self.audio, CLIENT_HOST, CLIENT_PORT),
            daemon=True,
            name="PiServer"
        )
        server_thread.start()
        logger.info(f"Pi server started on {CLIENT_HOST}:{CLIENT_PORT}")

        time.sleep(1)

        logger.info("Entering main loop. Listening for wake word...")
        logger.info(f"Server URL: {self.server_url}")

        try:
            while True:
                try:
                    self._handle_interaction()
                    self.consecutive_errors = 0

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.consecutive_errors += 1
                    logger.error(f"Interaction failed ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")

                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logger.error("Too many consecutive errors, restarting audio...")
                        self.audio.beep_error()
                        self.audio.cleanup()
                        time.sleep(2)
                        if not self.audio.initialize():
                            logger.error("Failed to restart audio, exiting")
                            break
                        self.consecutive_errors = 0

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.audio.cleanup()
            if self.wakeword_detector:
                self.wakeword_detector.delete()

    def _handle_interaction(self):
        """Handle a single voice interaction (may include follow-ups)."""
        logger.debug("Listening for wake word...")

        # Rolling buffer: keeps the last N chunks so we can save the detection audio
        chunk_bytes = 1280
        buffer_chunks = int(OWW_SAMPLE_BUFFER_SECONDS * SAMPLE_RATE * 2 / chunk_bytes) + 1
        audio_buffer = deque(maxlen=buffer_chunks) if OWW_AUTO_SAVE_SAMPLES else None

        stream = self.audio.open_stream()
        try:
            # Prime OWW's internal feature buffers with real audio so cold-start
            # doesn't cause false positives. We do NOT reset() afterward — the
            # detection loop's consecutive counter starts at 0 regardless, so
            # warmup scores can't accumulate toward a trigger.
            WARMUP_CHUNKS = 25  # ~2s at 80ms/frame, 16 kHz
            for _ in range(WARMUP_CHUNKS):
                self.wakeword_detector.predict(
                    stream.read(chunk_bytes, exception_on_overflow=False)
                )

            wake_word = None
            consecutive: dict[str, int] = {}
            frame_count = 0
            while not wake_word:
                audio_bytes = stream.read(chunk_bytes, exception_on_overflow=False)
                if audio_buffer is not None:
                    audio_buffer.append(audio_bytes)
                predictions = self.wakeword_detector.predict(audio_bytes)
                frame_count += 1
                for name, score in predictions.items():
                    if score > 0.1:
                        logger.debug(f"score[{name}]={score:.3f} (frame {frame_count})")
                    if score >= OWW_THRESHOLD:
                        consecutive[name] = consecutive.get(name, 0) + 1
                        if consecutive[name] >= OWW_TRIGGER_FRAMES:
                            wake_word = name
                            break
                    else:
                        consecutive[name] = 0
        finally:
            stream.close()

        if not wake_word:
            logger.warning("Wake word detection failed")
            return

        logger.info(f"Wake word detected: {wake_word}")
        self.wakeword_detector.reset()

        if audio_buffer:
            self._save_wake_sample(audio_buffer)
        time.sleep(0.2)

        try:
            self.audio.beep_start()
        except Exception as e:
            logger.warning(f"Start beep failed: {e}")

        time.sleep(0.1)
        self._process_and_respond(wake_word, is_followup=False)

    def _save_wake_sample(self, audio_buffer: deque):
        """Save the buffered detection audio as a positive training sample."""
        try:
            WAKE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            existing = len(list(WAKE_SAMPLES_DIR.glob("auto_*.wav")))
            filepath = WAKE_SAMPLES_DIR / f"auto_{existing:04d}.wav"
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(audio_buffer))
            logger.debug(f"Saved wake sample: {filepath.name}")
        except Exception as e:
            logger.warning(f"Failed to save wake sample: {e}")

    MAX_FOLLOWUP_DEPTH = 5

    def _process_and_respond(self, wake_word: str, is_followup: bool = False):
        """Record audio, send to server, play response, and handle follow-ups iteratively."""
        depth = 0
        while True:
            if not self.audio.record(TEMP_WAV, timeout=FOLLOWUP_TIMEOUT if is_followup else None):
                if is_followup:
                    logger.info("Follow-up timed out, ending conversation")
                    self.audio.beep_done()
                    return
                logger.error("Recording failed")
                self.audio.beep_error()
                return

            self.audio.beep_end()
            logger.info("Recording complete")

            try:
                audio_bytes = read_wav_file(TEMP_WAV)
                audio_b64 = encode_audio_base64(audio_bytes)

                request = ProcessInteractionRequest(
                    audio_base64=audio_b64,
                    wake_word=wake_word,
                    timestamp=get_timestamp(),
                    prefer_sonos_output=USE_SONOS_OUTPUT,
                )

                audio_size_kb = len(audio_bytes) / 1024
                logger.info(f"Sending audio to server ({audio_size_kb:.1f}KB)...")
                response = requests.post(
                    f"{self.server_url}{PROCESS_INTERACTION_ENDPOINT}",
                    json=request.model_dump(),
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()

                try:
                    result = response.json()
                except Exception:
                    logger.error("Server returned invalid JSON response")
                    self.audio.beep_error()
                    return

                transcription = result.get('transcription', '')
                logger.info(f"Transcription: '{transcription[:100]}...'" if len(transcription) > 100 else f"Transcription: '{transcription}'")

                if result.get('error'):
                    logger.error(f"Server reported error: {result['error']}")
                    self.audio.beep_error()
                    return

                if result.get('tts_routed'):
                    # TTS playing on Sonos — no local audio playback
                    logger.info("TTS routed to Sonos")
                    if result.get('await_followup') and depth < self.MAX_FOLLOWUP_DEPTH:
                        logger.info("Bot is awaiting follow-up response (Sonos routed)")
                        time.sleep(3.0)  # Approximate wait for Sonos playback to finish
                        self.audio.beep_start()
                        time.sleep(0.1)
                        depth += 1
                        is_followup = True
                        continue
                    else:
                        self.audio.beep_done()
                        logger.info("Interaction complete")
                        return
                elif result.get('audio_base64'):
                    response_audio = decode_audio_base64(result['audio_base64'])
                    if not self.audio.play_audio_bytes(response_audio):
                        logger.error("Failed to play response audio")
                        self.audio.beep_error()
                        return

                    if result.get('await_followup') and depth < self.MAX_FOLLOWUP_DEPTH:
                        logger.info("Bot is awaiting follow-up response")
                        time.sleep(0.3)
                        self.audio.beep_start()
                        time.sleep(0.1)
                        depth += 1
                        is_followup = True
                        continue
                    else:
                        self.audio.beep_done()
                        logger.info("Interaction complete")
                        return
                else:
                    logger.warning("No audio in response")
                    self.audio.beep_error()
                    return

            except requests.Timeout:
                logger.error(f"Server request timed out after {REQUEST_TIMEOUT}s")
                self.audio.beep_error()
                return
            except requests.ConnectionError:
                logger.error(f"Cannot connect to server at {self.server_url}")
                self.audio.beep_error()
                return
            except requests.RequestException as e:
                logger.error(f"Server request failed: {e}")
                self.audio.beep_error()
                return
            except Exception as e:
                logger.error(f"Unexpected error during interaction: {e}", exc_info=True)
                self.audio.beep_error()
                return


def main():
    """Main entry point."""
    client = PiClient()

    if not client.initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    logger.info("Initialization complete")
    logger.info("Ready to receive voice commands")

    client.run()


if __name__ == "__main__":
    main()
