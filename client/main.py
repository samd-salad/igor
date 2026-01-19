#!/usr/bin/env python3
"""Main entry point for Raspberry Pi client."""
import sys
import logging
import time
import threading
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from client.config import (
    SERVER_URL,
    CLIENT_HOST,
    CLIENT_PORT,
    WAKE_WORDS,
    WAKE_THRESHOLD,
    MODELS_DIR,
    TEMP_WAV,
    REQUEST_TIMEOUT
)
from client.audio import Audio
from client.wakeword import WakeWordDetector
from client.pi_server import run_pi_server
from shared.protocol import PROCESS_INTERACTION_ENDPOINT
from shared.models import ProcessInteractionRequest
from shared.utils import setup_logging, read_wav_file, encode_audio_base64, decode_audio_base64, get_timestamp

# Configure logging
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

        # Initialize audio
        logger.info("Loading audio system...")
        if not self.audio.initialize():
            logger.error("Failed to initialize audio")
            return False

        # Initialize wake word detector
        logger.info(f"Loading wake word detection ({WAKE_WORDS})...")
        try:
            models = []
            for wake_word in WAKE_WORDS:
                model_path = MODELS_DIR / f"{wake_word}.onnx"
                if model_path.exists():
                    models.append(str(model_path))
                else:
                    logger.error(f"Wake word model not found: {model_path}")
                    return False

            self.wakeword_detector = WakeWordDetector(models, threshold=WAKE_THRESHOLD)
            logger.info("Wake word detection initialized")

        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            return False

        logger.info("All components initialized successfully")
        return True

    def run(self):
        """Main client loop."""
        # Start Pi HTTP server in background thread
        server_thread = threading.Thread(
            target=run_pi_server,
            args=(self.audio, CLIENT_HOST, CLIENT_PORT),
            daemon=True,
            name="PiServer"
        )
        server_thread.start()
        logger.info(f"Pi server started on {CLIENT_HOST}:{CLIENT_PORT}")

        # Give server a moment to start
        time.sleep(1)

        logger.info("Entering main loop. Listening for wake word...")
        logger.info(f"Server URL: {self.server_url}")

        try:
            while True:
                try:
                    self._handle_interaction()
                    self.consecutive_errors = 0  # Reset on success

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

    def _handle_interaction(self):
        """Handle a single voice interaction."""
        # Step 1: Listen for wake word
        logger.debug("Listening for wake word...")
        stream = self.audio.open_stream()

        try:
            wake_word = None
            while not wake_word:
                # Read audio chunk (1280 samples = 80ms at 16kHz)
                audio_bytes = stream.read(1280, exception_on_overflow=False)

                # Get predictions for all wake words
                predictions = self.wakeword_detector.predict(audio_bytes)

                # Check if any wake word exceeded threshold
                for name, score in predictions.items():
                    if score >= WAKE_THRESHOLD:
                        wake_word = name
                        break
        finally:
            stream.close()

        if not wake_word:
            logger.warning("Wake word detection failed")
            return

        logger.info(f"Wake word detected: {wake_word}")

        # Reset wake word detector buffers to prevent interference with recording
        self.wakeword_detector.reset()

        # Give audio system time to flush buffers
        time.sleep(0.2)

        # Step 2: Play start beep and record speech
        try:
            self.audio.beep_start()
        except Exception as e:
            logger.warning(f"Start beep failed: {e}")

        time.sleep(0.1)  # Small delay to let beep finish
        if not self.audio.record(TEMP_WAV):
            logger.error("Recording failed")
            self.audio.beep_error()
            return

        self.audio.beep_end()
        logger.info("Recording complete")

        # Step 3: Send to server for processing
        try:
            # Read recorded audio
            audio_bytes = read_wav_file(TEMP_WAV)
            audio_b64 = encode_audio_base64(audio_bytes)

            # Build request
            request = ProcessInteractionRequest(
                audio_base64=audio_b64,
                wake_word=wake_word,
                timestamp=get_timestamp()
            )

            # Send to server
            audio_size_kb = len(audio_bytes) / 1024
            logger.info(f"Sending audio to server ({audio_size_kb:.1f}KB)...")
            response = requests.post(
                f"{self.server_url}{PROCESS_INTERACTION_ENDPOINT}",
                json=request.model_dump(),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()

            # Log transcription
            logger.info(f"Transcription: {result['transcription']}")

            # Check for errors
            if result.get('error'):
                logger.error(f"Server reported error: {result['error']}")
                self.audio.beep_error()
                return

            # Step 4: Play response audio
            if result.get('audio_base64'):
                response_audio = decode_audio_base64(result['audio_base64'])
                if self.audio.play_audio_bytes(response_audio):
                    self.audio.beep_done()
                    logger.info("Interaction complete")
                else:
                    logger.error("Failed to play response audio")
                    self.audio.beep_error()
            else:
                logger.warning("No audio in response")
                self.audio.beep_error()

        except requests.Timeout:
            logger.error(f"Server request timed out after {REQUEST_TIMEOUT}s")
            self.audio.beep_error()
        except requests.ConnectionError:
            logger.error(f"Cannot connect to server at {self.server_url}")
            self.audio.beep_error()
        except requests.RequestException as e:
            logger.error(f"Server request failed: {e}")
            self.audio.beep_error()
        except Exception as e:
            logger.error(f"Unexpected error during interaction: {e}", exc_info=True)
            self.audio.beep_error()


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
