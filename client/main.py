#!/usr/bin/env python3
"""Main entry point for Raspberry Pi client.

Responsibilities:
  1. Run Flask callback server in background thread (receives audio/beep/hardware
     commands from the PC server).
  2. Listen for wake word using OpenWakeWord.
  3. On wake word: record voice, send to PC server, play back response.
  4. Handle follow-up turns without re-detecting wake word.
  5. Route beeps and TTS through Sonos when USE_SONOS_OUTPUT=True.

Wake word → response flow:
  1. Stream mic audio through OpenWakeWord 80ms at a time.
  2. If a model scores ≥ OWW_THRESHOLD for OWW_TRIGGER_FRAMES consecutive frames
     → wake word confirmed.
  3. RMS energy check: reject low-energy detections (TV audio, distant mic).
  4. _beep("start") → Sonos or local beep.
  5. VAD records until silence → WAV saved to TEMP_WAV.
  6. _beep("end") signals recording complete.
  7. POST audio to server /api/process_interaction.
  8. If tts_routed=True: server played on Sonos; sleep tts_dur + 3.5s for startup lag.
  9. If audio_base64: play locally via PyAudio.
  10. If await_followup: stay in loop without re-triggering wake word.

Error handling:
  - consecutive_errors counter restarts audio subsystem after 5 failures.
  - _beep("error") called on all failure paths (respects USE_SONOS_OUTPUT routing).
"""
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
    CLIENT_ID,
    ROOM_ID,
    SAMPLE_RATE,
    OWW_MODEL_PATHS,
    OWW_THRESHOLD,
    OWW_VERIFIER_PATH,
    OWW_VERIFIER_THRESHOLD,
    OWW_TRIGGER_FRAMES,
    OWW_MIN_RMS,
    OWW_AUTO_SAVE_SAMPLES,
    OWW_MAX_AUTO_SAMPLES,
    OWW_SAMPLE_BUFFER_SECONDS,
    WAKE_SAMPLES_DIR,
    TEMP_WAV,
    REQUEST_TIMEOUT,
    FOLLOWUP_TIMEOUT,
    USE_SONOS_OUTPUT,
    INDICATOR_LIGHT,
)
from client.audio import Audio
from client.wakeword import WakeWordDetector
from client.pi_server import run_pi_server
from client.suppress import is_suppressed
from shared.protocol import PROCESS_INTERACTION_ENDPOINT, SONOS_BEEP_ENDPOINT, REGISTER_CLIENT_ENDPOINT
from shared.models import ProcessInteractionRequest
from shared.utils import setup_logging, read_wav_file, encode_audio_base64, decode_audio_base64, get_timestamp

logger = setup_logging('client', level=logging.INFO)


class PiClient:
    """Main Pi client for Igor voice assistant."""

    def __init__(self):
        self.audio = Audio()
        self.wakeword_detector: WakeWordDetector = None
        self.server_url = SERVER_URL
        # Consecutive error counter — after max_consecutive_errors, audio is restarted
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

    def initialize(self) -> bool:
        """Initialize audio and wake word detection subsystems.

        Returns:
            True if all components ready, False if any critical init failed.
        """
        logger.info("Initializing Igor Voice Assistant Client...")

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
                verifier_path=str(OWW_VERIFIER_PATH) if OWW_VERIFIER_PATH else None,
                verifier_threshold=OWW_VERIFIER_THRESHOLD,
            )
            keywords = [Path(p).stem.replace("_", " ") for p in OWW_MODEL_PATHS]
            logger.info(f"Wake word detection initialized. Keywords: {keywords}")

        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            return False

        logger.info("All components initialized successfully")
        return True

    def run(self):
        """Main client loop.

        Starts the Flask Pi callback server in a background daemon thread, then
        runs the wake word detection → interaction loop forever.  Audio subsystem
        is restarted after 5 consecutive failures to recover from PyAudio hangs.
        """
        # Flask Pi server — receives play_audio, hardware_control, play_beep,
        # suppress_wakeword callbacks from the PC server.
        server_thread = threading.Thread(
            target=run_pi_server,
            args=(self.audio, CLIENT_HOST, CLIENT_PORT),
            daemon=True,
            name="PiServer"
        )
        server_thread.start()
        logger.info(f"Pi server started on {CLIENT_HOST}:{CLIENT_PORT}")

        # Brief pause to let Flask bind its port before we start sending beeps
        time.sleep(1)

        # Register with server for multi-client routing
        self._register_with_server()

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
                        # Audio subsystem may be hung (e.g. ALSA buffer overrun).
                        # Restart it to recover without rebooting the Pi.
                        logger.error("Too many consecutive errors, restarting audio...")
                        self._beep("error")
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

    def _register_with_server(self):
        """Register this client with the server for multi-client routing.

        Non-critical — if registration fails, the server falls back to
        legacy ALLOWED_CLIENT_IPS behavior.
        """
        callback_url = f"http://{CLIENT_HOST}:{CLIENT_PORT}"
        try:
            resp = requests.post(
                f"{self.server_url}{REGISTER_CLIENT_ENDPOINT}",
                json={
                    "client_id": CLIENT_ID,
                    "room_id": ROOM_ID,
                    "client_type": "audio",
                    "callback_url": callback_url,
                },
                timeout=5,
            )
            if resp.ok:
                logger.info(f"Registered with server as '{CLIENT_ID}' in room '{ROOM_ID}'")
            else:
                logger.warning(f"Server registration returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Failed to register with server (non-critical): {e}")

    def _sonos_beep(self, beep_type: str):
        """Fire-and-forget beep via server Sonos endpoint.

        Sends a POST to /api/sonos_beep on the PC server, which plays the beep
        through the Sonos soundbar (or flashes the indicator light if TV is playing).
        Runs in a daemon thread so it never blocks the interaction loop.

        Args:
            beep_type: "start", "end", "error", or "alert".
        """
        def _send():
            try:
                requests.post(
                    f"{self.server_url}{SONOS_BEEP_ENDPOINT}",
                    json={"beep_type": beep_type, "indicator_light": INDICATOR_LIGHT},
                    timeout=3,
                )
            except Exception:
                pass  # Beep failure is non-critical; never raise
        threading.Thread(target=_send, daemon=True).start()

    def _beep(self, beep_type: str):
        """Play beep locally or via Sonos depending on output mode.

        When USE_SONOS_OUTPUT=True all audio goes through the Sonos soundbar
        (including beeps), so local PyAudio beeps are bypassed entirely.

        Beep semantics:
          'start'  — listening, about to record
          'end'    — recording stopped, sending to server
          'error'  — something failed (routes via this method so Sonos mode works)
          (no 'done' — the done chime is embedded in the TTS WAV itself)

        Args:
            beep_type: One of 'start', 'end', 'error', 'alert'.
        """
        if USE_SONOS_OUTPUT:
            self._sonos_beep(beep_type)
        else:
            # Calls e.g. self.audio.beep_start(), self.audio.beep_error()
            getattr(self.audio, f"beep_{beep_type}")()

    def _handle_interaction(self):
        """Handle a single detection cycle: listen → wake word → record → respond.

        Called in a tight loop from run().  Returns immediately if suppressed
        (e.g. within 20s after a TV command) or if wake word is not detected.

        Wake word detection details:
          - Mic audio is read in 1280-byte chunks (~80ms at 16kHz 16-bit mono).
          - Each chunk is fed to OWW; per-model confidence scores come back.
          - A model must exceed OWW_THRESHOLD for OWW_TRIGGER_FRAMES *consecutive*
            frames to confirm (avoids single-frame false positives).
          - RMS energy filter: if the buffered audio is too quiet, it's likely
            TV audio or distant room sound rather than close-mic speech.
          - The rolling audio buffer (OWW_SAMPLE_BUFFER_SECONDS) captures the
            wake word audio for saving as a training sample (auto-labeling).
        """
        # Skip entire cycle if wake word is currently suppressed
        if is_suppressed():
            logger.debug("Wake word suppressed — skipping detection cycle")
            time.sleep(0.1)
            return

        logger.debug("Listening for wake word...")

        # Rolling buffer: keeps the last N chunks so we can save the detection audio.
        # Only allocated if OWW_AUTO_SAVE_SAMPLES is enabled (saves disk I/O otherwise).
        chunk_bytes = 1280  # 80ms at 16kHz, 16-bit mono
        buffer_chunks = int(OWW_SAMPLE_BUFFER_SECONDS * SAMPLE_RATE * 2 / chunk_bytes) + 1
        audio_buffer = deque(maxlen=buffer_chunks) if OWW_AUTO_SAVE_SAMPLES else None

        stream = self.audio.open_stream()
        try:
            # Warmup: feed ~2s of real audio through OWW's feature extractor before
            # starting the detection loop.  OWW's mel-spectrogram + embedding buffers
            # need a few seconds to stabilize — without warmup the model starts with
            # stale/zero features and can generate false positives on the first frames.
            # We do NOT reset() afterward; the consecutive counter starts at 0 regardless
            # so warmup scores can't accumulate toward a trigger.
            #
            # Suppression check inside warmup: a TV command's suppress_wakeword RPC
            # can arrive while we're warming up.  Without this check, we'd warm up
            # for 2s, start detecting, and immediately trigger on TV startup audio.
            WARMUP_CHUNKS = 25  # ~2s at 80ms/frame, 16 kHz
            for _ in range(WARMUP_CHUNKS):
                if is_suppressed():
                    logger.info("Wake word suppressed during warmup — aborting")
                    self.wakeword_detector.reset()
                    return
                self.wakeword_detector.predict(
                    stream.read(chunk_bytes, exception_on_overflow=False)
                )

            wake_word = None
            consecutive: dict[str, int] = {}  # Per-model consecutive frame counter
            gap_frames: dict[str, int] = {}   # Frames since last above-threshold frame
            MAX_GAP = 3  # Allow brief dips (glottal stops, syllable boundaries)
            saw_dip = False  # Score must drop below DIP_SCORE before counting
            DIP_SCORE = 0.3  # Proves new audio entered OWW buffer (not stale silence scores)
            while not wake_word:
                audio_bytes = stream.read(chunk_bytes, exception_on_overflow=False)

                # Check mid-detection suppression (e.g. TV command triggered by another client)
                if is_suppressed():
                    logger.info("Wake word suppressed mid-detection — resetting")
                    self.wakeword_detector.reset()
                    return

                if audio_buffer is not None:
                    audio_buffer.append(audio_bytes)

                predictions = self.wakeword_detector.predict(audio_bytes)
                for name, score in predictions.items():
                    # Dip requirement: score must drop below 0.3 at least once
                    # before we start counting. Prevents stale 1.0 scores from
                    # OWW's internal buffer (e.g. after reset() seeds with noise)
                    # from immediately triggering.
                    if score < DIP_SCORE:
                        saw_dip = True

                    if score >= OWW_THRESHOLD and saw_dip:
                        consecutive[name] = consecutive.get(name, 0) + 1
                        gap_frames[name] = 0
                        if consecutive[name] >= OWW_TRIGGER_FRAMES:
                            # Required consecutive frames reached → confirmed detection
                            wake_word = name
                            break
                    elif score < OWW_THRESHOLD:
                        # Allow brief gaps (natural speech has glottal stops,
                        # syllable boundaries that can dip below threshold)
                        gap_frames[name] = gap_frames.get(name, 0) + 1
                        if gap_frames[name] > MAX_GAP:
                            consecutive[name] = 0
                            gap_frames[name] = 0

        finally:
            stream.close()

        if not wake_word:
            logger.warning("Wake word detection failed")
            return

        # RMS energy check: reject low-energy detections.
        # TV audio or room reflections can trigger OWW at low energy.
        # Close-mic speech is significantly louder — threshold set in config.
        if OWW_MIN_RMS > 0 and audio_buffer:
            import numpy as np
            buf_audio = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
            rms = int(np.sqrt(np.mean(buf_audio.astype(np.float32) ** 2)))
            if rms < OWW_MIN_RMS:
                logger.info(f"Wake word rejected — RMS {rms} below minimum {OWW_MIN_RMS} (likely room audio)")
                self.wakeword_detector.reset()
                return

        logger.info(f"Wake word detected: {wake_word}")
        self.wakeword_detector.reset()  # Clear OWW's internal state for next detection

        # Final suppression check after detection confirmed — catches the case where
        # a TV command's suppress RPC arrived between the detection loop exit and here.
        # Without this, TV startup audio can trigger a detection, pass the RMS check,
        # and start an interaction before the suppression window takes effect.
        if is_suppressed():
            logger.info("Wake word suppressed after detection — discarding")
            return

        if audio_buffer:
            # Save buffered audio as a positive training sample for OWW retraining
            self._save_wake_sample(audio_buffer)

        # Brief pause between wake word detection and start beep
        time.sleep(0.2)

        try:
            self._beep("start")
        except Exception as e:
            logger.warning(f"Start beep failed: {e}")

        # Wait for beep echo to decay before recording.  The Sonos soundbar is
        # in the same room as the Pi's USB mic — without enough gap, the beep's
        # acoustic reflection contaminates the first ~100ms of the recording and
        # Whisper produces garbage for the first word.  0.4s is enough for room
        # reflections to fall below the noise floor while still feeling responsive.
        # The VAD recorder also flushes 3 frames on open (vad_recorder.py:50-52)
        # which handles any remaining tail.
        time.sleep(0.4)
        self._process_and_respond(wake_word, is_followup=False)

    def _save_wake_sample(self, audio_buffer: deque):
        """Save buffered detection audio as a positive training sample.

        Auto-saved samples are named auto_NNNN.wav in WAKE_SAMPLES_DIR.
        Used for incremental OWW model retraining to improve detection accuracy.
        Failures are non-critical — sample saving never blocks the interaction.
        """
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
            # Rotate: keep only the newest OWW_MAX_AUTO_SAMPLES auto-saved files
            auto_files = sorted(WAKE_SAMPLES_DIR.glob("auto_*.wav"), key=lambda f: f.stat().st_mtime)
            excess = len(auto_files) - OWW_MAX_AUTO_SAMPLES
            if excess > 0:
                for f in auto_files[:excess]:
                    f.unlink()
                logger.debug(f"Rotated {excess} old auto-saved wake word samples")
        except Exception as e:
            logger.warning(f"Failed to save wake sample: {e}")

    # Maximum follow-up depth to prevent infinite conversation loops
    MAX_FOLLOWUP_DEPTH = 5

    def _process_and_respond(self, wake_word: str, is_followup: bool = False):
        """Record audio, send to server, play response, and handle follow-ups.

        Iterative (not recursive) follow-up loop — depth tracked by counter.
        Each iteration:
          1. Record audio (VAD stops on silence, or FOLLOWUP_TIMEOUT for follow-ups).
          2. POST to /api/process_interaction with audio + wake_word + prefer_sonos.
          3. Route response:
             a. tts_routed=True  → Sonos played it; sleep for duration + 3.5s.
             b. audio_base64     → decode and play locally via PyAudio.
          4. After audio ends: time-based suppression expires naturally.
          5. If await_followup=True and under depth limit: play start beep, stay in loop.

        Follow-up beep: both Sonos and local paths play a start beep via _beep()
        before re-recording.  _beep() routes through the configured output mode
        (local sox, Sonos play_uri, or LIFX flash).

        The 3.5s Sonos buffer covers TRANSITIONING→PLAYING startup lag (1-3s on
        Ray/Beam).  Without it, the follow-up mic opens before speech finishes.

        Args:
            wake_word: Detected wake word name (passed to server for context).
            is_followup: True when continuing a conversation without re-triggering.
        """
        depth = 0
        while True:
            # Record audio (VAD-gated); follow-up uses shorter timeout
            if not self.audio.record(TEMP_WAV, timeout=FOLLOWUP_TIMEOUT if is_followup else None):
                if is_followup:
                    logger.info("Follow-up timed out, ending conversation")
                    self._beep("end")  # Signal: mic closed, conversation over
                    return
                logger.error("Recording failed")
                self._beep("error")
                return

            self._beep("end")  # Signal: processing started
            logger.info("Recording complete")

            try:
                # Read WAV and encode to base64 for JSON transport
                audio_bytes = read_wav_file(TEMP_WAV)
                audio_b64 = encode_audio_base64(audio_bytes)

                request = ProcessInteractionRequest(
                    audio_base64=audio_b64,
                    wake_word=wake_word,
                    timestamp=get_timestamp(),
                    prefer_sonos_output=USE_SONOS_OUTPUT,
                    client_id=CLIENT_ID,
                    room_id=ROOM_ID,
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
                    self._beep("error")
                    return

                transcription = result.get('transcription', '')
                log_t = transcription[:100] + ('...' if len(transcription) > 100 else '')
                logger.info(f"Transcription: '{log_t}'")

                if result.get('error'):
                    logger.error(f"Server reported error: {result['error']}")
                    self._beep("error")
                    return

                # Quality gate rejection during TV playback: server returns empty
                # response with no error (silence is correct — don't interrupt TV).
                # Non-TV rejections now return "Didn't catch that." as audio, so
                # this branch only fires for TV-silent rejections.
                if not result.get('response_text') and not result.get('audio_base64') and not result.get('tts_routed'):
                    logger.info("Server returned empty response (quality gate rejection, TV silent)")
                    return

                if result.get('tts_routed'):
                    # TTS is playing on Sonos — no local audio needed.
                    # Sleep for TTS duration + Sonos startup lag before proceeding.
                    logger.info("TTS routed to Sonos")
                    tts_dur = result.get('tts_duration_seconds') or 5.0
                    # +3.5s: Sonos startup latency (TRANSITIONING→PLAYING) means
                    # audio hasn't started yet when server responds.  Sonos Ray/Beam
                    # can take 1-3s to start; 3.5s margin ensures we don't proceed
                    # until speech has fully finished playing.
                    time.sleep(tts_dur + 3.5)
                    # Don't unsuppress() here — TV commands set a 20s window that
                    # must run its full duration.  Time-based expiry handles it.

                    if result.get('await_followup') and depth < self.MAX_FOLLOWUP_DEPTH:
                        logger.info("Bot is awaiting follow-up response (Sonos routed)")
                        self._beep("start")  # Signal: listening for follow-up
                        time.sleep(0.4)       # Echo decay before recording
                        depth += 1
                        is_followup = True
                        continue
                    else:
                        logger.info("Interaction complete")
                        return

                elif result.get('audio_base64'):
                    # Server returned audio bytes — play locally via PyAudio
                    response_audio = decode_audio_base64(result['audio_base64'])
                    if not self.audio.play_audio_bytes(response_audio):
                        logger.error("Failed to play response audio")
                        self._beep("error")
                        return

                    # Don't unsuppress() here — let time-based suppression expire naturally.
                    # TV commands need the full 20s window for startup audio.

                    if result.get('await_followup') and depth < self.MAX_FOLLOWUP_DEPTH:
                        # Follow-up: stay in loop, play start beep to signal "listening"
                        logger.info("Bot is awaiting follow-up response")
                        time.sleep(0.3)
                        self._beep("start")
                        time.sleep(0.4)  # Echo decay before recording
                        depth += 1
                        is_followup = True
                        continue
                    else:
                        logger.info("Interaction complete")
                        return
                else:
                    logger.warning("No audio in response")
                    self._beep("error")
                    return

            except requests.Timeout:
                logger.error(f"Server request timed out after {REQUEST_TIMEOUT}s")
                self._beep("error")
                return
            except requests.ConnectionError:
                logger.error(f"Cannot connect to server at {self.server_url}")
                self._beep("error")
                return
            except requests.RequestException as e:
                logger.error(f"Server request failed: {e}")
                self._beep("error")
                return
            except Exception as e:
                logger.error(f"Unexpected error during interaction: {e}", exc_info=True)
                self._beep("error")
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
