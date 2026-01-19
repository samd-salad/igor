"""Audio I/O handling for Raspberry Pi client."""
import logging
import os
import sys
import subprocess
from typing import Optional
from contextlib import contextmanager

# Suppress ALSA/JACK warnings during PyAudio import
_stderr_fd = sys.stderr.fileno()
_null_fd = os.open(os.devnull, os.O_RDWR)
_saved_stderr = os.dup(_stderr_fd)
os.dup2(_null_fd, _stderr_fd)

import pyaudio

# Restore stderr
os.dup2(_saved_stderr, _stderr_fd)
os.close(_null_fd)
os.close(_saved_stderr)

from client.config import SAMPLE_RATE, AUDIO_DEVICE, TEMP_WAV
from client.vad_recorder import VADRecorder

logger = logging.getLogger(__name__)


@contextmanager
def suppress_alsa_errors():
    """Suppress ALSA/JACK warnings during PyAudio operations."""
    stderr_fd = sys.stderr.fileno()
    null_fd = os.open(os.devnull, os.O_RDWR)
    saved = os.dup(stderr_fd)
    os.dup2(null_fd, stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(null_fd)
        os.close(saved)


class Audio:
    """Handles all audio input/output operations on Pi."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: str = AUDIO_DEVICE):
        self.sample_rate = sample_rate
        self.device = device
        self.pa: Optional[pyaudio.PyAudio] = None
        self.vad_recorder: Optional[VADRecorder] = None
        logger.info(f"Audio initialized (device: {device}, rate: {sample_rate})")

    def initialize(self) -> bool:
        """Initialize PyAudio and VAD recorder."""
        try:
            with suppress_alsa_errors():
                self.pa = pyaudio.PyAudio()

            dev_index = self._get_device_index()
            if dev_index is None:
                logger.error("No USB microphone found")
                return False

            self.vad_recorder = VADRecorder(self.pa, dev_index)
            logger.info("Audio system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Audio initialization failed: {e}")
            return False

    def cleanup(self):
        """Clean up audio resources."""
        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            self.pa = None

    def _get_device_index(self) -> Optional[int]:
        """Find USB microphone device index."""
        if not self.pa:
            return None

        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if "USB" in info["name"] and info["maxInputChannels"] > 0:
                logger.info(f"Found USB microphone: {info['name']} (index {i})")
                return i

        logger.warning("No USB microphone found")
        return None

    def open_stream(self):
        """Open audio input stream for wake word detection."""
        if not self.pa:
            raise RuntimeError("Audio not initialized")

        dev_index = self._get_device_index()
        return self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1280,
            input_device_index=dev_index
        )

    def record(self, output_path: str = TEMP_WAV) -> bool:
        """
        Record audio with VAD.

        Args:
            output_path: Path to save WAV file

        Returns:
            True if speech was captured, False otherwise
        """
        if not self.vad_recorder:
            logger.error("VAD recorder not initialized")
            return False

        try:
            return self.vad_recorder.record_with_vad(output_path)
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return False

    def play_audio_bytes(self, audio_bytes: bytes) -> bool:
        """
        Play audio from bytes (WAV format).

        Args:
            audio_bytes: WAV audio data

        Returns:
            True on success
        """
        try:
            # Write to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                # Play with aplay
                result = subprocess.run(
                    ['aplay', '-D', self.device, temp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30
                )
                return result.returncode == 0
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            logger.error("Audio playback timed out")
            return False
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False

    # Beep sounds using sox/play
    @staticmethod
    def beep_start():
        """Rising tone: 'I'm listening'"""
        try:
            subprocess.run(
                ['play', '-n', 'synth', '0.12', 'sine', '500:900', 'vol', '0.3'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception as e:
            logger.debug(f"Beep start failed: {e}")

    @staticmethod
    def beep_end():
        """Falling tone: 'Got it'"""
        try:
            subprocess.run(
                ['play', '-n', 'synth', '0.12', 'sine', '700:400', 'vol', '0.25'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception as e:
            logger.debug(f"Beep end failed: {e}")

    @staticmethod
    def beep_done():
        """Two quick chirps: 'Ready'"""
        try:
            subprocess.run(
                ['play', '-n', 'synth', '0.06', 'sine', '1200', 'vol', '0.2',
                 'pad', '0', '0.04', 'synth', '0.06', 'sine', '1200', 'vol', '0.2'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception as e:
            logger.debug(f"Beep done failed: {e}")

    @staticmethod
    def beep_error():
        """Low buzz: 'Error occurred'"""
        try:
            subprocess.run(
                ['play', '-n', 'synth', '0.3', 'sine', '200', 'vol', '0.25'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception as e:
            logger.debug(f"Beep error failed: {e}")

    @staticmethod
    def beep_alert():
        """Alert beep: triple ascending chime"""
        try:
            subprocess.run(
                ['play', '-n', 'synth', '0.1', 'sine', '660', 'vol', '0.35',
                 'pad', '0', '0.08', 'synth', '0.1', 'sine', '880', 'vol', '0.35',
                 'pad', '0', '0.08', 'synth', '0.15', 'sine', '1100', 'vol', '0.4'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1
            )
        except Exception as e:
            logger.debug(f"Beep alert failed: {e}")
