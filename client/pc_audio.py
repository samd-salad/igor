"""Windows-compatible audio I/O using PyAudio.

Provides the same interface as client/audio.py (Pi ALSA) so the callback
server and main loop can work with either backend.
"""
import io
import logging
import math
import struct
import threading
import time
import wave
from typing import Optional

import numpy as np
import pyaudio

from client.pc_config import (
    SAMPLE_RATE, CHANNELS, CHUNK, FORMAT_BITS,
    AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE,
)

logger = logging.getLogger(__name__)

# PyAudio format constant
PA_FORMAT = pyaudio.paInt16


def _find_device(p: pyaudio.PyAudio, name_substr: str, is_input: bool) -> Optional[int]:
    """Find a device index by substring match on name."""
    if name_substr is None:
        return None
    name_lower = name_substr.lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        channels = info["maxInputChannels"] if is_input else info["maxOutputChannels"]
        if channels > 0 and name_lower in info["name"].lower():
            return i
    return None


class PCAudio:
    """Windows PyAudio wrapper matching the Audio interface."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._input_idx = _find_device(self._pa, AUDIO_INPUT_DEVICE, True)
        self._output_idx = _find_device(self._pa, AUDIO_OUTPUT_DEVICE, False)
        self._playback_lock = threading.Lock()
        logger.info(f"PCAudio: input={self._input_idx}, output={self._output_idx}")

    def open_input_stream(self, callback=None) -> pyaudio.Stream:
        """Open a microphone input stream."""
        kwargs = {
            "format": PA_FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "input": True,
            "frames_per_buffer": CHUNK,
        }
        if self._input_idx is not None:
            kwargs["input_device_index"] = self._input_idx
        if callback:
            kwargs["stream_callback"] = callback
        return self._pa.open(**kwargs)

    def play_wav_bytes(self, wav_bytes: bytes):
        """Play WAV audio bytes through the output device."""
        with self._playback_lock:
            try:
                buf = io.BytesIO(wav_bytes)
                with wave.open(buf, "rb") as wf:
                    stream = self._pa.open(
                        format=self._pa.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=self._output_idx,
                    )
                    data = wf.readframes(1024)
                    while data:
                        stream.write(data)
                        data = wf.readframes(1024)
                    stream.stop_stream()
                    stream.close()
            except Exception as e:
                logger.error(f"Playback failed: {e}")

    def _synth(self, freq_start: float, freq_end: float, duration: float, volume: float = 0.3):
        """Synthesize a frequency sweep (matching Pi sox 'sine start:end' behavior)."""
        try:
            n_samples = int(SAMPLE_RATE * duration)
            fade = int(SAMPLE_RATE * 0.005)
            samples = []
            for i in range(n_samples):
                t = i / n_samples  # 0..1 progress
                freq = freq_start + (freq_end - freq_start) * t
                # Fade envelope to avoid clicks
                env = 1.0
                if i < fade:
                    env = i / fade
                elif i > n_samples - fade:
                    env = (n_samples - i) / fade
                val = volume * env * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))
                samples.append(int(val * 32767))
            return struct.pack(f"<{len(samples)}h", *samples)
        except Exception:
            return b""

    def _play_raw(self, raw: bytes):
        """Play raw PCM samples."""
        try:
            stream = self._pa.open(
                format=PA_FORMAT, channels=1, rate=SAMPLE_RATE,
                output=True, output_device_index=self._output_idx,
            )
            stream.write(raw)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"Playback failed: {e}")

    def beep_start(self):
        """Rising sweep 500→900Hz: 'I'm listening' (matches Pi sox)."""
        self._play_raw(self._synth(500, 900, 0.12, 0.3))

    def beep_end(self):
        """Falling sweep 700→400Hz: 'Got it' (matches Pi sox)."""
        self._play_raw(self._synth(700, 400, 0.12, 0.25))

    def beep_error(self):
        """Low 200Hz buzz: 'Error' (matches Pi sox)."""
        self._play_raw(self._synth(200, 200, 0.3, 0.25))

    def beep_alert(self):
        """Triple ascending chime: 660→880→1100Hz (matches Pi sox)."""
        self._play_raw(
            self._synth(660, 660, 0.1, 0.35)
            + self._synth(0, 0, 0.08, 0)  # gap
            + self._synth(880, 880, 0.1, 0.35)
            + self._synth(0, 0, 0.08, 0)  # gap
            + self._synth(1100, 1100, 0.15, 0.4)
        )

    def terminate(self):
        """Clean up PyAudio."""
        self._pa.terminate()
