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

    def beep(self, frequency: int = 880, duration: float = 0.15, volume: float = 0.3):
        """Play a simple sine wave beep."""
        try:
            n_samples = int(SAMPLE_RATE * duration)
            samples = []
            for i in range(n_samples):
                t = i / SAMPLE_RATE
                # Apply fade in/out to avoid click
                env = 1.0
                fade = int(SAMPLE_RATE * 0.01)
                if i < fade:
                    env = i / fade
                elif i > n_samples - fade:
                    env = (n_samples - i) / fade
                val = volume * env * math.sin(2 * math.pi * frequency * t)
                samples.append(int(val * 32767))

            raw = struct.pack(f"<{len(samples)}h", *samples)
            stream = self._pa.open(
                format=PA_FORMAT, channels=1, rate=SAMPLE_RATE,
                output=True, output_device_index=self._output_idx,
            )
            stream.write(raw)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"Beep failed: {e}")

    def beep_start(self):
        """Ascending beep — listening started."""
        self.beep(frequency=880, duration=0.12, volume=0.25)

    def beep_end(self):
        """Descending beep — recording finished."""
        self.beep(frequency=440, duration=0.12, volume=0.2)

    def beep_error(self):
        """Low buzzy beep — error occurred."""
        self.beep(frequency=220, duration=0.3, volume=0.2)

    def beep_alert(self):
        """Timer alert beep."""
        for freq in [880, 1100, 880]:
            self.beep(frequency=freq, duration=0.15, volume=0.3)
            time.sleep(0.05)

    def terminate(self):
        """Clean up PyAudio."""
        self._pa.terminate()
