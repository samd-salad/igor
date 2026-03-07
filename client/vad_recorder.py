"""RMS-based Voice Activity Detection and Recording Module.

Adaptive silence detection: the silence duration required to end recording
scales with how much speech has been captured.  Short utterances (< 1s of
speech, like "pause") end after 0.6s of silence.  Longer utterances ramp
up to the full SILENCE_END_DURATION to avoid cutting off mid-thought pauses.
"""
import numpy as np
import pyaudio
import wave
from enum import Enum
from client.config import (
    SAMPLE_RATE, MIN_RECORDING, MAX_RECORDING,
    SILENCE_END_DURATION, RMS_SILENCE_THRESHOLD
)

# Adaptive silence: short commands end faster, long speech gets more patience.
# Speech under SHORT_SPEECH_THRESHOLD seconds uses SILENCE_SHORT.
# Speech over LONG_SPEECH_THRESHOLD seconds uses the full SILENCE_END_DURATION.
# In between, linearly interpolated.
SILENCE_SHORT = 0.6           # seconds of silence to end a short command
SHORT_SPEECH_THRESHOLD = 1.0  # speech duration (s) considered "short"
LONG_SPEECH_THRESHOLD = 3.0   # speech duration (s) where full patience kicks in


class VADState(Enum):
    WAITING_FOR_SPEECH = 1
    RECORDING = 2
    DONE = 3


class VADRecorder:
    """Records audio with RMS-based voice activity detection."""

    def __init__(self, pa_instance, device_index=None):
        self.pa = pa_instance
        self.device_index = device_index
        self.chunk_size = 1280  # Same as wake word detection (~80ms at 16kHz)
        self.sample_rate = SAMPLE_RATE

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate Root Mean Square of audio chunk."""
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    def record_with_vad(self, output_path: str, initial_timeout: float = None) -> bool:
        """
        Record audio with VAD state machine.

        Args:
            output_path: Path to save WAV file
            initial_timeout: Max seconds to wait for speech to begin (None = no limit)

        Returns True if speech was captured, False otherwise (including timeout).
        """
        stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index
        )

        # Flush initial frames to clear residual audio from buffers
        for _ in range(3):
            stream.read(self.chunk_size, exception_on_overflow=False)

        state = VADState.WAITING_FOR_SPEECH
        frames = []
        silence_frames = 0
        total_frames = 0
        waiting_frames = 0

        # Calculate frame counts
        frames_per_second = self.sample_rate / self.chunk_size
        min_frames = int(MIN_RECORDING * frames_per_second)
        max_frames = int(MAX_RECORDING * frames_per_second)
        silence_max_frames = int(SILENCE_END_DURATION * frames_per_second)
        silence_short_frames = int(SILENCE_SHORT * frames_per_second)
        short_speech_frames = int(SHORT_SPEECH_THRESHOLD * frames_per_second)
        long_speech_frames = int(LONG_SPEECH_THRESHOLD * frames_per_second)
        initial_timeout_frames = int(initial_timeout * frames_per_second) if initial_timeout else None

        # Count frames where speech was detected (not just total frames recorded)
        speech_frames = 0

        try:
            while state != VADState.DONE:
                audio_bytes = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                rms = self.calculate_rms(audio)

                is_speech = rms > RMS_SILENCE_THRESHOLD

                if state == VADState.WAITING_FOR_SPEECH:
                    waiting_frames += 1
                    # Check for initial timeout (e.g., follow-up mode)
                    if initial_timeout_frames and waiting_frames >= initial_timeout_frames:
                        state = VADState.DONE  # Timeout, no speech detected
                    elif is_speech:
                        state = VADState.RECORDING
                        frames.append(audio_bytes)
                        total_frames = 1
                        speech_frames = 1

                elif state == VADState.RECORDING:
                    frames.append(audio_bytes)
                    total_frames += 1

                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        # Adaptive silence threshold: short commands end fast,
                        # longer speech gets more patience for mid-thought pauses.
                        if speech_frames <= short_speech_frames:
                            needed = silence_short_frames
                        elif speech_frames >= long_speech_frames:
                            needed = silence_max_frames
                        else:
                            # Linear ramp between short and long
                            ratio = (speech_frames - short_speech_frames) / (long_speech_frames - short_speech_frames)
                            needed = int(silence_short_frames + ratio * (silence_max_frames - silence_short_frames))
                        if silence_frames >= needed and total_frames >= min_frames:
                            state = VADState.DONE

                    if total_frames >= max_frames:
                        state = VADState.DONE

        finally:
            stream.close()

        if frames:
            self._save_wav(output_path, frames)
            return True
        return False

    def _save_wav(self, path: str, frames: list):
        """Save frames as WAV file compatible with faster-whisper."""
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
