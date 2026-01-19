"""RMS-based Voice Activity Detection and Recording Module."""
import numpy as np
import pyaudio
import wave
from enum import Enum
from client.config import (
    SAMPLE_RATE, MIN_RECORDING, MAX_RECORDING,
    SILENCE_END_DURATION, RMS_SILENCE_THRESHOLD
)


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

    def record_with_vad(self, output_path: str) -> bool:
        """
        Record audio with VAD state machine.
        Returns True if speech was captured, False otherwise.
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

        # Calculate frame counts
        frames_per_second = self.sample_rate / self.chunk_size
        min_frames = int(MIN_RECORDING * frames_per_second)
        max_frames = int(MAX_RECORDING * frames_per_second)
        silence_threshold_frames = int(SILENCE_END_DURATION * frames_per_second)

        try:
            while state != VADState.DONE:
                audio_bytes = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                rms = self.calculate_rms(audio)

                is_speech = rms > RMS_SILENCE_THRESHOLD

                if state == VADState.WAITING_FOR_SPEECH:
                    if is_speech:
                        state = VADState.RECORDING
                        frames.append(audio_bytes)
                        total_frames = 1

                elif state == VADState.RECORDING:
                    frames.append(audio_bytes)
                    total_frames += 1

                    if not is_speech:
                        silence_frames += 1
                        if silence_frames >= silence_threshold_frames and total_frames >= min_frames:
                            state = VADState.DONE
                    else:
                        silence_frames = 0

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
