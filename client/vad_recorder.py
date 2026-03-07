"""RMS-based Voice Activity Detection and Recording Module.

Adaptive silence detection: the silence duration required to end recording
scales with how much speech has been captured.  Short utterances (< 1s of
speech, like "pause") end after 0.6s of silence.  Longer utterances ramp
up to the full SILENCE_END_DURATION to avoid cutting off mid-thought pauses.

Ambient noise calibration: before each recording, the noise floor is measured
from a few frames of room audio.  The speech/silence threshold is set relative
to this floor so that VAD works regardless of mic gain or room noise level.
"""
import logging
import numpy as np
import pyaudio
import wave
from enum import Enum
from client.config import (
    SAMPLE_RATE, MIN_RECORDING, MAX_RECORDING,
    SILENCE_END_DURATION, RMS_SILENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

# Adaptive silence: short commands end faster, long speech gets more patience.
# Speech under SHORT_SPEECH_THRESHOLD seconds uses SILENCE_SHORT.
# Speech over LONG_SPEECH_THRESHOLD seconds uses the full SILENCE_END_DURATION.
# In between, linearly interpolated.
SILENCE_SHORT = 0.6           # seconds of silence to end a short command
SHORT_SPEECH_THRESHOLD = 1.0  # speech duration (s) considered "short"
LONG_SPEECH_THRESHOLD = 3.0   # speech duration (s) where full patience kicks in

# Ambient noise calibration: multiplier applied to the measured noise floor
# to derive the effective speech/silence threshold.  1.8x means speech must be
# almost twice as loud as the room noise to count.  This handles varying mic
# gains and noisy environments automatically.
NOISE_FLOOR_MULTIPLIER = 1.8
CALIBRATION_FRAMES = 5        # frames to sample for noise floor (~400ms at 16kHz/1280)
# Hard ceiling on the effective threshold.  Even in noisy rooms or with beep
# echo contamination, the threshold must never exceed this.  Normal close-mic
# speech at 16kHz/16-bit is typically 2000-6000 RMS; 3000 catches all but
# whispers while preventing contaminated calibration from killing detection.
MAX_EFFECTIVE_THRESHOLD = 3000
# Safety timeout: max seconds to wait for speech even without an explicit
# timeout.  Prevents the VAD from hanging forever if calibration sets the
# threshold too high and speech is never detected.
WAITING_TIMEOUT = 10.0


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

        # Calibrate ambient noise floor from a few frames of room audio.
        # These frames also flush residual audio from the buffer.
        ambient_rms_samples = []
        for _ in range(CALIBRATION_FRAMES):
            cal_bytes = stream.read(self.chunk_size, exception_on_overflow=False)
            cal_audio = np.frombuffer(cal_bytes, dtype=np.int16)
            ambient_rms_samples.append(self.calculate_rms(cal_audio))

        ambient_rms = float(np.median(ambient_rms_samples))
        # Effective threshold: whichever is higher — the configured floor or
        # the measured ambient noise scaled up.  Capped at MAX_EFFECTIVE_THRESHOLD
        # to prevent contaminated calibration (beep echo, early speech) from
        # setting an impossibly high threshold that blocks all detection.
        effective_threshold = min(
            max(RMS_SILENCE_THRESHOLD, ambient_rms * NOISE_FLOOR_MULTIPLIER),
            MAX_EFFECTIVE_THRESHOLD,
        )
        logger.info(
            f"VAD calibration: ambient_rms={ambient_rms:.0f}, "
            f"effective_threshold={effective_threshold:.0f} "
            f"(config={RMS_SILENCE_THRESHOLD}, cap={MAX_EFFECTIVE_THRESHOLD})"
        )

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
        # Safety timeout: even without an explicit timeout, don't wait forever.
        # This catches the case where calibration sets the threshold too high.
        waiting_safety_frames = int(WAITING_TIMEOUT * frames_per_second)

        # Count frames where speech was detected (not just total frames recorded)
        speech_frames = 0

        try:
            while state != VADState.DONE:
                audio_bytes = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                rms = self.calculate_rms(audio)

                is_speech = rms > effective_threshold

                if state == VADState.WAITING_FOR_SPEECH:
                    waiting_frames += 1
                    # Check for initial timeout (e.g., follow-up mode)
                    if initial_timeout_frames and waiting_frames >= initial_timeout_frames:
                        state = VADState.DONE  # Timeout, no speech detected
                    elif waiting_frames >= waiting_safety_frames:
                        logger.warning(
                            f"VAD safety timeout: no speech detected in {WAITING_TIMEOUT}s "
                            f"(threshold={effective_threshold:.0f})"
                        )
                        state = VADState.DONE
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
