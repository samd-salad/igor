"""Wake word detector using Picovoice Porcupine.

.ppn keyword files must be downloaded from console.picovoice.ai for your
Pi's CPU (Cortex-A72 for Pi 4, Cortex-A53 for Pi 3).
Place them in porcupine_models/ and set paths in client/config.py.
"""
import numpy as np
from pathlib import Path
import pvporcupine


class WakeWordDetector:
    def __init__(self, access_key: str, keyword_paths: list[str],
                 sensitivities: list[float] | None = None):
        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities or [0.5] * len(keyword_paths),
        )
        # Derive clean names: "doctor-butts_en_raspberry-pi_v3_0_0" → "doctor-butts"
        self._keywords = [Path(p).stem.split('_en_')[0] for p in keyword_paths]
        self._buffer = np.array([], dtype=np.int16)

    def predict(self, audio: bytes | np.ndarray) -> dict[str, float]:
        """Feed a raw audio chunk; return {keyword: score} (1.0=detected, 0.0=not)."""
        if isinstance(audio, (bytes, bytearray)):
            audio = np.frombuffer(audio, dtype=np.int16)

        self._buffer = np.concatenate([self._buffer, audio])
        frame_len = self._porcupine.frame_length
        result = {kw: 0.0 for kw in self._keywords}

        while len(self._buffer) >= frame_len:
            frame = self._buffer[:frame_len]
            self._buffer = self._buffer[frame_len:]
            idx = self._porcupine.process(frame)
            if idx >= 0:
                result[self._keywords[idx]] = 1.0

        return result

    def reset(self):
        self._buffer = np.array([], dtype=np.int16)

    def delete(self):
        self._porcupine.delete()
