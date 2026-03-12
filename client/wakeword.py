"""Wake word detector using OpenWakeWord custom models.

Place trained .onnx files in oww_models/ (produced by train_wakeword.py).
Base models (melspectrogram + embedding) are downloaded automatically on first run.
"""
from typing import Dict, List, Union
import numpy as np
from pathlib import Path

from openwakeword.model import Model


class WakeWordDetector:
    def __init__(self, model_paths: List[str], threshold: float = 0.5,
                 verifier_path: str = None, verifier_threshold: float = 0.3):
        """
        Args:
            model_paths:        Paths to trained .onnx wakeword model files.
            threshold:          Detection score threshold (0–1). Higher = fewer false positives.
            verifier_path:      Path to a custom verifier .pkl (speaker-specific second stage).
            verifier_threshold: Base score must exceed this to trigger verifier.
        """
        if not model_paths:
            raise ValueError("At least one model path is required")

        self._threshold = threshold

        # Build verifier dict: maps each model stem to the verifier .pkl
        custom_verifier_models = {}
        if verifier_path:
            for p in model_paths:
                custom_verifier_models[Path(p).stem] = verifier_path

        self._model = Model(
            wakeword_models=model_paths,
            inference_framework="onnx",
            vad_threshold=0.5,  # Silero VAD pre-filter: non-speech audio never reaches classifier
            custom_verifier_models=custom_verifier_models or None,
            custom_verifier_threshold=verifier_threshold,
        )

        # Friendly names: model stem with underscores → spaces
        self._stems = {Path(p).stem: Path(p).stem.replace("_", " ") for p in model_paths}

    def predict(self, audio: Union[bytes, np.ndarray]) -> Dict[str, float]:
        """Feed one audio chunk (1280 samples / 80 ms); return {keyword: score} for each model."""
        if isinstance(audio, (bytes, bytearray)):
            audio = np.frombuffer(audio, dtype=np.int16)
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.int16)

        prediction = self._model.predict(audio)
        return {self._stems.get(stem, stem): float(score) for stem, score in prediction.items()}

    def reset(self):
        """Full OWW state reset — call after a detection.
        Clears prediction_buffer AND preprocessor state (melspectrogram, embeddings).
        Re-seeds feature_buffer with random noise; warmup on the next cycle displaces it.
        """
        self._model.reset()

    def delete(self):
        pass
