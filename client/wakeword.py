"""Wake word detector using Sherpa-ONNX keyword spotting.

Download model from https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
Recommended: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
Place extracted files in sherpa_onnx_models/.
"""
import numpy as np
from pathlib import Path
import sherpa_onnx


class WakeWordDetector:
    def __init__(self, model_dir: str, keywords: list[str], threshold: float = 0.25):
        model_dir = Path(model_dir)

        config = sherpa_onnx.KeywordSpotterConfig(
            feat_config=sherpa_onnx.FeatureExtractorConfig(
                sample_rate=16000,
                feature_dim=80,
            ),
            model_config=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=self._find(model_dir, "encoder"),
                    decoder=self._find(model_dir, "decoder"),
                    joiner=self._find(model_dir, "joiner"),
                ),
                tokens=str(model_dir / "tokens.txt"),
                model_type="transducer",
            ),
            max_active_paths=4,
            keywords_score=1.0,
            keywords_threshold=threshold,
            num_trailing_blanks=1,
        )

        self._spotter = sherpa_onnx.KeywordSpotter(config)
        self._keywords = keywords
        self._keywords_str = "\n".join(keywords) + "\n"
        self._stream = self._spotter.create_stream(keywords=self._keywords_str)

    @staticmethod
    def _find(model_dir: Path, prefix: str) -> str:
        """Find model file, preferring int8 quantized variant."""
        for pattern in (f"{prefix}*.int8.onnx", f"{prefix}*.onnx"):
            matches = list(model_dir.glob(pattern))
            if matches:
                return str(matches[0])
        raise FileNotFoundError(f"No {prefix} model found in {model_dir}")

    def predict(self, audio: bytes | np.ndarray) -> dict[str, float]:
        """Feed audio chunk; return {keyword: 1.0} on detection, else 0.0."""
        if isinstance(audio, (bytes, bytearray)):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        self._stream.accept_waveform(sample_rate=16000, waveform=audio)
        self._spotter.decode_stream(self._stream)

        result = {kw: 0.0 for kw in self._keywords}
        detected = self._stream.result.keyword.strip()
        if detected:
            detected_lower = detected.lower()
            for kw in self._keywords:
                if kw.lower() in detected_lower or detected_lower in kw.lower():
                    result[kw] = 1.0
                    break
            self._spotter.reset_stream(self._stream)

        return result

    def reset(self):
        self._spotter.reset_stream(self._stream)

    def delete(self):
        pass  # Sherpa-ONNX cleans up on GC
