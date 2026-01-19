"""Minimal wake word detector using raw ONNX inference."""
import numpy as np
import os
os.environ["ORT_DISABLE_ALL_LOGS"] = "1"
import onnxruntime as ort
ort.set_default_logger_severity(3)  # Suppress warnings
from pathlib import Path
import urllib.request

class WakeWordDetector:
    def __init__(self, model_paths: list[str], threshold: float = 0.5):
        self.threshold = threshold
        self.models = {}
        self.buffers = {}
        
        oww_path = Path(__file__).parent / "oww_models"
        oww_path.mkdir(exist_ok=True)
        
        self._ensure_base_models(oww_path)
        
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        
        self.melspec = ort.InferenceSession(str(oww_path / "melspectrogram.onnx"), opts, providers=["CPUExecutionProvider"])
        self.embedding = ort.InferenceSession(str(oww_path / "embedding_model.onnx"), opts, providers=["CPUExecutionProvider"])
        
        for path in model_paths:
            name = Path(path).stem
            sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            self.models[name] = {
                "session": sess,
                "input_name": sess.get_inputs()[0].name
            }
            self.buffers[name] = np.zeros((1, 16, 96), dtype=np.float32)
        
        self.mel_buffer = np.zeros((0, 32), dtype=np.float32)
    
    def _ensure_base_models(self, path: Path):
        urls = {
            "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
            "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
        }
        for fname, url in urls.items():
            fpath = path / fname
            if not fpath.exists() or fpath.stat().st_size < 1000:
                print(f"Downloading {fname}...")
                urllib.request.urlretrieve(url, fpath)
                print(f"  -> {fpath.stat().st_size} bytes")
    
    def predict(self, audio: np.ndarray) -> dict[str, float]:
        audio_float = audio.astype(np.float32).reshape(1, -1) / 32768.0
        mel = self.melspec.run(None, {"input": audio_float})[0]
        mel = (mel / 10.0) + 2.0
        mel = mel.squeeze()
        if mel.ndim == 1:
            mel = mel.reshape(1, -1)
        
        self.mel_buffer = np.concatenate([self.mel_buffer, mel], axis=0)
        results = {name: 0.0 for name in self.models}
        
        while self.mel_buffer.shape[0] >= 76:
            mel_chunk = self.mel_buffer[:76, :].reshape(1, 76, 32, 1)
            self.mel_buffer = self.mel_buffer[8:, :]
            emb = self.embedding.run(None, {"input_1": mel_chunk})[0]
            emb = emb.flatten()[:96]
            
            for name in self.models:
                self.buffers[name] = np.roll(self.buffers[name], -1, axis=1)
                self.buffers[name][0, -1, :] = emb
            
            for name, model_info in self.models.items():
                out = model_info["session"].run(None, {model_info["input_name"]: self.buffers[name]})[0]
                results[name] = max(results[name], float(out[0][0]))
        
        return results
    
    def reset(self):
        for name in self.buffers:
            self.buffers[name] = np.zeros((1, 16, 96), dtype=np.float32)
        self.mel_buffer = np.zeros((0, 32), dtype=np.float32)
