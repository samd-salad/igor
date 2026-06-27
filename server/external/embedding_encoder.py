"""ONNX-based embedding encoder for Igor's memory layer.

Wraps fastembed (Qdrant) with BAAI/bge-small-en-v1.5 (384-dim, ~33M params).
Lazy-loads on first call so test startup is cheap.
"""
from __future__ import annotations
from typing import Optional

from fastembed import TextEmbedding


_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384


class EmbeddingEncoder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[TextEmbedding] = None

    def encode(self, text: str) -> bytes:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name)
        vec = next(self._model.embed([text]))
        arr = vec.astype("float32")
        if arr.shape != (_EMBED_DIM,):
            raise RuntimeError(
                f"unexpected embedding shape {arr.shape}, expected ({_EMBED_DIM},)"
            )
        return arr.tobytes()
