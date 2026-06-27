import struct
from server.external.embedding_encoder import EmbeddingEncoder


def test_encode_returns_384_float32_bytes():
    enc = EmbeddingEncoder()
    out = enc.encode("dark roast coffee")
    assert isinstance(out, bytes)
    assert len(out) == 384 * 4   # 384 float32s
    floats = struct.unpack(f"<{384}f", out)
    assert any(f != 0.0 for f in floats)


def test_same_text_encodes_to_same_bytes():
    enc = EmbeddingEncoder()
    a = enc.encode("the cat sat on the mat")
    b = enc.encode("the cat sat on the mat")
    assert a == b


def test_different_text_encodes_to_different_bytes():
    enc = EmbeddingEncoder()
    a = enc.encode("dark roast coffee")
    b = enc.encode("the cat sat on the mat")
    assert a != b
