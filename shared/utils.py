"""Common utility functions shared between client and server."""
import base64
import logging
import time
from typing import Optional


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up consistent logging format for client or server."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def encode_audio_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64 string for transmission."""
    return base64.b64encode(audio_bytes).decode('utf-8')


def decode_audio_base64(audio_base64: str) -> bytes:
    """Decode base64 string back to audio bytes."""
    return base64.b64decode(audio_base64.encode('utf-8'))


def read_wav_file(file_path: str) -> bytes:
    """Read a WAV file and return its contents as bytes."""
    with open(file_path, 'rb') as f:
        return f.read()


def get_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()

