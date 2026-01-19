"""Speech-to-text transcription using Faster Whisper."""
import logging
from typing import Optional
from faster_whisper import WhisperModel

from server.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)


class Transcriber:
    """Handles speech-to-text transcription using Faster Whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model: Optional[WhisperModel] = None
        logger.info(f"Transcriber initialized with model: {model_name}")

    def initialize(self) -> bool:
        """Load Whisper model. Returns success."""
        try:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.model = WhisperModel(
                self.model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Whisper initialization failed: {e}")
            return False

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to WAV file

        Returns:
            Transcribed text or None on failure
        """
        if not self.model:
            logger.error("Transcriber not initialized")
            return None

        try:
            logger.debug(f"Transcribing audio file: {audio_path}")
            segments, info = self.model.transcribe(audio_path, beam_size=1)
            text = " ".join(seg.text for seg in segments).strip()

            if not text:
                logger.warning("Transcription resulted in empty text")
                return None

            logger.info(f"Transcribed: '{text}'")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def transcribe_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """
        Transcribe audio from bytes.

        Args:
            audio_bytes: WAV audio data as bytes

        Returns:
            Transcribed text or None on failure
        """
        import tempfile
        import os

        # Write bytes to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return self.transcribe(temp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
