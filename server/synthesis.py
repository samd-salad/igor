"""Text-to-speech synthesis using Piper."""
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

from server.config import PIPER_VOICE, PIPER_SAMPLE_RATE

logger = logging.getLogger(__name__)


class Synthesizer:
    """Handles text-to-speech using Piper TTS."""

    def __init__(self, voice_path: str = PIPER_VOICE):
        # Validate voice model path for security
        voice_path_obj = Path(voice_path).resolve()
        if not voice_path_obj.exists():
            raise ValueError(f"Voice model not found: {voice_path}")
        if not voice_path_obj.is_file():
            raise ValueError(f"Voice model is not a file: {voice_path}")

        self.voice_path = str(voice_path_obj)
        self.sample_rate = PIPER_SAMPLE_RATE
        logger.info(f"Synthesizer initialized with voice: {self.voice_path}")

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to speech and return audio as bytes.

        Args:
            text: Text to speak

        Returns:
            WAV audio data as bytes, or None on failure
        """
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None

        # Truncate text for logging to avoid leaking sensitive data
        log_text = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"Synthesizing: '{log_text}'")

        temp_wav = None
        try:
            # Create temp file for output (keep file handle open for security)
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav = temp_file.name
            temp_file.close()

            # Run Piper with list arguments (NO shell=True to prevent injection)
            # Pass text via stdin to avoid command line length limits and injection
            result = subprocess.run(
                ['piper', '--model', self.voice_path, '--output_file', temp_wav],
                input=text.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='replace')
                logger.error(f"Piper failed with code {result.returncode}: {error_msg[:200]}")
                return None

            # Read the generated audio
            if not os.path.exists(temp_wav):
                logger.error(f"Output file not created: {temp_wav}")
                return None

            with open(temp_wav, 'rb') as f:
                audio_data = f.read()

            if not audio_data:
                logger.error("Piper generated empty audio file")
                return None

            logger.info(f"Synthesized {len(audio_data)} bytes of audio")
            return audio_data

        except subprocess.TimeoutExpired:
            logger.error("TTS synthesis timed out after 30 seconds")
            return None
        except FileNotFoundError:
            logger.error("Piper executable not found. Is it installed and in PATH?")
            return None
        except Exception as e:
            logger.error(f"TTS synthesis failed: {type(e).__name__}: {str(e)[:200]}")
            return None
        finally:
            # Always clean up temp file
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_wav}: {e}")

    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to speech and save to file.

        Args:
            text: Text to speak
            output_path: Path to save WAV file

        Returns:
            True on success, False on failure
        """
        audio_data = self.synthesize(text)
        if not audio_data:
            return False

        try:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"Saved synthesized audio to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
