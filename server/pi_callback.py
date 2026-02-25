"""Client for making callbacks to Pi (for timer alerts, hardware control, etc.)."""
import logging
import requests
from typing import Optional

from shared.protocol import PLAY_AUDIO_ENDPOINT, HARDWARE_CONTROL_ENDPOINT, PLAY_BEEP_ENDPOINT, REQUEST_TIMEOUT
from shared.models import PlayAudioRequest, HardwareControlRequest, PlayBeepRequest, BeepType, Priority, Status
from shared.utils import encode_audio_base64

logger = logging.getLogger(__name__)


class PiCallbackClient:
    """Handles callbacks from server to Pi for audio playback and hardware control."""

    def __init__(self, pi_base_url: str):
        self.pi_base_url = pi_base_url
        logger.info(f"PiCallbackClient initialized: {pi_base_url}")

    def play_audio(self, audio_bytes: bytes, message: str, priority: str = "normal") -> bool:
        """
        Send audio to Pi for playback.

        Args:
            audio_bytes: WAV audio data
            message: Text message being spoken (for logging)
            priority: "normal" or "alert"

        Returns:
            True on success, False on failure
        """
        try:
            audio_b64 = encode_audio_base64(audio_bytes)
            request = PlayAudioRequest(
                audio_base64=audio_b64,
                message=message,
                priority=Priority(priority)
            )

            logger.info(f"Sending audio to Pi: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            response = requests.post(
                f"{self.pi_base_url}{PLAY_AUDIO_ENDPOINT}",
                json=request.model_dump(),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            if result.get('status') == Status.SUCCESS.value:
                logger.info("Audio sent to Pi successfully")
                return True
            else:
                logger.error(f"Pi reported error: {result.get('error')}")
                return False

        except requests.Timeout:
            logger.error("Pi callback timed out")
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to send audio to Pi: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending audio to Pi: {e}")
            return False

    def hardware_control(self, command: str, parameters: dict) -> Optional[str]:
        """
        Execute hardware command on Pi.

        Args:
            command: Command name (e.g., 'set_volume')
            parameters: Command parameters

        Returns:
            Result message or None on failure
        """
        try:
            request = HardwareControlRequest(
                command=command,
                parameters=parameters
            )

            logger.info(f"Sending hardware command to Pi: {command}({parameters})")
            response = requests.post(
                f"{self.pi_base_url}{HARDWARE_CONTROL_ENDPOINT}",
                json=request.model_dump(),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            if result.get('status') == Status.SUCCESS.value:
                logger.info(f"Hardware command succeeded: {result.get('result')}")
                return result.get('result')
            else:
                logger.error(f"Hardware command failed: {result.get('error')}")
                return None

        except requests.Timeout:
            logger.error("Pi hardware control timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to send hardware command to Pi: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending hardware command: {e}")
            return None

    def play_beep(self, beep_type: str) -> bool:
        """
        Play beep sound on Pi.

        Args:
            beep_type: Type of beep ("alert", "error", "done", "start", "end")

        Returns:
            True on success, False on failure
        """
        try:
            request = PlayBeepRequest(beep_type=BeepType(beep_type))

            logger.debug(f"Sending beep request to Pi: {beep_type}")
            response = requests.post(
                f"{self.pi_base_url}{PLAY_BEEP_ENDPOINT}",
                json=request.model_dump(),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            return result.get('status') == Status.SUCCESS.value

        except requests.Timeout:
            logger.error("Pi beep request timed out")
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to send beep to Pi: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending beep: {e}")
            return False

    def check_health(self) -> bool:
        """
        Check if Pi is reachable and healthy.

        Returns:
            True if Pi is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.pi_base_url}/api/health",
                timeout=2.0
            )
            response.raise_for_status()
            result = response.json()
            return result.get('status') == 'healthy'
        except Exception:
            return False
