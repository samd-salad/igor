"""Client for making callbacks from the PC server to the Raspberry Pi.

The Pi runs a Flask server (client/pi_server.py) that accepts HTTP requests
from this module.  Three use cases:
  1. Timer alerts — server synthesizes TTS and sends the WAV to the Pi to play.
  2. Hardware control — Pi owns the ALSA mixer; volume commands are RPC'd here.
  3. Beep playback — timer event loop signals the Pi to play alert beeps.
  4. Wake word suppression — after TV commands, server tells Pi to ignore mic
     for N seconds so startup audio doesn't retrigger the detector.

All methods are synchronous blocking calls — callers are responsible for running
them in background threads if they shouldn't block (e.g. suppress_wakeword is
always called from a daemon thread in orchestrator._execute_command).
"""
import logging
import requests
from typing import Optional

from shared.protocol import PLAY_AUDIO_ENDPOINT, HARDWARE_CONTROL_ENDPOINT, PLAY_BEEP_ENDPOINT, SUPPRESS_WAKEWORD_ENDPOINT, REQUEST_TIMEOUT
from shared.models import PlayAudioRequest, HardwareControlRequest, PlayBeepRequest, BeepType, Priority, Status
from shared.utils import encode_audio_base64

logger = logging.getLogger(__name__)


class PiCallbackClient:
    """Handles callbacks from server to Pi for audio playback and hardware control."""

    def __init__(self, pi_base_url: str):
        # Base URL of the Pi Flask server, e.g. "http://192.168.0.3:8080"
        self.pi_base_url = pi_base_url
        logger.info(f"PiCallbackClient initialized: {pi_base_url}")

    def play_audio(self, audio_bytes: bytes, message: str, priority: str = "normal") -> bool:
        """Send synthesized WAV audio to the Pi for playback through its speaker.

        Used by the timer event loop to deliver alerts: server synthesizes the
        "pasta timer is done" message and ships the WAV over HTTP to the Pi.
        The Pi's /api/play_audio endpoint decodes and plays it via PyAudio.

        Args:
            audio_bytes: Raw WAV audio data (not base64 — encoding done here).
            message: Human-readable description for logging (truncated to 50 chars).
            priority: "normal" or "alert" — currently informational only.

        Returns:
            True if Pi acknowledged playback, False on any error.
        """
        try:
            # Encode to base64 for JSON transport
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
        """Execute a hardware command on the Pi via RPC.

        Volume lives on the Pi (ALSA mixer), so set_volume / get_volume are
        forwarded here rather than executed locally on the server.  The Pi's
        HardwareController reads/writes the ALSA sink volume.

        Only 'set_volume' and 'get_volume' are whitelisted in HardwareControlRequest
        (shared/models.py) — any other command is rejected by Pydantic validation.

        Args:
            command: Whitelisted command name ('set_volume', 'get_volume').
            parameters: Command kwargs dict, e.g. {'level': 75}.

        Returns:
            Result string from Pi (e.g. "Volume set to 75%"), or None on failure.
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
        """Play a beep sound through the Pi's speaker.

        Called by the timer event loop before speaking the alert ("beep, then
        'pasta timer is done'") to give a heads-up signal.  Beep types map to
        BeepType enum values; the Pi's audio.py generates them via sox.

        Args:
            beep_type: One of "alert", "error", "done", "start", "end".

        Returns:
            True on success, False on any failure (non-critical — timer alert
            will still attempt TTS even if the beep fails).
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

    def suppress_wakeword(self, seconds: float = 20.0) -> bool:
        """Tell the Pi to suppress wake word detection for `seconds` seconds.

        Called after any TV command (power on, app launch, playback) so that
        Netflix startup audio / content audio doesn't immediately retrigger
        the wake word detector.

        This call is always made from a daemon thread in orchestrator._execute_command
        so it never blocks the main interaction response.  Timeout is intentionally
        short (2s) — suppression is best-effort; if the Pi is unreachable the
        interaction still completes, it just risks a false trigger.

        Args:
            seconds: How long to suppress from now.  Default 20s covers app launch
                     startup audio (Netflix, YouTube intro sounds, etc.).
        """
        try:
            response = requests.post(
                f"{self.pi_base_url}{SUPPRESS_WAKEWORD_ENDPOINT}",
                json={"seconds": seconds},
                timeout=2.0,  # Short timeout — this is fire-and-forget
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.debug(f"suppress_wakeword failed (non-critical): {e}")
            return False

    def check_health(self) -> bool:
        """Check if the Pi server is reachable and healthy.

        Used by the server health endpoint to report Pi connectivity status.
        Failure here does NOT prevent interactions — it's informational only.

        Returns:
            True if Pi responds with status='healthy', False otherwise.
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
