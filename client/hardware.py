"""Hardware control functions for Raspberry Pi."""
import logging
import subprocess
import re

logger = logging.getLogger(__name__)


class HardwareController:
    """Handles hardware control on Raspberry Pi."""

    @staticmethod
    def set_volume(level: int) -> str:
        """
        Set ALSA audio volume.

        Args:
            level: Volume level (0-100)

        Returns:
            Result message
        """
        try:
            # Clamp level to valid range
            level = max(0, min(100, int(level)))

            # Use amixer to set volume
            result = subprocess.run(
                ['amixer', 'set', 'Master', f'{level}%'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                logger.info(f"Volume set to {level}%")
                return f"Volume set to {level}%"
            else:
                error_msg = result.stderr.strip()
                logger.error(f"Failed to set volume: {error_msg}")
                return f"Failed to set volume: {error_msg}"

        except subprocess.TimeoutExpired:
            logger.error("Volume control timed out")
            return "Volume control timed out"
        except FileNotFoundError:
            logger.error("amixer command not found")
            return "amixer not available"
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return f"Error: {e}"

    @staticmethod
    def get_volume() -> str:
        """
        Get current ALSA audio volume.

        Returns:
            Volume level as string
        """
        try:
            result = subprocess.run(
                ['amixer', 'get', 'Master'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse output to extract volume percentage
                # Example output: "Mono: Playback 65 [72%] [on]"
                match = re.search(r'\[(\d+)%\]', result.stdout)
                if match:
                    volume = match.group(1)
                    logger.info(f"Current volume: {volume}%")
                    return f"Current volume is {volume}%"
                else:
                    logger.warning("Could not parse volume from amixer output")
                    return "Could not determine volume"
            else:
                error_msg = result.stderr.strip()
                logger.error(f"Failed to get volume: {error_msg}")
                return f"Failed to get volume: {error_msg}"

        except subprocess.TimeoutExpired:
            logger.error("Volume check timed out")
            return "Volume check timed out"
        except FileNotFoundError:
            logger.error("amixer command not found")
            return "amixer not available"
        except Exception as e:
            logger.error(f"Error getting volume: {e}")
            return f"Error: {e}"
