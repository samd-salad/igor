from .base import Command
from ._utils import parse_amount, parse_direction_updown, parse_volume_word

_VOL_S, _VOL_M, _VOL_L = 5, 15, 30


class VolumeCommand(Command):
    name = "set_volume"
    description = (
        "Set the system volume on the Raspberry Pi. "
        "Accepts a number (0-100) or a word like quiet, low, medium, loud, max."
    )

    @property
    def parameters(self) -> dict:
        return {
            "level": {
                "type": "string",
                "description": "Volume level: a number 0-100, or a word (quiet, low, medium, loud, max)"
            }
        }

    def execute(self, level) -> str:
        level_str = str(level).lower().strip()
        word_val = parse_volume_word(level_str)
        if word_val is not None:
            level = word_val
        else:
            try:
                level = max(0, min(100, int(float(level_str.rstrip("%")))))
            except ValueError:
                return f"Couldn't understand volume level '{level}'. Try a number or: quiet, low, medium, loud, max."
        if hasattr(self, 'pi_client') and self.pi_client:
            try:
                result = self.pi_client.hardware_control("set_volume", {"level": level})
                return result if result else f"Failed to set volume to {level}%"
            except Exception as e:
                return f"Error setting volume: {e}"
        return "Volume control not available (Pi not connected)"


class AdjustVolumeCommand(Command):
    name = "adjust_volume"
    description = (
        "Increase or decrease the volume relative to its current level. "
        "Use for 'turn it up a bit', 'slightly louder', 'a lot quieter', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {
                "type": "string",
                "description": "Direction: 'up' / 'louder' to increase, 'down' / 'quieter' to decrease"
            },
            "amount": {
                "type": "string",
                "description": "Step size: 'slightly', 'medium' (default), or 'a lot'"
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, amount: str = "medium") -> str:
        if not (hasattr(self, 'pi_client') and self.pi_client):
            return "Volume control not available (Pi not connected)"

        d = parse_direction_updown(direction)
        if d is None:
            return f"Unknown direction '{direction}'. Use 'up'/'louder' or 'down'/'quieter'."
        up = d == "up"
        step = parse_amount(amount, _VOL_S, _VOL_M, _VOL_L)

        try:
            current_result = self.pi_client.hardware_control("get_volume", {})
            current = int(''.join(filter(str.isdigit, current_result or "50")) or 50)
        except Exception:
            current = 50

        new_level = max(0, min(100, current + (step if up else -step)))
        try:
            result = self.pi_client.hardware_control("set_volume", {"level": new_level})
            return result if result else f"Volume {'increased' if up else 'decreased'} to {new_level}%"
        except Exception as e:
            return f"Error adjusting volume: {e}"


class GetVolumeCommand(Command):
    name = "get_volume"
    description = "Get the current system volume level on the Raspberry Pi"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        if hasattr(self, 'pi_client') and self.pi_client:
            try:
                result = self.pi_client.hardware_control("get_volume", {})
                return result if result else "Could not retrieve volume"
            except Exception as e:
                return f"Error getting volume: {e}"
        return "Volume control not available (Pi not connected)"
