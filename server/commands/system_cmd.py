from .base import Command

class VolumeCommand(Command):
    name = "set_volume"
    description = "Set the system volume level on the Raspberry Pi"

    @property
    def parameters(self) -> dict:
        return {
            "level": {
                "type": "integer",
                "description": "Volume level from 0 to 100"
            }
        }

    def execute(self, level: int) -> str:
        # Clamp level to valid range
        level = max(0, min(100, level))

        # RPC to Pi for hardware control
        if hasattr(self, 'pi_client') and self.pi_client:
            try:
                result = self.pi_client.hardware_control("set_volume", {"level": level})
                if result:
                    return result  # Result string from Pi
                else:
                    return f"Failed to set volume to {level}%"
            except Exception as e:
                return f"Error setting volume: {e}"
        else:
            return "Volume control not available (Pi not connected)"
