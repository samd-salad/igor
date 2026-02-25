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
        level = max(0, min(100, level))
        if hasattr(self, 'pi_client') and self.pi_client:
            try:
                result = self.pi_client.hardware_control("set_volume", {"level": level})
                return result if result else f"Failed to set volume to {level}%"
            except Exception as e:
                return f"Error setting volume: {e}"
        return "Volume control not available (Pi not connected)"


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
