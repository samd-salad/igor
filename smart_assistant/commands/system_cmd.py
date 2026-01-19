import subprocess
from .base import Command

class VolumeCommand(Command):
    name = "set_volume"
    description = "Set the system volume level"
    
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
        subprocess.run(f"amixer set Master {level}%", shell=True, capture_output=True)
        return f"Volume set to {level}%"
