from datetime import datetime
from .base import Command

class TimeCommand(Command):
    name = "get_time"
    description = "Get the current time and/or date"

    @property
    def required_parameters(self) -> list:
        return []

    @property
    def parameters(self) -> dict:
        return {
            "include_date": {
                "type": "boolean",
                "description": "Whether to include the date"
            }
        }
    
    def execute(self, include_date: bool = False) -> str:
        now = datetime.now()
        if include_date:
            return now.strftime("It's %I:%M %p on %A, %B %d")
        return now.strftime("It's %I:%M %p")
