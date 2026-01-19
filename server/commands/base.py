from abc import ABC, abstractmethod
from typing import Any

class Command(ABC):
    """Base class for all commands."""
    
    name: str = ""
    description: str = ""
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for command parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the command and return a response string."""
        pass
    
    def to_tool(self) -> dict:
        """Convert to Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }
