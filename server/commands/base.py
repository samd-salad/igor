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

    @property
    def required_parameters(self) -> list:
        """List of required parameter names. Override to make some optional."""
        return list(self.parameters.keys())

    def get_defaults(self, brain, hour: int) -> dict:
        """Return learned default parameters for this command at the given hour.

        Override in subclasses to enable contextual defaults.
        Returns empty dict by default.
        """
        return {}

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
                "required": self.required_parameters
            }
        }
