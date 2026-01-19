from pathlib import Path
from .base import Command

MEMORY_FILE = Path(__file__).parent.parent / "memory.txt"

class SaveMemoryCommand(Command):
    name = "save_memory"
    description = "Save important information about the user or home to persistent memory"
    
    @property
    def parameters(self) -> dict:
        return {
            "fact": {
                "type": "string",
                "description": "The fact to remember (e.g., 'User's name is Sam', 'Has a dog named Max', 'Prefers metric units')"
            }
        }
    
    def execute(self, fact: str) -> str:
        # Load existing memories
        if MEMORY_FILE.exists():
            existing = MEMORY_FILE.read_text().strip()
        else:
            existing = ""
        
        # Append new fact
        new_memory = f"{existing}\n- {fact}".strip()
        MEMORY_FILE.write_text(new_memory)
        
        return f"Saved to memory: {fact}"


def load_persistent_memory() -> str:
    """Load persistent memory for injection into prompt."""
    if MEMORY_FILE.exists():
        content = MEMORY_FILE.read_text().strip()
        return content if content else "(empty)"
    return "(empty)"
