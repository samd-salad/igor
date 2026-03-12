import logging
import re
from .base import Command

logger = logging.getLogger(__name__)


def _sanitize(text: str, max_len: int = 500) -> str:
    """Strip control characters and XML-like tags; limit length.

    Prevents prompt injection via persistent memory: stored values are
    injected verbatim into the system prompt, so we must treat them as
    untrusted even though they come through the LLM.
    """
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r' +', ' ', text)
    return text[:max_len].strip()


class SaveMemoryCommand(Command):
    name = "save_memory"
    description = "Save or update a fact about the user. Use category and key to organize. Calling with same category+key updates the value."

    @property
    def parameters(self) -> dict:
        return {
            "category": {
                "type": "string",
                "description": "Category: 'preferences', 'schedule', 'people', 'personal', 'home', 'behavior', or 'other'"
            },
            "key": {
                "type": "string",
                "description": "Short identifier for this memory (e.g., 'coffee', 'sleep_time', 'sister', 'name')"
            },
            "value": {
                "type": "string",
                "description": "The fact to remember (e.g., 'dark roast with oat milk', 'usually around 11pm')"
            }
        }

    def execute(self, category: str, key: str, value: str) -> str:
        category = _sanitize(category, max_len=50).lower().strip()
        key = _sanitize(key, max_len=50).lower().strip().replace(" ", "_")
        value = _sanitize(value, max_len=500)

        from server.brain import get_brain
        brain = get_brain()
        entry_id, is_update = brain.save_memory(category, key, value)

        action = "Updated" if is_update else "Saved"
        logger.info(f"Memory {action.lower()}: [{category}][{key}] = {value}")
        return f"{action}: {key} = {value}"


class ForgetMemoryCommand(Command):
    name = "forget_memory"
    description = "Remove a specific memory by category and key"

    @property
    def parameters(self) -> dict:
        return {
            "category": {
                "type": "string",
                "description": "Category of the memory to remove"
            },
            "key": {
                "type": "string",
                "description": "Key of the memory to remove"
            }
        }

    def execute(self, category: str, key: str) -> str:
        category = category.lower().strip()
        key = key.lower().strip().replace(" ", "_")

        from server.brain import get_brain
        brain = get_brain()
        if brain.forget_memory(category, key):
            logger.info(f"Memory removed: [{category}][{key}]")
            return f"Forgot: {key}"
        return f"No memory found for {category}/{key}"
