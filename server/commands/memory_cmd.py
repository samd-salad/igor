import json
import logging
import re
import threading
import time
from .base import Command
from server.config import MEMORY_FILE

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def _sanitize(text: str, max_len: int = 500) -> str:
    """Strip control characters and XML-like tags; limit length.

    Prevents prompt injection via persistent memory: stored values are
    injected verbatim into the system prompt, so we must treat them as
    untrusted even though they come through the LLM.
    """
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # strip control chars (keep \t \n \r)
    text = re.sub(r'<[^>]{0,100}>', '', text)   # strip XML/HTML-like tags
    return text[:max_len].strip()

# Use JSON file for structured memory
MEMORY_JSON_FILE = MEMORY_FILE.with_suffix('.json')


def _load_memories() -> dict:
    """Load memories from JSON file."""
    if MEMORY_JSON_FILE.exists():
        try:
            return json.loads(MEMORY_JSON_FILE.read_text())
        except json.JSONDecodeError:
            backup = MEMORY_JSON_FILE.with_name(f"memory.bak.{int(time.time())}.json")
            MEMORY_JSON_FILE.rename(backup)
            logger.warning(f"Memory file corrupted — backed up to {backup.name}, starting fresh")
            return {}
    # Migrate from old text format if it exists
    if MEMORY_FILE.exists():
        old_content = MEMORY_FILE.read_text().strip()
        if old_content:
            logger.info("Migrating old memory format to JSON")
            return {"migrated": {"old_memories": old_content}}
    return {}


def _save_memories(memories: dict):
    """Save memories to JSON file atomically."""
    MEMORY_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_JSON_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(memories, indent=2))
    tmp.replace(MEMORY_JSON_FILE)


class SaveMemoryCommand(Command):
    name = "save_memory"
    description = "Save or update a fact about the user. Use category and key to organize. Calling with same category+key updates the value."

    @property
    def parameters(self) -> dict:
        return {
            "category": {
                "type": "string",
                "description": "Category: 'preferences', 'schedule', 'people', 'personal', 'home', or 'other'"
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
        # Normalize and sanitize — values are injected into the system prompt
        category = _sanitize(category, max_len=50).lower().strip()
        key = _sanitize(key, max_len=50).lower().strip().replace(" ", "_")
        value = _sanitize(value, max_len=500)

        with _lock:
            memories = _load_memories()
            is_update = category in memories and key in memories.get(category, {})
            if category not in memories:
                memories[category] = {}
            memories[category][key] = value
            _save_memories(memories)

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

        with _lock:
            memories = _load_memories()
            if category in memories and key in memories[category]:
                del memories[category][key]
                if not memories[category]:
                    del memories[category]
                _save_memories(memories)
                logger.info(f"Memory removed: [{category}][{key}]")
                return f"Forgot: {key}"
        return f"No memory found for {category}/{key}"


def load_persistent_memory() -> str:
    """Load persistent memory formatted for prompt injection."""
    memories = _load_memories()

    if not memories:
        return "(empty)"

    # Format as readable text for the LLM
    lines = []
    for category, items in sorted(memories.items()):
        lines.append(f"[{category}]")
        for key, value in sorted(items.items()):
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
