"""Server configuration for Igor conversation agent."""
import os
from pathlib import Path

# Server network settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# LLM — Claude API
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_INPUT_COST_PER_M = 0.80    # USD per million input tokens
CLAUDE_OUTPUT_COST_PER_M = 4.00   # USD per million output tokens
MAX_CONVERSATION_HISTORY = 10     # last N messages kept

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BRAIN_FILE = DATA_DIR / "brain.json"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"
DATA_DIR.mkdir(exist_ok=True)

# Weather
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Arlington, VA")
