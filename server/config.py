"""Server configuration for PC backend."""
import os
from pathlib import Path

# Server network settings
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000

# Pi client settings (for callbacks)
PI_HOST = "192.168.0.3"
PI_PORT = 8080

# LLM configuration - Claude API
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_CONVERSATION_HISTORY = 10  # Keep last 10 messages

# Whisper model - speech recognition
WHISPER_MODEL = "small"  # tiny = fastest, base = balanced, small = slower/accurate
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Paths on PC
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Piper TTS model
PIPER_VOICE = str(BASE_DIR / "voices" / "en_US-arctic-medium.onnx")
PIPER_SAMPLE_RATE = 22050
MEMORY_FILE = DATA_DIR / "memory.txt"
KNOWN_DEVICES_FILE = DATA_DIR / "known_devices.json"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"
CONVERSATION_STATE_FILE = DATA_DIR / "conversation_state.json"

# Speaker identification
SPEAKER_EMBEDDINGS_FILE = DATA_DIR / "speaker_embeddings.json"
SPEAKER_SIMILARITY_THRESHOLD = 0.75  # 0-1, higher = stricter matching

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Request timeouts
REQUEST_TIMEOUT = 5.0  # Timeout for HTTP requests to Pi
