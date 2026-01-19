"""Server configuration for PC backend."""
import os
from pathlib import Path

# Server network settings
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000

# Pi client settings (for callbacks)
PI_HOST = "192.168.0.3"
PI_PORT = 8080

# LLM configuration - local Ollama instance
OLLAMA_URL = "http://localhost:11434"  # Running locally on PC
OLLAMA_MODEL = "qwen3:30b"
MAX_CONVERSATION_HISTORY = 10  # Keep last 10 messages

# Whisper model - speech recognition
WHISPER_MODEL = "base"  # tiny = fastest, base = balanced, small = slower/accurate
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Piper TTS model
PIPER_VOICE = os.path.expanduser("~/.local/share/piper-voices/en_US-arctic-medium.onnx")
PIPER_SAMPLE_RATE = 22050

# Paths on PC
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_FILE = DATA_DIR / "memory.txt"
KNOWN_DEVICES_FILE = DATA_DIR / "known_devices.json"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"
CONVERSATION_STATE_FILE = DATA_DIR / "conversation_state.json"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Request timeouts
REQUEST_TIMEOUT = 5.0  # Timeout for HTTP requests to Pi
