"""Server configuration for PC backend."""
import os
from pathlib import Path

# Server network settings
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000

# Pi client settings (for callbacks)
PI_HOST = "192.168.0.3"
PI_PORT = 8080

# IP allowlist for sensitive server endpoints — add more Pi IPs here when scaling
ALLOWED_CLIENT_IPS: set = {PI_HOST}

# LLM configuration - Claude API
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_INPUT_COST_PER_M  = 0.80   # USD per million input tokens
CLAUDE_OUTPUT_COST_PER_M = 4.00   # USD per million output tokens
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
MEMORY_FILE = DATA_DIR / "memory.txt"  # Legacy path; memory_cmd.py uses memory.json (derived via .with_suffix)
KNOWN_DEVICES_FILE = DATA_DIR / "known_devices.json"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"
CONVERSATION_STATE_FILE = DATA_DIR / "conversation_state.json"

# Speaker identification
SPEAKER_EMBEDDINGS_FILE = DATA_DIR / "speaker_embeddings.json"
SPEAKER_SIMILARITY_THRESHOLD = 0.75  # 0-1, higher = stricter matching

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Weather
DEFAULT_LOCATION = "Arlington, VA"  # Default city for get_weather when no location specified

# Request timeouts
REQUEST_TIMEOUT = 5.0  # Timeout for HTTP requests to Pi

# Sonos soundbar/speaker settings (local LAN, no API key)
SONOS_DISCOVERY_CACHE_TTL = 300  # seconds before forced re-discovery (5 min)
SONOS_DEFAULT_ZONE = "Living Room"  # Default zone for TV/soundbar/music requests
SONOS_TTS_OUTPUT = True   # Allow clients to route TTS through Sonos (per-client opt-in via prefer_sonos_output)
SERVER_EXTERNAL_HOST = "192.168.0.4"  # PC's LAN IP (Sonos fetches TTS audio from here)

# LIFX smart bulb settings (local LAN UDP, no API key)
LIFX_DISCOVERY_CACHE_TTL = 300  # seconds before forced re-discovery (5 min)

# Google TV remote settings
GOOGLE_TV_HOST = "192.168.0.20"
GOOGLE_TV_CERT_FILE = str(DATA_DIR / "google_tv_cert.pem")
GOOGLE_TV_KEY_FILE  = str(DATA_DIR / "google_tv_key.pem")
GOOGLE_TV_CLIENT_NAME = "DrButts"
ADB_KEY_FILE = str(DATA_DIR / "adbkey")
