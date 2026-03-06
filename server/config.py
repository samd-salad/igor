"""Server configuration for PC backend."""
import os
from pathlib import Path

# Server network settings — bind to LAN IP only (not 0.0.0.0)
SERVER_HOST = os.getenv("SERVER_HOST", "192.168.0.4")
SERVER_PORT = 8000

# Pi client settings (for callbacks)
PI_HOST = os.getenv("PI_HOST", "192.168.0.3")
PI_PORT = int(os.getenv("PI_PORT", "8080"))

# IP allowlist for sensitive server endpoints — add more Pi IPs here when scaling
ALLOWED_CLIENT_IPS: set = {PI_HOST}

# LLM configuration - Claude API
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
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

# Kokoro TTS (kokoro-onnx) — model files in kokoro/ dir
# Download from https://github.com/thewh1teagle/kokoro-onnx/releases
KOKORO_MODEL_FILE = str(BASE_DIR / "kokoro" / "kokoro-v1.0.onnx")
KOKORO_VOICES_FILE = str(BASE_DIR / "kokoro" / "voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_onyx")  # am_onyx, am_michael, am_fenrir, af_heart, af_bella
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_SAMPLE_RATE = 24000
MEMORY_FILE = DATA_DIR / "memory.txt"  # Legacy path; memory_cmd.py uses memory.json (derived via .with_suffix)
KNOWN_DEVICES_FILE = DATA_DIR / "known_devices.json"
BENCHMARK_FILE = DATA_DIR / "benchmark.csv"
CONVERSATION_STATE_FILE = DATA_DIR / "conversation_state.json"
ROUTINES_FILE = DATA_DIR / "routines.json"

# Speaker identification
SPEAKER_EMBEDDINGS_FILE = DATA_DIR / "speaker_embeddings.json"
SPEAKER_SIMILARITY_THRESHOLD = 0.75  # 0-1, higher = stricter matching

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Weather
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Arlington, VA")  # Default city for get_weather

# Request timeouts
REQUEST_TIMEOUT = 5.0  # Timeout for HTTP requests to Pi

# Sonos soundbar/speaker settings (local LAN, no API key)
SONOS_DISCOVERY_CACHE_TTL = 300  # seconds before forced re-discovery (5 min)
SONOS_DEFAULT_ZONE = "Living Room"  # Default zone for TV/soundbar/music requests
SONOS_TTS_OUTPUT = True   # Allow clients to route TTS through Sonos (per-client opt-in via prefer_sonos_output)
SERVER_EXTERNAL_HOST = os.getenv("SERVER_EXTERNAL_HOST", "192.168.0.4")  # PC's LAN IP (Sonos fetches TTS audio)

# LIFX smart bulb settings (local LAN UDP, no API key)
LIFX_DISCOVERY_CACHE_TTL = 300  # seconds before forced re-discovery (5 min)

# Room groups — map room names to lists of light labels for natural language targeting
LIGHT_GROUPS: dict = {
    "living room": ["corner lamp", "table lamp", "tall lamp"],
    "office":      ["office lamp"],
}

# Named lighting scenes — each entry maps light labels (or "*" for all) to settings dicts.
# Settings keys: brightness (0-1 float), kelvin (int), power (bool, default True)
LIGHT_SCENES: dict = {
    "warm mix": {
        "corner lamp": {"brightness": 0.70, "kelvin": 2700},
        "table lamp":  {"brightness": 0.50, "kelvin": 2700},
        "tall lamp":   {"brightness": 0.60, "kelvin": 2700},
        "office lamp": {"brightness": 0.40, "kelvin": 2700},
    },
    "bright": {
        "*": {"brightness": 1.0, "kelvin": 5000},
    },
    "evening": {
        "*": {"brightness": 0.45, "kelvin": 2700},
    },
    "movie": {
        "corner lamp": {"brightness": 0.25, "kelvin": 2200},
        "table lamp":  {"brightness": 0.20, "kelvin": 2200},
        "tall lamp":   {"power": False},
        "office lamp": {"power": False},
    },
    "focus": {
        "office lamp": {"brightness": 1.0, "kelvin": 5000},
        "corner lamp": {"power": False},
        "table lamp":  {"power": False},
        "tall lamp":   {"power": False},
    },
}

# Google TV remote settings
GOOGLE_TV_HOST = os.getenv("GOOGLE_TV_HOST", "192.168.0.20")
GOOGLE_TV_CERT_FILE = str(DATA_DIR / "google_tv_cert.pem")
GOOGLE_TV_KEY_FILE  = str(DATA_DIR / "google_tv_key.pem")
GOOGLE_TV_CLIENT_NAME = "Igor"
ADB_KEY_FILE = str(DATA_DIR / "adbkey")
