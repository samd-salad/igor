"""PC client configuration. Runs on the same machine as the server."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Network — PC is both server and client, different ports
SERVER_HOST = os.getenv("SERVER_HOST", "192.168.0.4")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Callback server — must use LAN IP (not localhost) for registration validation
CLIENT_HOST = os.getenv("CLIENT_HOST", SERVER_HOST)
CLIENT_PORT = int(os.getenv("PC_CLIENT_PORT", "8081"))
CLIENT_ID = os.getenv("CLIENT_ID", "office_pc")
ROOM_ID = os.getenv("ROOM_ID", "office")

# Audio devices — BlackShark V3 Pro defaults
# Set AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE to override
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", None)  # None = default
AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", None)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT_BITS = 16  # paInt16

# Recording
SILENCE_THRESHOLD = 100  # Peak-based (BlackShark silence=17-27, speech=333+)
SILENCE_DURATION = 2.0
MIN_RECORDING = 0.5
MAX_RECORDING = 30

# Wake word
OWW_MODELS_DIR = BASE_DIR / "oww_models"
OWW_CHUNK = 1280  # 80ms frames for OpenWakeWord
OWW_THRESHOLD = float(os.getenv("OWW_THRESHOLD", "0.8"))
OWW_TRIGGER_FRAMES = int(os.getenv("OWW_TRIGGER_FRAMES", "5"))

# Audio normalization for low-gain mics
NORMALIZE_TARGET_PEAK = 16000
NORMALIZE_FLOOR = 50  # Below this peak, treat as silence

# Speech detection for wake word gating
SPEECH_PEAK_MIN = 100
SPEECH_TRAIL_FRAMES = 10  # Frames after last speech to keep "active"
DIP_SCORE = 0.3  # Score must dip below this before counting
MAX_GAP_FRAMES = 3  # Silence frames allowed between speech frames

# Wake word sample management
OWW_AUTO_SAVE_SAMPLES = True
OWW_MAX_AUTO_SAMPLES = 200
OWW_SAMPLE_BUFFER_SECONDS = 2.5
WAKE_SAMPLES_DIR = BASE_DIR / "wakeword_samples" / "positive"

# Output
USE_SONOS_OUTPUT = False
TEMP_WAV = BASE_DIR / "data" / "pc_recording.wav"

# Follow-up
MAX_FOLLOWUP_DEPTH = 5
FOLLOWUP_TIMEOUT = 10.0  # Seconds to wait for follow-up speech
