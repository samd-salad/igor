"""Configuration for Raspberry Pi client."""
import os
from pathlib import Path

# Server (PC) settings
SERVER_HOST = "192.168.0.4"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Pi server settings (for callbacks from PC)
CLIENT_HOST = "0.0.0.0"
CLIENT_PORT = 8080

# Audio hardware settings
SAMPLE_RATE = 16000
AUDIO_DEVICE = "plughw:2,0"  # Run 'arecord -L' to find yours

# Porcupine wake word detection
# Get a free access key at console.picovoice.ai
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY", "")

# Download .ppn files from console.picovoice.ai for your Pi's CPU:
#   Pi 4 → Cortex-A72    Pi 3 → Cortex-A53
PORCUPINE_KEYWORD_PATHS = [
    str(Path(__file__).parent.parent / "porcupine_models" / "doctor-butts_en_raspberry-pi_v3_0_0.ppn"),
    str(Path(__file__).parent.parent / "porcupine_models" / "stop_en_raspberry-pi_v3_0_0.ppn"),
]
PORCUPINE_SENSITIVITIES = [0.5, 0.5]  # Per-keyword, 0.0–1.0 (higher = more sensitive)

# Voice Activity Detection (VAD) settings
SILENCE_END_DURATION = 2.0
RMS_SILENCE_THRESHOLD = 1200
MIN_RECORDING = 0.7
MAX_RECORDING = 15

# Follow-up mode settings
FOLLOWUP_TIMEOUT = 5.0

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMP_WAV = "/tmp/recording.wav"

# Request timeouts
REQUEST_TIMEOUT = 60.0
