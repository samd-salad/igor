"""Configuration for Raspberry Pi client."""
import os
from pathlib import Path

# Server (PC) settings
SERVER_HOST = "192.168.0.4"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Pi server settings (for callbacks from PC)
CLIENT_HOST = "0.0.0.0"  # Listen on all interfaces
CLIENT_PORT = 8080

# Audio hardware settings
SAMPLE_RATE = 16000
AUDIO_DEVICE = "plughw:2,0"  # ALSA hardware device

# Wake word settings
WAKE_WORDS = ["doctor_butts"]
WAKE_THRESHOLD = 0.1  # Lower = triggers more easily

# Voice Activity Detection (VAD) settings
SILENCE_END_DURATION = 2.0  # Seconds of silence to stop recording
RMS_SILENCE_THRESHOLD = 1200  # RMS value below which audio is silence
MIN_RECORDING = 0.7  # Minimum recording duration in seconds
MAX_RECORDING = 15  # Maximum recording duration in seconds

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"  # Wake word models
OWW_MODELS_DIR = BASE_DIR / "oww_models"  # OpenWakeWord models
TEMP_WAV = "/tmp/recording.wav"  # Temporary recording file

# Request timeouts
REQUEST_TIMEOUT = 60.0  # Timeout for server requests (STT + LLM + TTS can take a while)
