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

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMP_WAV = "/tmp/recording.wav"

# Sherpa-ONNX keyword spotting
# Download model: https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
# Recommended: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
# Extract contents into sherpa_onnx_models/
SHERPA_MODEL_DIR = str(BASE_DIR / "sherpa_onnx_models")
WAKE_WORDS = ["doctor butts", "stop"]
WAKE_THRESHOLD = 0.25  # Lower = more sensitive (more false positives)

# Voice Activity Detection (VAD) settings
SILENCE_END_DURATION = 2.0
RMS_SILENCE_THRESHOLD = 1200
MIN_RECORDING = 0.7
MAX_RECORDING = 15

# Follow-up mode settings
FOLLOWUP_TIMEOUT = 5.0

# Request timeouts
REQUEST_TIMEOUT = 60.0
