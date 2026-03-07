"""Configuration for Raspberry Pi client."""
import os
from pathlib import Path

# Server (PC) settings
SERVER_HOST = os.getenv("SERVER_HOST", "192.168.0.4")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Pi server settings (for callbacks from PC) — bind to LAN IP only (not 0.0.0.0)
CLIENT_HOST = os.getenv("CLIENT_HOST", "192.168.0.3")
CLIENT_PORT = 8080

# Audio hardware settings
SAMPLE_RATE = 16000
AUDIO_DEVICE = "plughw:2,0"  # Run 'arecord -L' to find yours

# Paths
BASE_DIR = Path(__file__).parent.parent
TEMP_WAV = str(BASE_DIR / "data" / "recording.wav")  # local path avoids /tmp race conditions

# OpenWakeWord custom models
# Train models with train_wakeword.py on your PC, then copy .onnx files to oww_models/
OWW_MODEL_DIR = BASE_DIR / "oww_models"
OWW_MODEL_PATHS = [str(p) for p in sorted(OWW_MODEL_DIR.glob("*.onnx"))]
OWW_THRESHOLD = 0.75  # Detection threshold (0–1). Higher = fewer false positives.
OWW_TRIGGER_FRAMES = 5  # Consecutive frames above threshold required to trigger.
                         # Each frame is ~80ms (1280 samples at 16kHz).
                         # Increase to suppress false positives from brief noisy spikes.
OWW_MIN_RMS = 500        # Minimum RMS of detection audio to confirm wake word.
                         # Rejects low-energy false positives (TV audio, room noise).
                         # Set to 0 to disable. Close-mic speech is typically 800-4000+.

# Auto-save detected wake words as training samples
# Saves the audio that triggered each detection to wakeword_samples/positive/
OWW_AUTO_SAVE_SAMPLES = True
OWW_SAMPLE_BUFFER_SECONDS = 2.5  # How much audio before detection to capture
WAKE_SAMPLES_DIR = BASE_DIR / "wakeword_samples" / "positive"

# Voice Activity Detection (VAD) settings
#
# RMS_SILENCE_THRESHOLD is the MINIMUM floor — the VAD also calibrates ambient
# noise at the start of each recording and uses whichever is higher:
#   effective = max(RMS_SILENCE_THRESHOLD, ambient_rms * 2.0)
# This auto-adapts to mic gain and room noise.  The static value below is a
# safety net for unusually quiet environments.
#
# SILENCE_END_DURATION: consecutive seconds below threshold to end recording.
# 1.5s catches natural sentence endings without cutting off mid-thought pauses
# (most inter-phrase pauses are 0.3-0.8s; 1.5s requires sustained silence).
# Short commands (<1s speech) use 0.6s instead (see vad_recorder.py).
SILENCE_END_DURATION = 1.5
RMS_SILENCE_THRESHOLD = 1000
MIN_RECORDING = 0.7
MAX_RECORDING = 15

# Follow-up mode settings
# How long to wait for user to START speaking after a follow-up prompt.
# 10s gives the user time to think, read the TV, or get distracted without
# the conversation silently ending.  Old value of 5s caused frequent
# "follow-up timed out" when users paused to think.
FOLLOWUP_TIMEOUT = 10.0

# Audio output routing
USE_SONOS_OUTPUT = False  # Set True to route TTS through Sonos instead of Pi speaker
INDICATOR_LIGHT = None    # LIFX light label to flash as listening indicator when TV is playing (e.g. "corner lamp")

# Request timeouts
REQUEST_TIMEOUT = 60.0
