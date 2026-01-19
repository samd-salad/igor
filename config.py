import os
import subprocess
from pathlib import Path

# LLM configuration - local Ollama instance
OLLAMA_URL = "http://192.168.0.4:11434"
OLLAMA_MODEL = "qwen3:30b"

# Whisper model - speech recognition accuracy vs speed
# tiny = fastest, least accurate | base = balanced | small = slower, more accurate
WHISPER_MODEL = "base"

# Wake word detection
WAKE_WORDS = ["doctor_butts"]
WAKE_THRESHOLD = 0.1              # Confidence score (0.0-1.0) required to trigger wake word
                                  # Lower = triggers more easily, Higher = requires clearer match

# Audio hardware
SAMPLE_RATE = 16000               # Audio samples per second sent to speech recognition
AUDIO_DEVICE = "plughw:2,0"       # ALSA hardware device identifier for mic/speaker

# Silence detection - controls when recording starts and stops
SILENCE_END_DURATION = 2.0       # Seconds of silence required to STOP recording
RMS_SILENCE_THRESHOLD = 1000     # RMS value below which audio is considered silence
                                 # Typical speech: 1000-10000, silence: 50-200
                                 # Adjust based on your microphone gain
MIN_RECORDING = 0.7              # Minimum recording duration in seconds
MAX_RECORDING = 15               # Hard limit in seconds - recording stops regardless of sound

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"  # Directory containing wake word .onnx files
PIPER_VOICE = os.path.expanduser("~/.local/share/piper-voices/en_US-arctic-medium.onnx")  # TTS voice model
TEMP_WAV = "/tmp/recording.wav"   # Temporary file where recordings are saved before transcription

# Weather (requires OPENWEATHERMAP_API_KEY environment variable)
# Also set DEFAULT_LOCATION env var for default city (e.g., "Seattle, WA")