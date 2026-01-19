"""API protocol definitions and endpoint constants."""

# Server (PC) endpoints
SERVER_HOST = "192.168.0.4"
SERVER_PORT = 8000
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Server API endpoints
PROCESS_INTERACTION_ENDPOINT = "/api/process_interaction"
SERVER_HEALTH_ENDPOINT = "/api/health"

# Client (Pi) endpoints
CLIENT_HOST = "192.168.0.3"
CLIENT_PORT = 8080
CLIENT_BASE_URL = f"http://{CLIENT_HOST}:{CLIENT_PORT}"

# Client API endpoints
PLAY_AUDIO_ENDPOINT = "/api/play_audio"
HARDWARE_CONTROL_ENDPOINT = "/api/hardware_control"
PLAY_BEEP_ENDPOINT = "/api/play_beep"
CLIENT_HEALTH_ENDPOINT = "/api/health"

# Timeout settings
REQUEST_TIMEOUT = 5.0  # seconds for HTTP requests
AUDIO_UPLOAD_TIMEOUT = 10.0  # seconds for uploading audio to server
AUDIO_DOWNLOAD_TIMEOUT = 10.0  # seconds for downloading audio from server
