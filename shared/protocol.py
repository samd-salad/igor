"""API endpoint path constants shared between client and server."""

# Server (PC) endpoints
PROCESS_INTERACTION_ENDPOINT = "/api/process_interaction"
SERVER_HEALTH_ENDPOINT = "/api/health"

# Client (Pi) endpoints
PLAY_AUDIO_ENDPOINT = "/api/play_audio"
HARDWARE_CONTROL_ENDPOINT = "/api/hardware_control"
PLAY_BEEP_ENDPOINT = "/api/play_beep"
CLIENT_HEALTH_ENDPOINT = "/api/health"

# Default timeout for Pi callback HTTP requests
REQUEST_TIMEOUT = 5.0
