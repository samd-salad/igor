"""API endpoint path constants shared between client and server."""

# Server (PC) endpoints
PROCESS_INTERACTION_ENDPOINT = "/api/process_interaction"
TEXT_INTERACTION_ENDPOINT = "/api/text_interaction"
REGISTER_CLIENT_ENDPOINT = "/api/register"
SERVER_HEALTH_ENDPOINT = "/api/health"
SONOS_BEEP_ENDPOINT = "/api/sonos_beep"

# Client (Pi) endpoints
PLAY_AUDIO_ENDPOINT = "/api/play_audio"
HARDWARE_CONTROL_ENDPOINT = "/api/hardware_control"
PLAY_BEEP_ENDPOINT = "/api/play_beep"
SUPPRESS_WAKEWORD_ENDPOINT = "/api/suppress_wakeword"
CLIENT_HEALTH_ENDPOINT = "/api/health"

# Default timeout for Pi callback HTTP requests
REQUEST_TIMEOUT = 5.0
