#!/usr/bin/env bash
set -e

echo "=== Igor Client Setup (Raspberry Pi) ==="
echo ""

# ---- System dependencies ----
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq portaudio19-dev libasound2-dev sox

# ---- Python environment ----
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi
echo "Installing/updating dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements-client.txt -q
# Install openwakeword without tflite-runtime (no Python 3.13 build for aarch64).
# We use inference_framework="onnx" so only onnxruntime is needed, not tflite.
.venv/bin/pip install openwakeword --no-deps -q

# ---- Pre-download OpenWakeWord base models ----
echo "Pre-downloading OpenWakeWord base models..."
.venv/bin/python -c "
from openwakeword.utils import download_models
download_models()
print('Base models ready.')
"

# ---- Data directory ----
mkdir -p data

# ---- Configuration ----
echo ""
echo "=== Configuration ==="

# Server IP
read -p "Server (PC) IP address [192.168.0.4]: " SERVER_IP_INPUT
SERVER_IP="${SERVER_IP_INPUT:-192.168.0.4}"

# This Pi's IP
DEFAULT_PI_IP=$(hostname -I | awk '{print $1}')
read -p "This Pi's IP address [$DEFAULT_PI_IP]: " PI_IP_INPUT
PI_IP="${PI_IP_INPUT:-$DEFAULT_PI_IP}"

# Audio device
echo ""
echo "Available audio capture devices:"
arecord -L 2>/dev/null | grep -E "^plughw:|^hw:|^default" || echo "  (run 'arecord -L' to list)"
read -p "Audio device [plughw:2,0]: " AUDIO_INPUT
AUDIO_DEVICE="${AUDIO_INPUT:-plughw:2,0}"

# Room identity
read -p "Room ID (must match rooms.yaml on server) [default]: " ROOM_INPUT
ROOM_ID="${ROOM_INPUT:-default}"
CLIENT_ID="${ROOM_ID}"

# Sonos output
read -p "Route TTS through Sonos? [y/N] " SONOS_INPUT
if [[ "$SONOS_INPUT" =~ ^[Yy]$ ]]; then
    USE_SONOS="True"
    read -p "LIFX indicator light label (blank for none): " INDICATOR_INPUT
    INDICATOR_LIGHT="${INDICATOR_INPUT:-None}"
    if [ "$INDICATOR_LIGHT" != "None" ]; then
        INDICATOR_LIGHT="\"$INDICATOR_LIGHT\""
    fi
else
    USE_SONOS="False"
    INDICATOR_LIGHT="None"
fi

# ---- Write config overrides ----
# Only override values that differ from defaults
CONFIG_FILE="client/config.py"
if [ "$SERVER_IP" != "192.168.0.4" ]; then
    sed -i "s|SERVER_HOST = os.getenv(\"SERVER_HOST\", \"192.168.0.4\")|SERVER_HOST = os.getenv(\"SERVER_HOST\", \"$SERVER_IP\")|" "$CONFIG_FILE"
fi
if [ "$PI_IP" != "192.168.0.3" ]; then
    sed -i "s|CLIENT_HOST = os.getenv(\"CLIENT_HOST\", \"192.168.0.3\")|CLIENT_HOST = os.getenv(\"CLIENT_HOST\", \"$PI_IP\")|" "$CONFIG_FILE"
fi
if [ "$AUDIO_DEVICE" != "plughw:2,0" ]; then
    sed -i "s|AUDIO_DEVICE = \"plughw:2,0\"|AUDIO_DEVICE = \"$AUDIO_DEVICE\"|" "$CONFIG_FILE"
fi
if [ "$USE_SONOS" = "True" ]; then
    sed -i "s|USE_SONOS_OUTPUT = False|USE_SONOS_OUTPUT = True|" "$CONFIG_FILE"
    if [ "$INDICATOR_LIGHT" != "None" ]; then
        sed -i "s|INDICATOR_LIGHT = None|INDICATOR_LIGHT = $INDICATOR_LIGHT|" "$CONFIG_FILE"
    fi
fi

echo ""
echo "Config updated: SERVER=$SERVER_IP, PI=$PI_IP, AUDIO=$AUDIO_DEVICE, ROOM=$ROOM_ID"

# ---- Systemd service ----
if [ -d "/etc/systemd/system" ]; then
    echo ""
    read -p "Set up systemd service for auto-start? [Y/n] " SETUP_SERVICE
    if [[ ! "$SETUP_SERVICE" =~ ^[Nn]$ ]]; then
        WORK_DIR="$(pwd)"
        PYTHON_PATH="$WORK_DIR/.venv/bin/python"
        CURRENT_USER="$(whoami)"

        sudo bash -c "cat > /etc/systemd/system/igor-client.service << SERVICEEOF
[Unit]
Description=Igor Voice Assistant Client
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$WORK_DIR
Environment=\"CLIENT_ID=$CLIENT_ID\"
Environment=\"ROOM_ID=$ROOM_ID\"
ExecStart=$PYTHON_PATH -m client.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF"

        sudo systemctl daemon-reload
        sudo systemctl enable igor-client
        echo "Service installed. Start with: sudo systemctl start igor-client"
        echo "Logs: sudo journalctl -u igor-client -f"
    fi
fi

echo ""
echo "=== Client setup complete ==="
echo ""
if [ -z "$(ls oww_models/*.onnx 2>/dev/null)" ]; then
    echo "Next: copy wake word models from PC:"
    echo "  scp pc:smart_assistant/oww_models/*.onnx oww_models/"
    echo ""
fi
echo "Quick start:  CLIENT_ID=$CLIENT_ID ROOM_ID=$ROOM_ID .venv/bin/python -m client.main"
echo "Health check: curl http://$PI_IP:8080/api/health"
