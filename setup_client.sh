#!/usr/bin/env bash
set -e

echo "=== Igor Client Setup (Raspberry Pi) ==="
echo ""

# ---- Pre-flight checks ----
# Must run from the repo directory
if [ ! -f "requirements-client.txt" ]; then
    echo "ERROR: Run this script from the smart_assistant repo directory."
    echo ""
    echo "If you haven't cloned the repo yet:"
    echo "  git clone <your-repo-url> ~/smart_assistant"
    echo "  cd ~/smart_assistant"
    echo "  bash setup_client.sh"
    exit 1
fi

# Check Python version (need 3.11+)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
if [ "$(echo "$PYTHON_VERSION < 3.11" | bc -l 2>/dev/null || echo 1)" = "1" ] && [ "$PYTHON_VERSION" != "0" ]; then
    # bc might not be installed, just check if python3 works at all
    python3 --version
fi
echo "Python: $(python3 --version)"

# ---- System dependencies ----
echo ""
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq portaudio19-dev libasound2-dev sox

# ---- Python environment ----
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi
echo "Installing/updating Python dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements-client.txt -q
# Install openwakeword without tflite-runtime (no Python 3.13 build for aarch64).
# We use inference_framework="onnx" so only onnxruntime is needed, not tflite.
.venv/bin/pip install openwakeword --no-deps -q

# ---- Pre-download OpenWakeWord base models ----
echo "Pre-downloading OpenWakeWord base models (~50MB)..."
.venv/bin/python -c "
from openwakeword.utils import download_models
download_models()
print('Base models ready.')
"

# ---- Directories ----
mkdir -p data
mkdir -p oww_models

# ---- Configuration ----
echo ""
echo "=== Configuration ==="
echo "Press Enter to accept defaults shown in [brackets]."
echo ""

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
arecord -L 2>/dev/null | grep -E "^plughw:|^hw:|^default" | head -10 || echo "  (none found — run 'arecord -L' manually)"
echo ""
read -p "Audio capture device [plughw:2,0]: " AUDIO_INPUT
AUDIO_DEVICE="${AUDIO_INPUT:-plughw:2,0}"

# Test audio device
echo "Testing audio device..."
if arecord -D "$AUDIO_DEVICE" -d 1 -f S16_LE -r 16000 /dev/null 2>/dev/null; then
    echo "Audio device OK."
else
    echo "WARNING: Could not open $AUDIO_DEVICE. You may need to adjust AUDIO_DEVICE in client/config.py."
    echo "Run 'arecord -L' to list available devices."
fi

# Room identity
echo ""
echo "Room ID must match a key in data/rooms.yaml on the server."
echo "Examples: living_room, bedroom, office, default"
read -p "Room ID [default]: " ROOM_INPUT
ROOM_ID="${ROOM_INPUT:-default}"
CLIENT_ID="${ROOM_ID}"

# Sonos output
echo ""
read -p "Route TTS through Sonos instead of Pi speaker? [y/N] " SONOS_INPUT
if [[ "$SONOS_INPUT" =~ ^[Yy]$ ]]; then
    USE_SONOS="True"
    read -p "LIFX indicator light label for beep flashes (blank for audio beeps): " INDICATOR_INPUT
    INDICATOR_LIGHT="${INDICATOR_INPUT:-None}"
    if [ "$INDICATOR_LIGHT" != "None" ]; then
        INDICATOR_LIGHT="\"$INDICATOR_LIGHT\""
    fi
else
    USE_SONOS="False"
    INDICATOR_LIGHT="None"
fi

# ---- Write config overrides ----
CONFIG_FILE="client/config.py"
echo ""
echo "Updating $CONFIG_FILE..."

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

echo "Config: SERVER=$SERVER_IP  PI=$PI_IP  AUDIO=$AUDIO_DEVICE  ROOM=$ROOM_ID  SONOS=$USE_SONOS"

# ---- Wake word models ----
echo ""
if [ -z "$(ls oww_models/*.onnx 2>/dev/null)" ]; then
    echo "No wake word models found in oww_models/."
    echo "Copy from PC:  scp user@$SERVER_IP:~/smart_assistant/oww_models/*.onnx oww_models/"
    read -p "Do this now? [Y/n] " COPY_MODELS
    if [[ ! "$COPY_MODELS" =~ ^[Nn]$ ]]; then
        read -p "PC username [$(whoami)]: " PC_USER_INPUT
        PC_USER="${PC_USER_INPUT:-$(whoami)}"
        echo "Copying wake word models from $PC_USER@$SERVER_IP..."
        scp "$PC_USER@$SERVER_IP:~/smart_assistant/oww_models/*.onnx" oww_models/ || {
            echo "WARNING: SCP failed. Copy models manually before starting."
        }
    fi
else
    echo "Wake word models found: $(ls oww_models/*.onnx | wc -l) model(s)"
fi

# ---- Systemd service ----
if [ -d "/etc/systemd/system" ]; then
    echo ""
    read -p "Set up systemd service for auto-start on boot? [Y/n] " SETUP_SERVICE
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

        echo ""
        read -p "Start the service now? [Y/n] " START_NOW
        if [[ ! "$START_NOW" =~ ^[Nn]$ ]]; then
            sudo systemctl start igor-client
            sleep 2
            systemctl status igor-client --no-pager || true
        else
            echo "Start later with: sudo systemctl start igor-client"
        fi
        echo "View logs: sudo journalctl -u igor-client -f"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Manual start:  CLIENT_ID=$CLIENT_ID ROOM_ID=$ROOM_ID .venv/bin/python -m client.main"
echo "Health check:  curl http://$PI_IP:8080/api/health"
echo "Server check:  curl http://$SERVER_IP:8000/api/health"
