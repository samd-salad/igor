#!/usr/bin/env bash
set -e

echo "=== Igor Server Setup (PC) ==="
echo ""

# ---- Python environment ----
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi
echo "Installing/updating dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements-server.txt -q

# ---- Data directory ----
mkdir -p data

# ---- Kokoro TTS voice model ----
KOKORO_DIR="kokoro"
mkdir -p "$KOKORO_DIR"
if [ ! -f "$KOKORO_DIR/kokoro-v1.0.onnx" ]; then
    echo ""
    echo "WARNING: Kokoro model not found in $KOKORO_DIR/."
    echo "Download kokoro-v1.0.onnx and voices-v1.0.bin from:"
    echo "  https://github.com/thewh1teagle/kokoro-onnx/releases"
    echo "Place them in the $KOKORO_DIR/ directory."
else
    echo "Kokoro TTS model present."
fi

# ---- Room configuration ----
if [ ! -f "data/rooms.yaml" ]; then
    echo ""
    echo "No data/rooms.yaml found — using default single-room config."
    echo "Copy data/rooms.yaml.example to data/rooms.yaml to configure multiple rooms."
fi

# ---- ANTHROPIC_API_KEY ----
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "WARNING: ANTHROPIC_API_KEY is not set."
    echo "Add it to your shell profile or .env file:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
fi

# ---- Systemd service (Linux only) ----
if [ -d "/etc/systemd/system" ]; then
    echo ""
    read -p "Set up systemd service for auto-start? [y/N] " SETUP_SERVICE
    if [[ "$SETUP_SERVICE" =~ ^[Yy]$ ]]; then
        WORK_DIR="$(pwd)"
        PYTHON_PATH="$WORK_DIR/.venv/bin/python"
        CURRENT_USER="$(whoami)"

        # Prompt for PI_HOST
        read -p "Pi IP address [192.168.0.3]: " PI_HOST_INPUT
        PI_HOST="${PI_HOST_INPUT:-192.168.0.3}"

        sudo bash -c "cat > /etc/systemd/system/igor-server.service << SERVICEEOF
[Unit]
Description=Igor Voice Assistant Server
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$WORK_DIR
Environment=\"ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}\"
Environment=\"PI_HOST=${PI_HOST}\"
ExecStart=$PYTHON_PATH -m server.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF"

        sudo systemctl daemon-reload
        sudo systemctl enable igor-server
        echo "Service installed. Start with: sudo systemctl start igor-server"
        echo "Logs: sudo journalctl -u igor-server -f"
    fi
fi

echo ""
echo "=== Server setup complete ==="
echo ""
echo "Quick start:  source .venv/bin/activate && python -m server.main"
echo "Health check: curl http://\$(hostname -I | awk '{print \$1}'):8000/api/health"
