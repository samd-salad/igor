#!/usr/bin/env bash
set -e

# Python environment
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-server.txt

# Kokoro TTS voice model
KOKORO_DIR="kokoro"
if [ ! -f "$KOKORO_DIR/kokoro-v1.0.onnx" ]; then
    echo "Kokoro model not found in $KOKORO_DIR/."
    echo "Download kokoro-v1.0.onnx and voices-v1.0.bin from:"
    echo "  https://github.com/thewh1teagle/kokoro-onnx/releases"
    echo "Place them in the $KOKORO_DIR/ directory."
else
    echo "Kokoro TTS model already present."
fi

# Room configuration
if [ ! -f "data/rooms.yaml" ]; then
    echo ""
    echo "No data/rooms.yaml found — using default single-room config."
    echo "Copy data/rooms.yaml.example to data/rooms.yaml to configure multiple rooms."
fi

echo ""
echo "Server setup complete."
echo "  1. Set ANTHROPIC_API_KEY environment variable"
echo "  2. Edit server/config.py (PI_HOST, etc.)"
echo "  3. Run: python -m server.main"
