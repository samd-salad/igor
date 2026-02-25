#!/usr/bin/env bash
set -e

# Python environment
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-server.txt

# Piper voice model
VOICE_DIR="voices"
VOICE_BASE="en_US-arctic-medium"
HF_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium"

if [ ! -f "$VOICE_DIR/$VOICE_BASE.onnx" ]; then
    echo "Downloading Piper voice model..."
    mkdir -p "$VOICE_DIR"
    curl -L -o "$VOICE_DIR/$VOICE_BASE.onnx" "$HF_URL/$VOICE_BASE.onnx"
    curl -L -o "$VOICE_DIR/$VOICE_BASE.onnx.json" "$HF_URL/$VOICE_BASE.onnx.json"
else
    echo "Piper voice model already present, skipping."
fi

echo "Server setup complete. Set ANTHROPIC_API_KEY and run: python -m server.main"
