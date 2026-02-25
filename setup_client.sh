#!/usr/bin/env bash
set -e

# System dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Python environment
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-client.txt

# Sherpa-ONNX wake word model
MODEL_DIR="sherpa_onnx_models"
MODEL_TAR="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$MODEL_TAR"

if [ ! -f "$MODEL_DIR/tokens.txt" ]; then
    echo "Downloading Sherpa-ONNX model..."
    mkdir -p "$MODEL_DIR"
    wget -O "/tmp/$MODEL_TAR" "$MODEL_URL"
    tar xf "/tmp/$MODEL_TAR" --strip-components=1 -C "$MODEL_DIR"
    rm "/tmp/$MODEL_TAR"
else
    echo "Sherpa-ONNX model already present, skipping."
fi

echo "Client setup complete. Run: python -m client.main"
