#!/usr/bin/env bash
set -e

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-server.txt

mkdir -p voices
curl -L -o voices/en_US-arctic-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx
curl -L -o voices/en_US-arctic-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json

echo "Server setup complete. Set ANTHROPIC_API_KEY and run: python -m server.main"
