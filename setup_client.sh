#!/usr/bin/env bash
set -e

# System dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Python environment
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-client.txt

# Pre-download OpenWakeWord base models (melspectrogram + embedding)
# These are ~50MB and download automatically on first Model() init,
# but pre-fetching here avoids latency on first run.
echo "Pre-downloading OpenWakeWord base models..."
.venv/bin/python -c "
from openwakeword.utils import download_models
download_models()
print('Base models ready.')
"

echo ""
echo "Client setup complete."
echo ""
echo "Next steps:"
echo "  1. Record wake word samples: python record_samples.py"
echo "  2. Transfer wakeword_samples/ to PC and run train_wakeword.py"
echo "  3. Copy oww_models/*.onnx back here"
echo "  4. Run: .venv/bin/python -m client.main"
