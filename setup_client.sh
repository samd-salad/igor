#!/usr/bin/env bash
set -e

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-client.txt

mkdir -p sherpa_onnx_models && cd sherpa_onnx_models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
tar xf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2 --strip-components=1
rm sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
cd ..

echo "Client setup complete. Run: python -m client.main"
