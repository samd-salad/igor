"""Public contract for Igor's wake-word model. Both the training pipeline AND
the runtime systemd unit read from this file. If you change any constant here,
re-render the runtime config (`python -m wakeword.render_runtime`) and retrain
in the same commit.
"""
from __future__ import annotations

# Feature pipeline (lives in wyoming-openwakeword's runtime; training mirrors it)
FEATURE_LIBRARY = "pyopen_wakeword"
FEATURE_LIBRARY_VERSION = ">=1.1,<2"
FEATURE_DIM = 96
FEATURE_RATE_HZ = 42.7

# Wake-word classifier model
MODEL_INPUT_SHAPE = (1, 16, 96)
MODEL_INPUT_DTYPE = "float32"
MODEL_OUTPUT_SHAPE = (1, 1)
MODEL_OUTPUT_RANGE = (0.0, 1.0)

# wyoming-openwakeword expects this filename pattern in its custom-models dir
MODEL_FILENAME_PATTERN = "{name}_v{version}.tflite"

# Default tuning (override in deploy/wyoming-openwakeword.service if needed)
DEFAULT_THRESHOLD = 0.5
DEFAULT_TRIGGER_LEVEL = 3
DEFAULT_REFRACTORY_SECONDS = 2.0
