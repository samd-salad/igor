"""Confirm wakeword contract constants are present and consistent."""
from wakeword import contracts


def test_constants_present():
    assert contracts.FEATURE_LIBRARY == "pyopen_wakeword"
    assert contracts.FEATURE_DIM == 96
    assert contracts.MODEL_INPUT_SHAPE == (1, 16, 96)
    assert contracts.MODEL_OUTPUT_SHAPE == (1, 1)
    assert contracts.MODEL_OUTPUT_RANGE == (0.0, 1.0)
    assert "{name}" in contracts.MODEL_FILENAME_PATTERN
    assert "{version}" in contracts.MODEL_FILENAME_PATTERN
    assert contracts.FEATURE_RATE_HZ > 0
