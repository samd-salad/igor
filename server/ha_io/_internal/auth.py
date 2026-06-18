"""X-Igor-Token check. No-op when IGOR_API_TOKEN env var is unset (dev mode)."""
from __future__ import annotations
import os


def check_token(provided: str | None) -> bool:
    expected = os.environ.get("IGOR_API_TOKEN", "")
    if not expected:
        return True
    return provided == expected
