"""Wake word suppression — temporarily silence detection after TV activity."""
import threading
import time

_lock = threading.Lock()
_suppress_until: float = 0.0


def suppress(seconds: float):
    """Suppress wake word detection for `seconds` seconds from now."""
    global _suppress_until
    with _lock:
        _suppress_until = max(_suppress_until, time.time() + seconds)


def unsuppress():
    """Clear any active suppression immediately."""
    global _suppress_until
    with _lock:
        _suppress_until = 0.0


def is_suppressed() -> bool:
    """Return True if wake word detection is currently suppressed."""
    with _lock:
        return time.time() < _suppress_until
