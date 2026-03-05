"""Wake word suppression — temporarily silence detection after TV activity.

When Igor executes a TV command (power on, launch app, playback), the TV's
startup audio or content audio can retrigger the wake word detector within
seconds.  To prevent this, the server sends a suppress_wakeword RPC to the Pi
after every TV command, setting a future timestamp here.  The main loop and
detection loop both check is_suppressed() before acting.

Key design choices:
  - Module-level singleton (no class needed — there's only one mic on the Pi).
  - suppress() uses max() so calling it again never *shortens* an active window.
  - unsuppress() hard-resets to 0.0 so TTS completion can clear it immediately,
    regardless of however much time was originally requested.  This is separate
    from suppress(0) which would call max(current, 0) and never shorten.
  - All mutations guarded by a threading.Lock because the Flask callback server
    calls suppress() from its HTTP thread while the main loop reads is_suppressed()
    from the wake word thread.
"""
import threading
import time

# Thread lock protecting _suppress_until — Flask thread writes, main loop reads.
_lock = threading.Lock()

# Unix timestamp after which wake word detection is allowed again.
# 0.0 means "not suppressed" (time.time() is always > 0).
_suppress_until: float = 0.0


def suppress(seconds: float):
    """Suppress wake word detection for `seconds` seconds from now.

    If suppression is already active, this extends it — it never shortens.
    Called after TV commands to prevent startup audio from immediately
    reactivating the assistant.

    Args:
        seconds: Duration to suppress from this moment.
    """
    global _suppress_until
    with _lock:
        # max() ensures we only ever *extend* an active suppression window,
        # never accidentally shorten one that was set with a longer duration.
        _suppress_until = max(_suppress_until, time.time() + seconds)


def unsuppress():
    """Clear any active suppression immediately.

    Called as soon as TTS audio finishes playing so the user can talk again
    without waiting for the full suppression window to expire.

    NOT equivalent to suppress(0): suppress(0) calls max(current, now+0) which
    leaves an active window unchanged.  This function hard-resets to 0.0.
    """
    global _suppress_until
    with _lock:
        _suppress_until = 0.0


def is_suppressed() -> bool:
    """Return True if wake word detection is currently suppressed.

    Checked at the top of _handle_interaction() to skip entire detection
    cycles, and also inside the detection loop so mid-stream suppression
    (from a TV command mid-utterance) resets immediately.
    """
    with _lock:
        return time.time() < _suppress_until
