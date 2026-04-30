"""Background event loop for timers and scheduled events.

Architecture:
  - A single background thread (_run) polls for expired timers every 0.5s.
  - When a timer fires, it spawns a daemon thread for the per-timer callback
    so HA calls or other I/O don't block the polling loop.
  - Timer state is protected by a single lock.
  - Audio delivery is NOT this module's concern — HA owns STT/TTS. Callbacks
    can call HA services (notify.*, assist_satellite.announce, etc.) to alert
    the user if desired.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Timer data structure: name, fire time, optional callback, optional room."""
    name: str
    fire_at: float
    callback: Optional[Callable[[str], None]] = None
    room_id: Optional[str] = None


class EventLoop:
    """Background event loop for scheduled timers."""

    def __init__(self):
        self._timers: dict[str, Timer] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        logger.info("EventLoop initialized")

    def start(self):
        if self._thread is not None:
            logger.warning("Event loop already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="EventLoop")
        self._thread.start()
        logger.info("Event loop started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Event loop stopped")

    def _run(self):
        while self._running:
            self._check_timers()
            time.sleep(0.5)

    def _check_timers(self):
        now = time.time()
        fired = []
        with self._lock:
            for name, timer in list(self._timers.items()):
                if now >= timer.fire_at:
                    fired.append(timer)
                    del self._timers[name]
        for timer in fired:
            threading.Thread(
                target=self._fire_timer,
                args=(timer,),
                daemon=True,
                name=f"Timer-{timer.name}",
            ).start()

    def _fire_timer(self, timer: Timer):
        """Fire a timer: mark reminder fired in brain, then invoke callback.

        Alert audio is the callback's responsibility (it can POST to HA
        notify.* or assist_satellite.announce if needed).
        """
        logger.info(f"Timer fired: {timer.name} (room={timer.room_id or 'default'})")

        try:
            from server.brain import get_brain
            brain = get_brain()
            for entry in brain.get_pending_reminders():
                if entry["data"].get("name") == timer.name:
                    brain.fire_reminder(entry["id"])
                    break
        except Exception:
            pass

        if timer.callback:
            try:
                timer.callback(timer.name)
            except Exception as e:
                logger.error(f"Timer callback failed: {e}")

    def add_timer(self, name: str, seconds: float,
                  callback: Optional[Callable[[str], None]] = None,
                  room_id: Optional[str] = None) -> bool:
        """Add a timer that fires after `seconds`. Returns False if name exists."""
        with self._lock:
            if name in self._timers:
                logger.warning(f"Timer already exists: {name}")
                return False

            fire_at = time.time() + seconds
            self._timers[name] = Timer(
                name=name,
                fire_at=fire_at,
                callback=callback,
                room_id=room_id,
            )

        logger.info(f"Timer added: {name} ({seconds}s)")

        try:
            from server.brain import get_brain
            brain = get_brain()
            brain.add_reminder(name, fire_at, seconds, room_id=room_id)
        except Exception:
            pass

        return True

    def cancel_timer(self, name: str) -> bool:
        with self._lock:
            if name in self._timers:
                del self._timers[name]
                logger.info(f"Timer cancelled: {name}")
                try:
                    from server.brain import get_brain
                    brain = get_brain()
                    brain.cancel_reminder(name)
                except Exception:
                    pass
                return True
        logger.warning(f"Timer not found: {name}")
        return False

    def list_timers(self) -> list[tuple[str, float]]:
        now = time.time()
        with self._lock:
            return [(name, max(0, timer.fire_at - now)) for name, timer in self._timers.items()]

    def get_timer_count(self) -> int:
        with self._lock:
            return len(self._timers)

    def load_pending_reminders(self):
        """Reload pending reminders from brain on startup.

        - fire_at in the past but within 5 minutes: fire immediately
        - fire_at in the past and older than 5 minutes: mark as fired (missed)
        - fire_at in the future: register as an in-memory timer
        """
        try:
            from server.brain import get_brain
            brain = get_brain()
            pending = brain.get_pending_reminders()
        except Exception as e:
            logger.warning(f"Could not load pending reminders from brain: {e}")
            return

        now = time.time()
        loaded = 0

        for entry in pending:
            try:
                data = entry["data"]
                reminder_id = entry["id"]
                name = data.get("name", "unknown")
                fire_at = data.get("fire_at", 0)
                room_id = data.get("room_id")

                if fire_at <= now:
                    age_seconds = now - fire_at
                    if age_seconds <= 300:
                        logger.info(f"Firing missed reminder (age={age_seconds:.0f}s): {name}")
                        brain.fire_reminder(reminder_id)
                        with self._lock:
                            self._timers[name] = Timer(
                                name=name, fire_at=time.time(), room_id=room_id,
                            )
                        loaded += 1
                    else:
                        logger.info(f"Marking stale reminder as fired (age={age_seconds:.0f}s): {name}")
                        brain.fire_reminder(reminder_id)
                else:
                    logger.info(f"Restoring reminder: {name} ({fire_at - now:.0f}s remaining)")
                    with self._lock:
                        self._timers[name] = Timer(
                            name=name, fire_at=fire_at, room_id=room_id,
                        )
                    loaded += 1
            except Exception as e:
                logger.error(f"Error loading reminder {entry}: {e}")

        logger.info(f"Loaded {loaded} pending reminders from brain")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_event_loop: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    """Get or create the global event loop."""
    global _event_loop
    if _event_loop is None:
        _event_loop = EventLoop()
        _event_loop.start()
    return _event_loop
