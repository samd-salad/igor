"""Background event loop for timers and scheduled events with Pi callback support.

Architecture:
  - A single background thread (_run) polls for expired timers every 0.5s.
  - When a timer fires, it spawns a NEW daemon thread for that timer's alert so
    TTS synthesis + HTTP to Pi (1-2s each) don't block the polling loop.
  - Timer state is protected by a single lock — the poll thread modifies
    _timers (removes fired entries) while the main thread may add/cancel.

Timer alert flow:
  1. _check_timers() finds an expired timer, removes it from _timers.
  2. _fire_timer() runs in its own thread:
     a. Plays alert beep on Pi (heads-up before speech).
     b. Synthesizes TTS on server ("pasta timer is done").
     c. Sends WAV to Pi for playback via PiCallbackClient.play_audio().
     d. Falls back to a second beep if TTS synthesis fails.
  3. Optional per-timer callback fires after audio, for future extensibility.

Global singleton pattern: initialize_event_loop() must be called once at server
startup with pi_client and synthesizer injected.  get_event_loop() returns the
existing instance; calling it before initialization creates a non-functional
instance (no audio delivery).
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Timer data structure holding identity, fire time, and optional callback."""
    name: str
    fire_at: float  # Unix timestamp when this timer should fire
    callback: Optional[Callable[[str], None]] = None  # Optional post-alert hook
    room_id: Optional[str] = None  # Room where the timer was set (for delivery routing)


class EventLoop:
    """Background event loop for scheduled events with Pi callback support."""

    def __init__(self, pi_client=None, synthesizer=None):
        """
        Initialize event loop.

        Args:
            pi_client: PiCallbackClient for sending audio/beeps to Pi.
                       If None, timers fire silently (no audio delivery).
            synthesizer: Synthesizer for generating TTS alert audio.
                         If None, only the beep plays (no speech).
        """
        self._timers: dict[str, Timer] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pi_client = pi_client
        self._synthesizer = synthesizer
        # Optional Sonos TTS routing — set via set_sonos_tts_func() after
        # orchestrator is created.  When set, timer alert TTS routes through
        # Sonos instead of Pi local playback.
        self._sonos_tts_func: Optional[Callable[[bytes], bool]] = None
        # Optional client registry for room-aware timer delivery
        self._client_registry = None
        logger.info("EventLoop initialized")

    def set_client_registry(self, registry):
        """Set the client registry for room-aware timer delivery."""
        self._client_registry = registry
        logger.info("EventLoop: client registry set for room-aware delivery")

    def set_sonos_tts_func(self, func: Callable[[bytes], bool]):
        """Set Sonos TTS routing function (called after orchestrator init).

        Args:
            func: Callable(audio_bytes) -> bool. Returns True if Sonos
                  accepted the audio, False to fall back to Pi playback.
        """
        self._sonos_tts_func = func
        logger.info("EventLoop: Sonos TTS routing enabled for timer alerts")

    def start(self):
        """Start the background polling thread (idempotent — safe to call twice)."""
        if self._thread is not None:
            logger.warning("Event loop already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="EventLoop")
        self._thread.start()
        logger.info("Event loop started")

    def stop(self):
        """Stop the polling thread and wait for it to exit (up to 2s)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Event loop stopped")

    def _run(self):
        """Main polling loop — checks for expired timers every 500ms.

        500ms granularity is fine for timer alerts — human perception of
        "late" starts around 2-3 seconds.
        """
        while self._running:
            self._check_timers()
            time.sleep(0.5)

    def _check_timers(self):
        """Atomically collect all expired timers, then fire each in its own thread.

        Fires in separate threads because TTS synthesis + HTTP to Pi takes ~1-2s.
        If we fired inline, the poll loop would stall and all timers due at the
        same tick would be delayed by N × 2s.
        """
        now = time.time()
        fired = []

        with self._lock:
            for name, timer in list(self._timers.items()):
                if now >= timer.fire_at:
                    fired.append(timer)
                    # Remove immediately so it can't fire twice if poll overlaps
                    del self._timers[name]

        # Fire each timer in its own thread — TTS synthesis + HTTP to Pi would
        # otherwise block the event loop for 1-2s per timer.
        for timer in fired:
            threading.Thread(
                target=self._fire_timer,
                args=(timer,),
                daemon=True,
                name=f"Timer-{timer.name}",
            ).start()

    def _get_pi_client_for_timer(self, timer: Timer):
        """Get the appropriate Pi client for a timer's room.

        If the timer has a room_id and the client registry is available,
        looks up the audio client for that room. Falls back to legacy pi_client.
        """
        if timer.room_id and self._client_registry:
            callback_url = self._client_registry.get_callback_url_for_room(timer.room_id)
            if callback_url:
                from server.pi_callback import PiCallbackClient
                return PiCallbackClient(callback_url)
        return self._pi_client

    def _fire_timer(self, timer: Timer):
        """Handle a fired timer: play alert beep, synthesize and deliver TTS.

        Runs in its own daemon thread per timer so multiple concurrent timers
        don't serialize.  All errors are caught — a failed alert is logged but
        never crashes the event loop.

        Fallback chain:
          1. Beep first (immediate audio signal even before speech is ready).
          2. Synthesize "X timer is done".
          3. Send WAV to Pi (room-aware if timer has room_id).
          4. If synthesis fails → second beep as fallback (requires pi_client).
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

        # Get the correct client for this timer's room
        pi_client = self._get_pi_client_for_timer(timer)

        try:
            # Step 1: Execute callback FIRST — for delayed commands, the action
            # (lights off, TV mute) should happen immediately, not after audio.
            if timer.callback:
                try:
                    timer.callback(timer.name)
                except Exception as e:
                    logger.error(f"Timer callback failed: {e}")

            # Step 2: Alert beep — gives immediate audio feedback while TTS synthesizes
            if pi_client:
                pi_client.play_beep("alert")
                time.sleep(0.3)  # Brief gap between beep and speech

            # Step 3: Generate TTS message and deliver to the right output
            message = f"{timer.name} timer is done"

            if self._synthesizer:
                audio_data = self._synthesizer.synthesize(message)

                if audio_data:
                    # Route to the timer's room client first; Sonos is fallback
                    # for rooms without a direct client (e.g. living room Pi is off).
                    delivered = False
                    if pi_client:
                        success = pi_client.play_audio(
                            audio_data, message, priority="alert"
                        )
                        if success:
                            logger.info(f"Timer alert sent to client: {timer.name}")
                            delivered = True

                    if not delivered and self._sonos_tts_func:
                        if self._sonos_tts_func(audio_data):
                            logger.info(f"Timer alert routed to Sonos: {timer.name}")
                            delivered = True

                    if not delivered:
                        logger.warning(f"No output available for timer alert: {timer.name}")
                else:
                    logger.error(f"Failed to synthesize timer alert: {timer.name}")
                    if pi_client:
                        pi_client.play_beep("alert")
            else:
                logger.warning(f"No synthesizer - cannot play timer alert: {timer.name}")

        except Exception as e:
            logger.error(f"Error firing timer '{timer.name}': {e}", exc_info=True)

    def add_timer(self, name: str, seconds: float, callback: Optional[Callable[[str], None]] = None,
                  room_id: Optional[str] = None) -> bool:
        """Add a timer that fires after `seconds` seconds.

        Timer names are unique — attempting to add a second timer with the same
        name returns False.  The LLM timer command handles this by telling the
        user to use a different name or cancel the existing one.

        Args:
            name: Unique timer label (e.g. "pasta", "laundry").
            seconds: Duration until firing.
            callback: Optional callable(name) invoked after audio playback.
            room_id: Room where the timer was set (for delivery routing).

        Returns:
            True on success, False if name already exists.
        """
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
            pass  # Non-critical — timer still works in-memory

        return True

    def cancel_timer(self, name: str) -> bool:
        """Cancel a timer by name before it fires.

        Args:
            name: Timer label to cancel.

        Returns:
            True if found and cancelled, False if not found.
        """
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
        """Return all active timers as (name, seconds_remaining) tuples.

        Seconds remaining is clamped to 0.0 for timers that are about to fire
        but haven't been reaped by the poll loop yet (can happen in the <500ms
        window between expiry and the next _check_timers() call).
        """
        now = time.time()
        with self._lock:
            return [(name, max(0, timer.fire_at - now)) for name, timer in self._timers.items()]

    def get_timer_count(self) -> int:
        """Return the number of currently active timers."""
        with self._lock:
            return len(self._timers)

    def load_pending_reminders(self):
        """Reload pending reminders from brain on startup.

        Handles three cases:
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
                    if age_seconds <= 300:  # Within 5 minutes — fire immediately
                        logger.info(f"Firing missed reminder (age={age_seconds:.0f}s): {name}")
                        brain.fire_reminder(reminder_id)
                        # Add directly to _timers (already marked fired in brain)
                        with self._lock:
                            self._timers[name] = Timer(
                                name=name, fire_at=time.time(), room_id=room_id,
                            )
                        loaded += 1
                    else:
                        logger.info(f"Marking stale reminder as fired (age={age_seconds:.0f}s): {name}")
                        brain.fire_reminder(reminder_id)
                else:
                    # Add directly to _timers (already persisted in brain)
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
# Global singleton — one event loop per server process
# ---------------------------------------------------------------------------
_event_loop: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    """Get or create the global event loop.

    WARNING: If called before initialize_event_loop(), creates an instance
    with no pi_client or synthesizer — timers will fire but produce no audio.
    Always call initialize_event_loop() at server startup.
    """
    global _event_loop
    if _event_loop is None:
        _event_loop = EventLoop()
        _event_loop.start()
    return _event_loop


def initialize_event_loop(pi_client, synthesizer) -> EventLoop:
    """Initialize the global event loop with audio delivery dependencies.

    Must be called once at server startup (server/main.py) with the fully
    constructed PiCallbackClient and Synthesizer.  Subsequent calls are
    no-ops (returns existing instance with a warning).

    Args:
        pi_client: PiCallbackClient instance for sending audio to Pi.
        synthesizer: Synthesizer instance for TTS generation.

    Returns:
        The initialized (and already started) EventLoop singleton.
    """
    global _event_loop
    if _event_loop is not None:
        logger.warning("Event loop already initialized")
        return _event_loop

    _event_loop = EventLoop(pi_client=pi_client, synthesizer=synthesizer)
    _event_loop.start()
    return _event_loop
