"""Background event loop for timers and scheduled events with Pi callback support."""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Timer data structure."""
    name: str
    fire_at: float  # Unix timestamp
    callback: Optional[Callable[[str], None]] = None


class EventLoop:
    """Background event loop for scheduled events with Pi callback support."""

    def __init__(self, pi_client=None, synthesizer=None):
        """
        Initialize event loop.

        Args:
            pi_client: PiCallbackClient for sending audio to Pi
            synthesizer: Synthesizer for generating alert audio
        """
        self._timers: dict[str, Timer] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pi_client = pi_client
        self._synthesizer = synthesizer
        logger.info("EventLoop initialized")

    def start(self):
        """Start the background event loop."""
        if self._thread is not None:
            logger.warning("Event loop already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="EventLoop")
        self._thread.start()
        logger.info("Event loop started")

    def stop(self):
        """Stop the background event loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Event loop stopped")

    def _run(self):
        """Main loop - checks timers periodically."""
        while self._running:
            self._check_timers()
            time.sleep(0.5)

    def _check_timers(self):
        """Check and fire any expired timers."""
        now = time.time()
        fired = []

        with self._lock:
            for name, timer in list(self._timers.items()):
                if now >= timer.fire_at:
                    fired.append(timer)
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

    def _fire_timer(self, timer: Timer):
        """
        Handle a fired timer.

        Synthesizes audio on server and sends to Pi for playback.
        """
        logger.info(f"Timer fired: {timer.name}")

        try:
            # Play alert beep on Pi first
            if self._pi_client:
                self._pi_client.play_beep("alert")
                time.sleep(0.3)  # Small delay between beep and speech

            # Generate TTS message
            message = f"{timer.name} timer is done"

            # Synthesize audio and send to Pi
            if self._synthesizer and self._pi_client:
                logger.debug(f"Synthesizing timer alert: '{message}'")
                audio_data = self._synthesizer.synthesize(message)

                if audio_data:
                    # Send to Pi for playback
                    success = self._pi_client.play_audio(
                        audio_data,
                        message,
                        priority="alert"
                    )
                    if success:
                        logger.info(f"Timer alert sent to Pi: {timer.name}")
                    else:
                        logger.error(f"Failed to send timer alert to Pi: {timer.name}")
                else:
                    logger.error(f"Failed to synthesize timer alert: {timer.name}")
                    # Fallback: extra beep so user knows the timer fired even without speech
                    self._pi_client.play_beep("alert")
            else:
                logger.warning(f"No synthesizer or Pi client - cannot play timer alert: {timer.name}")

            # Call custom callback if provided
            if timer.callback:
                try:
                    timer.callback(timer.name)
                except Exception as e:
                    logger.error(f"Timer callback failed: {e}")

        except Exception as e:
            logger.error(f"Error firing timer '{timer.name}': {e}", exc_info=True)

    def add_timer(self, name: str, seconds: float, callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Add a timer.

        Args:
            name: Timer name/label
            seconds: Duration in seconds
            callback: Optional callback to call when timer fires

        Returns:
            False if name already exists, True on success
        """
        with self._lock:
            if name in self._timers:
                logger.warning(f"Timer already exists: {name}")
                return False

            fire_at = time.time() + seconds
            self._timers[name] = Timer(
                name=name,
                fire_at=fire_at,
                callback=callback
            )

        logger.info(f"Timer added: {name} ({seconds}s)")
        return True

    def cancel_timer(self, name: str) -> bool:
        """
        Cancel a timer by name.

        Args:
            name: Timer name

        Returns:
            False if not found, True on success
        """
        with self._lock:
            if name in self._timers:
                del self._timers[name]
                logger.info(f"Timer cancelled: {name}")
                return True

        logger.warning(f"Timer not found: {name}")
        return False

    def list_timers(self) -> list[tuple[str, float]]:
        """
        List all active timers.

        Returns:
            List of (name, seconds_remaining) tuples
        """
        now = time.time()
        with self._lock:
            return [(name, max(0, timer.fire_at - now)) for name, timer in self._timers.items()]

    def get_timer_count(self) -> int:
        """Get number of active timers."""
        with self._lock:
            return len(self._timers)


# Global singleton
_event_loop: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    """
    Get or create the global event loop.

    Note: For server use, you should initialize with pi_client and synthesizer
    before calling this.
    """
    global _event_loop
    if _event_loop is None:
        _event_loop = EventLoop()
        _event_loop.start()
    return _event_loop


def initialize_event_loop(pi_client, synthesizer) -> EventLoop:
    """
    Initialize the global event loop with dependencies.

    Args:
        pi_client: PiCallbackClient instance
        synthesizer: Synthesizer instance

    Returns:
        Initialized EventLoop
    """
    global _event_loop
    if _event_loop is not None:
        logger.warning("Event loop already initialized")
        return _event_loop

    _event_loop = EventLoop(pi_client=pi_client, synthesizer=synthesizer)
    _event_loop.start()
    return _event_loop
