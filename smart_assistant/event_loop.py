"""Background event loop for timers, network monitoring, and future scheduled triggers."""
import logging
import threading
import time
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional
from config import AUDIO_DEVICE, PIPER_VOICE

log = logging.getLogger(__name__)


@dataclass
class Timer:
    name: str
    fire_at: float  # Unix timestamp
    callback: Optional[Callable[[str], None]] = None


class EventLoop:
    """Background event loop for scheduled events and monitoring."""

    def __init__(self):
        self._timers: dict[str, Timer] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._network_monitor = None
        self._last_network_check = 0
        self._network_check_interval = 30  # Check monitor state every 30s

    def start(self):
        """Start the background event loop."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background event loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def enable_network_monitoring(self):
        """Enable network monitoring for unknown devices."""
        if self._network_monitor is not None:
            return

        try:
            from network_monitor import NetworkMonitor
            self._network_monitor = NetworkMonitor(alert_callback=self._on_network_alert)
            log.info("Network monitoring enabled")
        except ImportError as e:
            log.error(f"Failed to enable network monitoring: {e}")

    def _on_network_alert(self, device, is_reminder=False):
        """Handle network alert - play sound and announce."""
        try:
            # Play network alert sound (different from timer)
            subprocess.run(
                'play -n synth 0.2 sine 300 vol 0.3',
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            if is_reminder:
                message = f"Reminder: unknown device still on network. IP {device.ip}"
            else:
                message = f"Unknown device on network. IP {device.ip}, MAC {device.mac}"

            self._speak(message)
            log.info(f"Network alert: {device.ip} ({device.mac})")
        except Exception as e:
            log.error(f"Failed to announce network alert: {e}")

    def _run(self):
        """Main loop - checks timers and network monitor."""
        while self._running:
            self._check_timers()
            self._check_network_monitor()
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

        for timer in fired:
            self._fire_timer(timer)

    def _check_network_monitor(self):
        """Periodically tick the network monitor."""
        if self._network_monitor is None:
            return

        now = time.time()
        if now - self._last_network_check < self._network_check_interval:
            return

        self._last_network_check = now
        try:
            self._network_monitor.tick()
        except Exception as e:
            log.error(f"Network monitor tick failed: {e}")

    def _fire_timer(self, timer: Timer):
        """Handle a fired timer - play sound and speak."""
        # Play alert sound: triple ascending chime
        subprocess.run(
            'play -n synth 0.1 sine 660 vol 0.35 pad 0 0.08 synth 0.1 sine 880 vol 0.35 pad 0 0.08 synth 0.15 sine 1100 vol 0.4',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Speak the timer name
        message = f"{timer.name} timer is done"
        self._speak(message)

        # Call custom callback if provided
        if timer.callback:
            try:
                timer.callback(timer.name)
            except Exception:
                pass

    def _speak(self, text: str):
        """Speak text using piper TTS."""
        safe_text = text.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
        subprocess.run(
            f'echo "{safe_text}" | piper --model {PIPER_VOICE} --output-raw 2>/dev/null | '
            f'aplay -D {AUDIO_DEVICE} -r 22050 -f S16_LE -t raw - 2>/dev/null',
            shell=True, stdout=subprocess.DEVNULL
        )

    def add_timer(self, name: str, seconds: float, callback: Optional[Callable[[str], None]] = None) -> bool:
        """Add a timer. Returns False if name already exists."""
        with self._lock:
            if name in self._timers:
                return False
            self._timers[name] = Timer(
                name=name,
                fire_at=time.time() + seconds,
                callback=callback
            )
        return True

    def cancel_timer(self, name: str) -> bool:
        """Cancel a timer by name. Returns False if not found."""
        with self._lock:
            if name in self._timers:
                del self._timers[name]
                return True
            return False

    def list_timers(self) -> list[tuple[str, float]]:
        """List all active timers as (name, seconds_remaining) tuples."""
        now = time.time()
        with self._lock:
            return [(name, max(0, timer.fire_at - now)) for name, timer in self._timers.items()]

    def force_network_scan(self):
        """Force an immediate network scan."""
        if self._network_monitor:
            return self._network_monitor.scan_and_alert()
        return []


# Global singleton
_event_loop: Optional[EventLoop] = None


def get_event_loop() -> EventLoop:
    """Get or create the global event loop."""
    global _event_loop
    if _event_loop is None:
        _event_loop = EventLoop()
        _event_loop.start()
    return _event_loop
