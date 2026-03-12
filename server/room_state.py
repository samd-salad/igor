"""Per-room mutable state: TV playback, Sonos cache, TTS buffer.

RoomState holds the state that was previously global on the Orchestrator.
RoomStateManager creates and manages RoomState instances, starting
per-room TV pollers for rooms that have a TV.
"""
import logging
import threading
import time
from typing import Optional, Callable

from server.rooms import RoomConfig

logger = logging.getLogger(__name__)


class RoomState:
    """Mutable per-room state."""

    def __init__(self, room: RoomConfig):
        self.room = room

        # TV playback state — updated by poller if room has a TV
        self._tv_state: str = "unknown"
        self._tv_state_lock = threading.Lock()
        self._adb_consecutive_failures: int = 0
        self._adb_backoff_interval: float = 5.0

        # Sonos device cache
        self._sonos_device = None
        self._sonos_cache_time: float = 0.0
        self._sonos_lock = threading.Lock()

        # In-memory TTS audio buffer — served at /audio/tts/{room_id}
        self._tts_audio: bytes = b""
        self._tts_audio_lock = threading.Lock()

    @property
    def tv_state(self) -> str:
        with self._tv_state_lock:
            return self._tv_state

    @tv_state.setter
    def tv_state(self, value: str):
        with self._tv_state_lock:
            self._tv_state = value

    @property
    def tts_audio(self) -> bytes:
        with self._tts_audio_lock:
            return self._tts_audio

    @tts_audio.setter
    def tts_audio(self, value: bytes):
        with self._tts_audio_lock:
            self._tts_audio = value

    def get_sonos_device(self):
        """Return cached Sonos device, rediscovering if stale."""
        from server.config import SONOS_DISCOVERY_CACHE_TTL
        if not self.room.has_sonos:
            return None
        try:
            import soco
            now = time.time()
            with self._sonos_lock:
                if self._sonos_device is None or now - self._sonos_cache_time > SONOS_DISCOVERY_CACHE_TTL:
                    devices = list(soco.discover(timeout=2) or [])
                    self._sonos_device = None
                    for d in devices:
                        if d.player_name == self.room.sonos_zone:
                            self._sonos_device = d
                            break
                    if self._sonos_device is None and devices:
                        self._sonos_device = devices[0]
                    self._sonos_cache_time = now
                return self._sonos_device
        except Exception as e:
            logger.warning(f"Sonos discovery failed for room '{self.room.room_id}': {e}")
            return None

    def invalidate_sonos_cache(self):
        """Force Sonos rediscovery on next access."""
        with self._sonos_lock:
            self._sonos_device = None
            self._sonos_cache_time = 0.0


class RoomStateManager:
    """Manages RoomState instances and per-room TV pollers."""

    def __init__(self, rooms: dict[str, RoomConfig]):
        self._states: dict[str, RoomState] = {}
        self._poller_threads: list[threading.Thread] = []

        for room_id, room in rooms.items():
            self._states[room_id] = RoomState(room)

    def get(self, room_id: str) -> Optional[RoomState]:
        """Get room state by ID."""
        return self._states.get(room_id)

    def get_or_default(self, room_id: str) -> RoomState:
        """Get room state, falling back to first available."""
        state = self._states.get(room_id)
        if state is None and self._states:
            state = next(iter(self._states.values()))
        return state

    def all_states(self) -> dict[str, RoomState]:
        return dict(self._states)

    def start_tv_pollers(self):
        """Start background TV state pollers for rooms that have a TV."""
        for room_id, state in self._states.items():
            if state.room.has_tv:
                t = threading.Thread(
                    target=self._poll_tv_state,
                    args=(state,),
                    daemon=True,
                    name=f"TVPoll-{room_id}",
                )
                t.start()
                self._poller_threads.append(t)
                logger.info(f"Started TV poller for room '{room_id}' (host={state.room.tv_host})")

    def _poll_tv_state(self, state: RoomState):
        """Background thread: poll TV playback state via ADB with circuit breaker."""
        from server.commands.adb_cmd import _get_tv_playback_state_for_host

        while True:
            try:
                tv_state = _get_tv_playback_state_for_host(state.room.tv_host)
                with state._tv_state_lock:
                    prev = state._tv_state
                    if tv_state == "unknown":
                        state._adb_consecutive_failures += 1
                        n = state._adb_consecutive_failures
                        if n >= 10 and state._adb_backoff_interval < 120:
                            state._adb_backoff_interval = 120.0
                            logger.warning(f"ADB circuit breaker ({state.room.room_id}): {n} failures, backing off to 120s")
                        elif n >= 6 and state._adb_backoff_interval < 60:
                            state._adb_backoff_interval = 60.0
                        elif n >= 3 and state._adb_backoff_interval < 30:
                            state._adb_backoff_interval = 30.0
                        if prev != "unknown":
                            logger.debug(f"TV state poll ({state.room.room_id}) returned 'unknown' — keeping '{prev}'")
                    else:
                        if state._adb_consecutive_failures > 0:
                            logger.info(f"ADB recovered ({state.room.room_id}) after {state._adb_consecutive_failures} failures")
                            state._adb_consecutive_failures = 0
                            state._adb_backoff_interval = 5.0
                        if tv_state != prev:
                            logger.info(f"TV state ({state.room.room_id}): {prev} → {tv_state}")
                        state._tv_state = tv_state
            except Exception as e:
                logger.debug(f"TV state poll failed ({state.room.room_id}): {e}")
                with state._tv_state_lock:
                    state._adb_consecutive_failures += 1
                    n = state._adb_consecutive_failures
                    if n >= 10 and state._adb_backoff_interval < 120:
                        state._adb_backoff_interval = 120.0
                    elif n >= 6 and state._adb_backoff_interval < 60:
                        state._adb_backoff_interval = 60.0
                    elif n >= 3 and state._adb_backoff_interval < 30:
                        state._adb_backoff_interval = 30.0
            time.sleep(state._adb_backoff_interval)
