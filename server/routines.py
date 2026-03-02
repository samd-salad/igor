"""Command pattern logging and routine detection."""
import json
import threading
from collections import Counter
from datetime import datetime

from server.config import ROUTINES_FILE

_lock = threading.Lock()
_MAX_ENTRIES = 1000

# Commands worth tracking — skip noisy/transient ones (calculate, get_time, etc.)
_TRACKED = {
    'get_weather', 'set_timer',
    'set_light', 'set_brightness', 'set_color', 'set_color_temp',
    'adjust_brightness', 'adjust_color_temp', 'shift_hue',
    'set_sonos_volume', 'adjust_sonos_volume', 'sonos_mute',
    'set_volume', 'adjust_volume',
    'tv_power', 'tv_launch', 'tv_playback',
}


def log_command(name: str):
    """Log a command execution for pattern analysis."""
    if name not in _TRACKED:
        return
    now = datetime.now()
    entry = {
        "command": name,
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.weekday(),  # 0=Monday
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
        _save(entries)


def get_patterns(min_occurrences: int = 3, top_n: int = 5) -> str:
    """Return top recurring patterns as a formatted string for prompt injection.
    Groups by (command, 2-hour bucket) day-agnostic.
    Returns empty string if no patterns meet the threshold.
    """
    entries = _load()
    if not entries:
        return ""

    counts = Counter()
    for e in entries:
        bucket = (e["hour"] // 2) * 2
        counts[(e["command"], bucket)] += 1

    patterns = [(k, v) for k, v in counts.items() if v >= min_occurrences]
    patterns.sort(key=lambda x: -x[1])

    if not patterns:
        return ""

    lines = []
    for (cmd, bucket), count in patterns[:top_n]:
        time_str = f"{bucket}:00\u2013{bucket + 2}:00"
        lines.append(f"- {cmd}: {time_str} ({count}\u00d7)")
    return "\n".join(lines)


def _load() -> list:
    if not ROUTINES_FILE.exists():
        return []
    try:
        return json.loads(ROUTINES_FILE.read_text())
    except Exception:
        return []


def _save(entries: list):
    ROUTINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROUTINES_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(entries))
    tmp.replace(ROUTINES_FILE)
