"""Command pattern logging and routine detection."""
import logging
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

# Commands worth tracking — skip noisy/transient ones (calculate, get_time, etc.)
_TRACKED = {
    'get_weather', 'set_timer',
    'set_light', 'set_brightness', 'set_color', 'set_color_temp',
    'adjust_brightness', 'adjust_color_temp', 'shift_hue',
    'set_sonos_volume', 'adjust_sonos_volume', 'sonos_mute',
    'set_volume', 'adjust_volume',
    'tv_power', 'tv_launch', 'tv_playback',
}


def log_command(name: str, params: dict = None):
    """Log a command execution for pattern analysis."""
    if name not in _TRACKED:
        return
    now = datetime.now()
    from server.brain import get_brain
    brain = get_brain()
    brain.log_routine(name, now.hour, now.weekday(), params=params)


def get_patterns(min_occurrences: int = 3, top_n: int = 5) -> str:
    """Return top recurring patterns as a formatted string for prompt injection.
    Groups by (command, 2-hour bucket) day-agnostic.
    Returns empty string if no patterns meet the threshold.
    """
    from server.brain import get_brain
    brain = get_brain()
    entries = brain.get_routine_entries()
    if not entries:
        return ""

    counts = Counter()
    for e in entries:
        d = e.get("data", {})
        bucket = (d.get("hour", 0) // 2) * 2
        counts[(d.get("command", ""), bucket)] += 1

    patterns = [(k, v) for k, v in counts.items() if v >= min_occurrences]
    patterns.sort(key=lambda x: -x[1])

    if not patterns:
        return ""

    lines = []
    for (cmd, bucket), count in patterns[:top_n]:
        time_str = f"{bucket}:00\u2013{bucket + 2}:00"
        lines.append(f"- {cmd}: {time_str} ({count}\u00d7)")
    return "\n".join(lines)
