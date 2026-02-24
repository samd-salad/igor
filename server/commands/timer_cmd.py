"""Timer command for setting named timers."""
import re
from .base import Command
from server.event_loop import get_event_loop


def parse_duration(duration_str: str) -> float | None:
    """Parse a duration string into seconds.

    Supports formats like:
    - "5 minutes", "5 min", "5m"
    - "30 seconds", "30 sec", "30s"
    - "1 hour", "1 hr", "1h"
    - "1 hour 30 minutes", "1h30m"
    - "90" (assumes seconds)
    """
    duration_str = duration_str.lower().strip()

    # Try to parse as just a number (assume seconds)
    try:
        return float(duration_str)
    except ValueError:
        pass

    total_seconds = 0.0

    # Pattern to match number + unit pairs
    pattern = r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)?'
    matches = re.findall(pattern, duration_str)

    if not matches:
        return None

    for value_str, unit in matches:
        value = float(value_str)
        unit = unit.lower() if unit else 's'

        if unit in ('h', 'hr', 'hrs', 'hour', 'hours'):
            total_seconds += value * 3600
        elif unit in ('m', 'min', 'mins', 'minute', 'minutes'):
            total_seconds += value * 60
        elif unit in ('s', 'sec', 'secs', 'second', 'seconds', ''):
            total_seconds += value

    return total_seconds if total_seconds > 0 else None


class SetTimerCommand(Command):
    name = "set_timer"
    description = "Set a named timer that will alert when done"

    @property
    def parameters(self) -> dict:
        return {
            "name": {
                "type": "string",
                "description": "Name of the timer (e.g., 'pasta', 'laundry', 'break')"
            },
            "duration": {
                "type": "string",
                "description": "Duration (e.g., '5 minutes', '1 hour 30 minutes', '90 seconds', '1h30m')"
            }
        }

    def execute(self, name: str, duration: str) -> str:
        name = name.strip()
        if not name:
            return "Timer name is required"

        seconds = parse_duration(duration)
        if seconds is None or seconds <= 0:
            return f"Could not parse duration: '{duration}'"

        event_loop = get_event_loop()

        if not event_loop.add_timer(name, seconds):
            return f"A timer named '{name}' already exists. Cancel it first or use a different name."

        # Format duration for confirmation
        if seconds >= 3600:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            if mins:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''} {mins} minute{'s' if mins != 1 else ''}"
            else:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''}"
        elif seconds >= 60:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            if secs:
                duration_str = f"{mins} minute{'s' if mins != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
            else:
                duration_str = f"{mins} minute{'s' if mins != 1 else ''}"
        else:
            duration_str = f"{int(seconds)} second{'s' if int(seconds) != 1 else ''}"

        return f"Timer '{name}' set for {duration_str}"


class CancelTimerCommand(Command):
    name = "cancel_timer"
    description = "Cancel an active timer"

    @property
    def parameters(self) -> dict:
        return {
            "name": {
                "type": "string",
                "description": "Name of the timer to cancel"
            }
        }

    def execute(self, name: str) -> str:
        name = name.strip()
        if not name:
            return "Timer name is required"

        event_loop = get_event_loop()

        if event_loop.cancel_timer(name):
            return f"Timer '{name}' cancelled"
        else:
            return f"No active timer named '{name}'"


class ListTimersCommand(Command):
    name = "list_timers"
    description = "List all active timers and their remaining time"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self) -> str:
        event_loop = get_event_loop()
        timers = event_loop.list_timers()

        if not timers:
            return "No active timers"

        lines = []
        for name, remaining in timers:
            if remaining >= 3600:
                hours = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                time_str = f"{hours}h {mins}m"
            elif remaining >= 60:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                time_str = f"{mins}m {secs}s"
            else:
                time_str = f"{int(remaining)}s"
            lines.append(f"- {name}: {time_str} remaining")

        return "Active timers:\n" + "\n".join(lines)
