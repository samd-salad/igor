"""Timer command for setting named timers."""
import re
from typing import Optional
from .base import Command
from server.event_loop import get_event_loop


# Word-to-number map for natural language durations ("half an hour", "a minute")
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "forty five": 45,
    "forty5": 45, "sixty": 60, "ninety": 90,
}


def _normalize_words(text: str) -> str:
    """Replace word-numbers and fractional phrases with digits.

    Handles: "half an hour" -> "30 minutes", "quarter hour" -> "15 minutes",
    "a minute and a half" -> "1.5 minutes", "five minutes" -> "5 minutes",
    "three and a half minutes" -> "3.5 minutes".

    Order matters — "and a half" must be resolved before word->digit replacement,
    because "a" is both a word-number (one) and an article in "a half".
    """
    # Step 1: Fixed fractional phrases
    text = re.sub(r'\bhalf\s+(?:an?\s+)?hour\b', '30 minutes', text)
    text = re.sub(r'\bquarter\s+(?:of\s+an?\s+|an?\s+)?hour\b', '15 minutes', text)

    # Step 2: "[word/digit] [unit] and a half" -> "[N.5] [unit]"
    # Must run before word->digit so "a" in "a half" isn't misread as "one".
    # Build a combined pattern that matches both word-numbers and digits.
    _word_pattern = "|".join(re.escape(w) for w in sorted(_WORD_NUMBERS, key=lambda x: -len(x)))

    def _half_repl(m):
        """Convert the number part (word or digit) to float and add 0.5."""
        num_str = m.group(1)
        val = _WORD_NUMBERS.get(num_str)
        if val is None:
            val = float(num_str)
        return f"{val + 0.5} {m.group(2)}"

    # "X [unit] and a half" (e.g. "a minute and a half", "three hours and a half")
    text = re.sub(
        rf'\b({_word_pattern}|\d+(?:\.\d+)?)\s+(minutes?|hours?|seconds?)\s+and\s+a\s+half\b',
        _half_repl, text,
    )
    # "X and a half [unit]" (e.g. "three and a half minutes", "2 and a half hours")
    text = re.sub(
        rf'\b({_word_pattern}|\d+(?:\.\d+)?)\s+and\s+a\s+half\s+(minutes?|hours?|seconds?)',
        _half_repl, text,
    )

    # Step 3: Word-numbers to digits (longest-first to match "forty-five" before "forty")
    for word, val in sorted(_WORD_NUMBERS.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\b{re.escape(word)}\b', str(val), text)

    return text


def parse_duration(duration_str: str) -> Optional[float]:
    """Parse a duration string into seconds.

    Supports:
    - Numeric: "5 minutes", "5 min", "5m", "1h30m", "90"
    - Word numbers: "five minutes", "thirty seconds"
    - Fractions: "half an hour", "quarter hour", "a minute and a half"
    """
    duration_str = duration_str.lower().strip()

    # Try to parse as just a number (assume seconds)
    try:
        val = float(duration_str)
        if not (0 < val < float('inf')):
            return None
        return val
    except ValueError:
        pass

    # Normalize word-numbers and fractions to digits
    normalized = _normalize_words(duration_str)

    total_seconds = 0.0

    # Pattern to match number + unit pairs
    pattern = r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)?'
    matches = re.findall(pattern, normalized)

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
        if seconds > 86400:  # 24 hours max
            return "Maximum timer duration is 24 hours"

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

    def execute(self, **_) -> str:
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
