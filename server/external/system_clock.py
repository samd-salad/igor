"""SystemClock — wall-clock implementation of cognition.ports.ClockPort."""
from datetime import datetime, UTC


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
