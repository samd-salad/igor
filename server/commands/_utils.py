"""Shared parsing utilities for command parameters."""


def parse_amount(amount: str, slightly: int, medium: int, a_lot: int) -> int:
    """Map an amount word or number to a step value.

    Accepts words like 'slightly', 'a lot', numeric strings like '10', '10%',
    or anything else (falls back to medium). Never errors.
    """
    a = str(amount).lower().strip().rstrip("%")

    # Numeric
    try:
        return max(1, int(float(a)))
    except ValueError:
        pass

    # Word groups
    if a in ("slightly", "a little", "a bit", "small", "tiny", "little", "bit", "barely"):
        return slightly
    if a in ("a lot", "much", "very", "large", "big", "significantly", "way", "really",
             "alot", "lot", "tons", "ton", "huge", "massive", "drastically"):
        return a_lot
    # Everything else (medium, moderate, some, "", unknown)
    return medium


def parse_direction_updown(direction: str) -> str | None:
    """Return 'up', 'down', or None if unrecognised."""
    d = direction.lower().strip()
    if d in ("up", "higher", "more", "increase", "louder", "brighter", "raise", "boost"):
        return "up"
    if d in ("down", "lower", "less", "decrease", "quieter", "dimmer", "softer",
             "reduce", "drop", "dim"):
        return "down"
    return None


def parse_volume_word(level: str) -> int | None:
    """Map a volume word to 0-100. Returns None if not a known word."""
    return {
        "mute": 0, "silent": 0, "off": 0,
        "quiet": 20, "low": 20, "soft": 20,
        "medium": 50, "moderate": 50, "normal": 50, "mid": 50, "half": 50,
        "loud": 75, "high": 75,
        "max": 100, "full": 100, "maximum": 100,
    }.get(level.lower().strip())
