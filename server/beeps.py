"""Pre-generated beep sounds for Sonos output (stdlib only, no sox required).

Matches the tone definitions in client/audio.py so Sonos beeps sound the same
as the Pi's local beeps.
"""
import io
import logging
import math
import struct
import wave

from server.config import DATA_DIR

logger = logging.getLogger(__name__)

_RATE = 44100


def _sweep(f0: float, f1: float, dur: float, vol: float) -> list:
    """Frequency sweep from f0→f1 Hz over dur seconds."""
    n = int(_RATE * dur)
    phase = 0.0
    out = []
    for i in range(n):
        out.append(int(vol * 32767 * math.sin(phase)))
        phase += 2 * math.pi * (f0 + (f1 - f0) * i / n) / _RATE
    return out


def _tone(freq: float, dur: float, vol: float) -> list:
    """Constant frequency tone."""
    n = int(_RATE * dur)
    step = 2 * math.pi * freq / _RATE
    phase = 0.0
    out = []
    for _ in range(n):
        out.append(int(vol * 32767 * math.sin(phase)))
        phase += step
    return out


def _gap(dur: float) -> list:
    return [0] * int(_RATE * dur)


def _to_wav(samples: list) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_RATE)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))
    return buf.getvalue()


# Mirror of client/audio.py beep definitions
_DEFS: dict[str, callable] = {
    'start': lambda: _to_wav(_sweep(500, 900, 0.12, 0.30)),
    'end':   lambda: _to_wav(_sweep(700, 400, 0.12, 0.25)),
    'done':  lambda: _to_wav(
        _tone(1200, 0.06, 0.20) + _gap(0.04) + _tone(1200, 0.06, 0.20)
    ),
    'error': lambda: _to_wav(_tone(200, 0.30, 0.25)),
    'alert': lambda: _to_wav(
        _tone(660,  0.10, 0.35) + _gap(0.08) +
        _tone(880,  0.10, 0.35) + _gap(0.08) +
        _tone(1100, 0.15, 0.40)
    ),
}

_cache: dict[str, bytes] = {}


def get_beep_wav(beep_type: str) -> bytes | None:
    """Return WAV bytes for the given beep type, generating on first call."""
    if beep_type not in _DEFS:
        return None
    if beep_type not in _cache:
        _cache[beep_type] = _DEFS[beep_type]()
    return _cache[beep_type]


def write_beep_files():
    """Write all beep WAVs to data/ so Sonos can fetch them by URI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in _DEFS:
        path = DATA_DIR / f"beep_{name}.wav"
        path.write_bytes(get_beep_wav(name))
    logger.info(f"Beep WAV files written to {DATA_DIR}")
