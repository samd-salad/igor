"""Google TV remote commands via androidtvremote2."""
import asyncio
import concurrent.futures
import logging

from .base import Command
from server.config import (
    GOOGLE_TV_HOST,
    GOOGLE_TV_CERT_FILE,
    GOOGLE_TV_KEY_FILE,
    GOOGLE_TV_CLIENT_NAME,
)

logger = logging.getLogger(__name__)

try:
    from androidtvremote2 import AndroidTVRemote, CannotConnect, ConnectionClosed, InvalidAuth
    _ATV_AVAILABLE = True
except ImportError:
    _ATV_AVAILABLE = False
    logger.warning("androidtvremote2 not installed; TV commands unavailable. Run: pip install androidtvremote2")

KEY_MAP = {
    "play":         "KEYCODE_MEDIA_PLAY",
    "pause":        "KEYCODE_MEDIA_PAUSE",
    "play_pause":   "KEYCODE_MEDIA_PLAY_PAUSE",
    "stop":         "KEYCODE_MEDIA_STOP",
    "home":         "KEYCODE_HOME",
    "back":         "KEYCODE_BACK",
    "up":           "KEYCODE_DPAD_UP",
    "down":         "KEYCODE_DPAD_DOWN",
    "left":         "KEYCODE_DPAD_LEFT",
    "right":        "KEYCODE_DPAD_RIGHT",
    "select":       "KEYCODE_DPAD_CENTER",
    "volume_up":    "KEYCODE_VOLUME_UP",
    "volume_down":  "KEYCODE_VOLUME_DOWN",
    "mute":         "KEYCODE_VOLUME_MUTE",
    "power":        "KEYCODE_POWER",
}


async def _connect(host: str = None) -> tuple:
    """Connect and wait for is_on to populate. Returns (remote, error_str)."""
    remote = AndroidTVRemote(
        client_name=GOOGLE_TV_CLIENT_NAME,
        certfile=GOOGLE_TV_CERT_FILE,
        keyfile=GOOGLE_TV_KEY_FILE,
        host=host or GOOGLE_TV_HOST,
    )
    await remote.async_generate_cert_if_missing()
    try:
        await remote.async_connect()
    except InvalidAuth:
        return None, "TV not paired. Run: python server/pair_google_tv.py"
    except (CannotConnect, ConnectionClosed, OSError) as e:
        return None, f"TV unavailable: cannot connect ({e})"
    # Brief wait for is_on state to be populated after handshake
    await asyncio.sleep(0.3)
    if remote.is_on is None:
        return None, "TV connected but not ready (is_on is None)"
    return remote, None


async def _async_power(state: str, host: str = None) -> str:
    """Smart power: uses is_on to avoid toggling in the wrong direction."""
    remote, err = await _connect(host)
    if err:
        return err
    try:
        is_on = remote.is_on
        if state == "on" and is_on:
            return "TV is already on"
        if state == "off" and not is_on:
            return "TV is already off"
        # KEYCODE_POWER toggles reliably on this TV
        remote.send_key_command("KEYCODE_POWER")
        await asyncio.sleep(2.0)
        return f"Turned TV {'on' if state == 'on' else 'off'}"
    except Exception as e:
        return f"TV power failed: {e}"
    finally:
        try:
            remote.disconnect()
        except Exception:
            pass


async def _async_send_key(keycode: str, host: str = None) -> str:
    remote, err = await _connect(host)
    if err:
        return err
    try:
        remote.send_key_command(keycode)
        await asyncio.sleep(1.0)
        return f"Sent {keycode} to TV"
    except Exception as e:
        return f"TV command failed: {e}"
    finally:
        try:
            remote.disconnect()
        except Exception:
            pass



def _run_tv(coro) -> str:
    """Run a TV coroutine in a fresh thread (no running event loop) via ThreadPoolExecutor.

    execute() runs on the uvicorn event loop thread, so asyncio.run() would
    raise RuntimeError. A worker thread has no event loop, making it safe.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            return "TV command timed out"
        except Exception as e:
            return f"TV command error: {e}"


def _get_tv_host(_ctx=None) -> str:
    """Get TV host from context or fall back to config."""
    if _ctx and hasattr(_ctx, 'room') and _ctx.room.tv_host:
        return _ctx.room.tv_host
    return GOOGLE_TV_HOST


def _tv_available(_ctx=None) -> str | None:
    """Return an error string if TV is unusable, None if ready."""
    if not _ATV_AVAILABLE:
        return "Google TV unavailable: androidtvremote2 not installed"
    host = _get_tv_host(_ctx)
    if not host or "0.X" in host:
        return "Google TV unavailable: set GOOGLE_TV_HOST in server/config.py or rooms.yaml"
    return None


class TvPowerCommand(Command):
    name = "tv_power"
    description = "Turn the Google TV on, off, or toggle power"

    @property
    def parameters(self) -> dict:
        return {
            "state": {
                "type": "string",
                "description": "Power action: 'on', 'off', or 'toggle'"
            }
        }

    def execute(self, state: str, _ctx=None) -> str:
        err = _tv_available(_ctx)
        if err:
            return err
        state_lower = state.lower().strip()
        if state_lower not in ("on", "off", "toggle"):
            return f"Unknown state '{state}'. Use: on, off, or toggle."
        return _run_tv(_async_power(state_lower, host=_get_tv_host(_ctx)))



class TvKeyCommand(Command):
    name = "tv_key"
    description = (
        "Send a remote key to the Google TV. "
        "Keys: play, pause, play_pause, stop, home, back, "
        "up, down, left, right, select, volume_up, volume_down, mute, power"
    )

    @property
    def parameters(self) -> dict:
        return {
            "key": {
                "type": "string",
                "description": (
                    "Key to send: play, pause, play_pause, stop, home, back, "
                    "up, down, left, right, select, volume_up, volume_down, mute, power"
                )
            }
        }

    def execute(self, key: str, _ctx=None) -> str:
        err = _tv_available(_ctx)
        if err:
            return err
        key_lower = key.lower().strip()
        keycode = KEY_MAP.get(key_lower)
        if not keycode:
            valid = ", ".join(sorted(KEY_MAP))
            return f"Unknown key '{key}'. Valid keys: {valid}"
        return _run_tv(_async_send_key(keycode, host=_get_tv_host(_ctx)))
