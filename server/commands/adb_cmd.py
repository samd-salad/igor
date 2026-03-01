"""Google TV control via ADB (adb-shell).

Handles reliable app launch, media playback controls, skipping, and text input.
Requires 'Network debugging' (ADB over Wi-Fi) enabled on the TV.
Run tv_adb_connect once to approve the RSA key on the TV.
"""
import logging
import os
import shlex

from .base import Command
from server.config import GOOGLE_TV_HOST, ADB_KEY_FILE

logger = logging.getLogger(__name__)

try:
    from adb_shell.adb_device import AdbDeviceTcp
    from adb_shell.auth.sign_pythonrsa import PythonRSASigner
    _ADB_AVAILABLE = True
except ImportError:
    _ADB_AVAILABLE = False
    logger.warning("adb-shell not installed; ADB TV commands unavailable. Run: pip install adb-shell")

ADB_PORT = 5555

APP_PACKAGES = {
    "netflix":       "com.netflix.ninja",
    "youtube":       "com.google.android.youtube.tv",
    "smarttube":     "is.xyz.smarttube",
    "spotify":       "com.spotify.tv.android",
    "disney":        "com.disney.disneyplus",
    "disney+":       "com.disney.disneyplus",
    "hulu":          "com.hulu.plus",
    "max":           "com.hbo.hbonow",
    "hbo":           "com.hbo.hbonow",
    "prime":         "com.amazon.amazonvideo.livingroom",
    "amazon":        "com.amazon.amazonvideo.livingroom",
    "prime video":   "com.amazon.amazonvideo.livingroom",
    "plex":          "com.plexapp.android",
    "twitch":        "tv.twitch.android.app",
    "apple tv":      "com.apple.atve.androidtv.appletv",
    "peacock":       "com.peacocktv.peacockandroid",
    "paramount":     "com.cbs.ott",
    "paramount+":    "com.cbs.ott",
}

PLAYBACK_KEYS = {
    "play":          126,  # KEYCODE_MEDIA_PLAY (discrete — does not toggle)
    "pause":         127,  # KEYCODE_MEDIA_PAUSE (discrete — does not toggle)
    "play_pause":    85,   # KEYCODE_MEDIA_PLAY_PAUSE (toggle)
    "stop":          86,   # KEYCODE_MEDIA_STOP
    "next":          87,   # KEYCODE_MEDIA_NEXT
    "previous":      88,   # KEYCODE_MEDIA_PREVIOUS
    "prev":          88,
    "rewind":        89,   # KEYCODE_MEDIA_REWIND
    "fast_forward":  90,   # KEYCODE_MEDIA_FAST_FORWARD
    "forward":       90,
}


def _get_signer() -> "PythonRSASigner":
    key_path = ADB_KEY_FILE
    if not os.path.exists(key_path):
        from adb_shell.auth.keygen import keygen
        keygen(key_path)
        logger.info(f"Generated ADB RSA key at {key_path}")
    with open(key_path) as f:
        priv = f.read()
    pub_path = key_path + ".pub"
    if os.path.exists(pub_path):
        with open(pub_path) as f:
            pub = f.read()
    else:
        pub = ""
    return PythonRSASigner(pub, priv)


def _adb_shell(command: str, auth_timeout: float = 1.0, cmd_timeout: float = 10.0) -> tuple:
    """Run a shell command on the TV via ADB. Returns (output, error_str)."""
    if not _ADB_AVAILABLE:
        return "", "adb-shell not installed. Run: pip install adb-shell"
    try:
        signer = _get_signer()
        device = AdbDeviceTcp(GOOGLE_TV_HOST, ADB_PORT)
        # transport_timeout_s gates the TCP socket connect; without it the call can block indefinitely
        device.connect(
            rsa_keys=[signer],
            auth_timeout_s=auth_timeout,
            read_timeout_s=cmd_timeout,
            transport_timeout_s=auth_timeout,
        )
        try:
            result = device.shell(command, read_timeout_s=cmd_timeout)
            return result or "", None
        finally:
            device.close()
    except Exception as e:
        return "", f"ADB error: {e}"


def _tv_adb_available() -> str | None:
    if not _ADB_AVAILABLE:
        return "adb-shell not installed. Run: pip install adb-shell"
    if not GOOGLE_TV_HOST or "0.X" in GOOGLE_TV_HOST:
        return "GOOGLE_TV_HOST not configured in server/config.py"
    return None


def _get_tv_playback_state() -> str:
    """Query ADB for current media playback state.

    Returns 'playing', 'paused', 'stopped', or 'unknown'.
    'unknown' on any error — callers should treat unknown as non-playing.
    """
    if not _ADB_AVAILABLE or not GOOGLE_TV_HOST or "0.X" in GOOGLE_TV_HOST:
        return "unknown"
    out, err = _adb_shell("dumpsys media_session", cmd_timeout=3.0)
    if err or not out:
        return "unknown"
    # STATE_PLAYING=3, STATE_PAUSED=2, STATE_STOPPED=1
    if "state=PlaybackState {state=3" in out:
        return "playing"
    if "state=PlaybackState {state=2" in out:
        return "paused"
    if "state=PlaybackState {state=1" in out:
        return "stopped"
    return "unknown"



class TvAdbConnectCommand(Command):
    name = "tv_adb_connect"
    description = (
        "Test the ADB connection to the Google TV. "
        "Run this once after enabling ADB on the TV — you may need to accept a prompt on the TV screen."
    )

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        err = _tv_adb_available()
        if err:
            return err
        # Use longer auth timeout for initial approval dialog
        out, err = _adb_shell("echo ok", auth_timeout=30.0)
        if err:
            return f"ADB connection failed: {err}"
        return "ADB connected to TV successfully"


class TvLaunchCommand(Command):
    name = "tv_launch"
    description = (
        "Launch an app on the Google TV by name via ADB. "
        "Supported: netflix, youtube, smarttube, spotify, disney+, hulu, max, "
        "prime video, plex, twitch, apple tv, peacock, paramount+"
    )

    @property
    def parameters(self) -> dict:
        return {
            "app": {
                "type": "string",
                "description": "App name, e.g. 'youtube', 'netflix', 'spotify'"
            }
        }

    def execute(self, app: str) -> str:
        err = _tv_adb_available()
        if err:
            return err
        app_lower = app.lower().strip()
        package = APP_PACKAGES.get(app_lower)
        if not package:
            return f"Unknown app '{app}'. Supported: {', '.join(sorted(APP_PACKAGES.keys()))}"
        out, err = _adb_shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
        if err:
            return f"Failed to launch {app}: {err}"
        return f"Launched {app} on TV"


class TvPlaybackCommand(Command):
    name = "tv_playback"
    description = (
        "Control media playback on the Google TV: play, pause, play/pause toggle, "
        "stop, next, previous, rewind, fast_forward."
    )

    @property
    def parameters(self) -> dict:
        return {
            "action": {
                "type": "string",
                "description": "Playback action: play, pause, play_pause, stop, next, previous, rewind, fast_forward"
            }
        }

    def execute(self, action: str) -> str:
        err = _tv_adb_available()
        if err:
            return err
        action_lower = action.lower().strip()
        keycode = PLAYBACK_KEYS.get(action_lower)
        if keycode is None:
            valid = ", ".join(sorted(PLAYBACK_KEYS))
            return f"Unknown action '{action}'. Use: {valid}"
        out, err = _adb_shell(f"input keyevent {keycode}")
        if err:
            return f"Playback command failed: {err}"
        return f"TV playback: {action}"


class TvSkipCommand(Command):
    name = "tv_skip"
    description = (
        "Skip forward or backward in the current video by a number of seconds. "
        "Works in YouTube and most video apps. Each press skips ~5 seconds."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {
                "type": "string",
                "description": "Direction: 'forward' or 'back'"
            },
            "seconds": {
                "type": "integer",
                "description": "Number of seconds to skip (approximate, rounded to 5s increments). Default 30."
            }
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, seconds: int = 30) -> str:
        err = _tv_adb_available()
        if err:
            return err
        direction = direction.lower().strip()
        if direction in ("forward", "ahead", "right"):
            keycode = 22  # KEYCODE_DPAD_RIGHT
        elif direction in ("back", "backward", "rewind", "left"):
            keycode = 21  # KEYCODE_DPAD_LEFT
        else:
            return f"Unknown direction '{direction}'. Use 'forward' or 'back'."

        try:
            seconds = int(seconds)
        except (ValueError, TypeError):
            return f"Invalid seconds value '{seconds}'. Use a number."
        presses = max(1, min(120, round(seconds / 5)))  # cap at 120 (~10 min)
        cmd = " && ".join(f"input keyevent {keycode}" for _ in range(presses))
        out, err = _adb_shell(cmd)
        if err:
            return f"Skip failed: {err}"
        actual = presses * 5
        return f"Skipped {direction} ~{actual}s on TV"


class TvSearchYouTubeCommand(Command):
    name = "tv_search_youtube"
    description = (
        "Open YouTube on the Google TV and search for something. "
        "Use when the user wants to find or watch something on YouTube."
    )

    @property
    def parameters(self) -> dict:
        return {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'lo-fi beats', 'cooking pasta'"
            }
        }

    def execute(self, query: str) -> str:
        err = _tv_adb_available()
        if err:
            return err
        cmd = (
            f"am start -a android.intent.action.SEARCH "
            f"-n com.google.android.youtube.tv/com.google.android.apps.youtube.tv.activity.ShellActivity "
            f"--es query {shlex.quote(query)}"
        )
        out, err = _adb_shell(cmd)
        if err:
            return f"YouTube search failed: {err}"
        return f"Searching YouTube for '{query}'"


