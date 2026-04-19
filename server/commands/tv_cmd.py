"""HA-backed TV commands. Replaces the old tv_cmd.py (androidtvremote2)
and adb_cmd.py (adb-shell) — both now subsumed by the
`androidtv_remote` integration in Home Assistant.

HA exposes:
  - remote.<tv>           → send_command, turn_on/off, plus `activity:` deep-link launch
  - media_player.<tv>     → play_media (apps), volume, play/pause/next/etc.
                            Already covered by media_cmd.py for shared services.

Service field names verified against /api/services on the live HA (2026-04-18):
  remote.send_command  → device, command, num_repeats, delay_secs, hold_secs
  remote.turn_on       → activity (Android intent URL or package/activity)

Key naming: the androidtv_remote integration accepts AndroidKeyEvent constants
without the KEYCODE_ prefix (e.g. DPAD_UP, MEDIA_PLAY_PAUSE).
"""
import logging
from typing import Optional
from urllib.parse import quote_plus

from .base import Command
from server.ha_client import HAError, get_client

logger = logging.getLogger(__name__)


# Friendly key names → AndroidKeyEvent (without KEYCODE_ prefix) accepted by remote.send_command.
KEY_MAP = {
    "play":        "MEDIA_PLAY",
    "pause":       "MEDIA_PAUSE",
    "play_pause":  "MEDIA_PLAY_PAUSE",
    "stop":        "MEDIA_STOP",
    "next":        "MEDIA_NEXT",
    "previous":    "MEDIA_PREVIOUS",
    "rewind":      "MEDIA_REWIND",
    "fast_forward": "MEDIA_FAST_FORWARD",
    "home":        "HOME",
    "back":        "BACK",
    "menu":        "MENU",
    "up":          "DPAD_UP",
    "down":        "DPAD_DOWN",
    "left":        "DPAD_LEFT",
    "right":       "DPAD_RIGHT",
    "select":      "DPAD_CENTER",
    "ok":          "DPAD_CENTER",
    "enter":       "DPAD_CENTER",
    "volume_up":   "VOLUME_UP",
    "volume_down": "VOLUME_DOWN",
    "mute":        "VOLUME_MUTE",
    "power":       "POWER",
    "channel_up":  "CHANNEL_UP",
    "channel_down": "CHANNEL_DOWN",
}

# App name → Android package. The androidtv_remote integration launches via
# `activity:` on remote.turn_on. For most apps the package alone is enough;
# HA resolves the launcher activity.
APP_PACKAGES = {
    "netflix":     "com.netflix.ninja",
    "youtube":     "com.google.android.youtube.tv",
    "smarttube":   "is.xyz.smarttube",
    "spotify":     "com.spotify.tv.android",
    "disney":      "com.disney.disneyplus",
    "disney+":     "com.disney.disneyplus",
    "hulu":        "com.hulu.plus",
    "max":         "com.hbo.hbonow",
    "hbo":         "com.hbo.hbonow",
    "prime":       "com.amazon.amazonvideo.livingroom",
    "amazon":      "com.amazon.amazonvideo.livingroom",
    "prime video": "com.amazon.amazonvideo.livingroom",
    "plex":        "com.plexapp.android",
    "twitch":      "tv.twitch.android.app",
    "apple tv":    "com.apple.atve.androidtv.appletv",
    "peacock":     "com.peacocktv.peacockandroid",
    "paramount":   "com.cbs.ott",
    "paramount+":  "com.cbs.ott",
}


def _resolve_tv_remote(_ctx=None) -> Optional[str]:
    """Resolve the remote.* entity for the current room, or first available."""
    ha = get_client()
    if _ctx is not None and getattr(_ctx, "room", None) is not None:
        ha_area = getattr(_ctx.room, "ha_area", "") or ""
        if ha_area:
            in_area = ha.entities_in_area(ha_area, domain="remote")
            if in_area:
                return in_area[0]
    remotes = ha.states_in_domain("remote")
    return remotes[0]["entity_id"] if remotes else None


def _resolve_tv_media_player(_ctx=None) -> Optional[str]:
    """Resolve the media_player.* (device_class=tv) entity for the current room."""
    ha = get_client()
    if _ctx is not None and getattr(_ctx, "room", None) is not None:
        ha_area = getattr(_ctx.room, "ha_area", "") or ""
        if ha_area:
            for eid in ha.entities_in_area(ha_area, domain="media_player"):
                state = next((s for s in ha.states_in_domain("media_player") if s["entity_id"] == eid), None)
                if state and (state.get("attributes") or {}).get("device_class") == "tv":
                    return eid
    for s in ha.states_in_domain("media_player"):
        if (s.get("attributes") or {}).get("device_class") == "tv":
            return s["entity_id"]
    return None


def _send_remote_command(remote_entity: str, command: str | list[str], num_repeats: int = 1) -> Optional[str]:
    """Call remote.send_command. Returns None on success, error string on failure."""
    try:
        get_client().call_service("remote", "send_command", {
            "entity_id": remote_entity,
            "command": command,
            "num_repeats": num_repeats,
        })
        return None
    except HAError as e:
        logger.warning(f"remote.send_command failed: {e}")
        return f"TV command failed: {e}"


# -- commands --

class TvPowerCommand(Command):
    name = "tv_power"
    description = "Turn the TV on, off, or toggle power"

    @property
    def parameters(self) -> dict:
        return {"state": {"type": "string", "description": "'on', 'off', or 'toggle'"}}

    def execute(self, state: str, _ctx=None) -> str:
        s = state.lower().strip()
        if s not in ("on", "off", "toggle"):
            return f"Unknown state '{state}'. Use: on, off, or toggle."
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        ha = get_client()
        try:
            if s == "toggle":
                ha.call_service("remote", "toggle", {"entity_id": remote})
            elif s == "on":
                ha.call_service("remote", "turn_on", {"entity_id": remote})
            else:
                ha.call_service("remote", "turn_off", {"entity_id": remote})
        except HAError as e:
            return f"TV power failed: {e}"
        return f"TV {s if s != 'toggle' else 'toggled'}"


class TvKeyCommand(Command):
    name = "tv_key"
    description = (
        "Send a remote key to the TV. Keys: play, pause, play_pause, stop, "
        "home, back, menu, up, down, left, right, select, "
        "volume_up, volume_down, mute, power, channel_up, channel_down, "
        "next, previous, rewind, fast_forward."
    )

    @property
    def parameters(self) -> dict:
        return {"key": {"type": "string", "description": "Key name (see description)"}}

    def execute(self, key: str, _ctx=None) -> str:
        key_lower = key.lower().strip()
        command = KEY_MAP.get(key_lower)
        if not command:
            valid = ", ".join(sorted(KEY_MAP))
            return f"Unknown key '{key}'. Valid keys: {valid}"
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        err = _send_remote_command(remote, command)
        return err or f"Sent {key_lower} to TV"


class TvLaunchCommand(Command):
    name = "tv_launch"
    description = (
        "Launch an app on the TV by name. "
        "Supported: netflix, youtube, smarttube, spotify, disney+, hulu, max, "
        "prime video, plex, twitch, apple tv, peacock, paramount+"
    )

    @property
    def parameters(self) -> dict:
        return {"app": {"type": "string", "description": "App name (e.g. 'youtube', 'netflix')"}}

    def execute(self, app: str, _ctx=None) -> str:
        app_lower = app.lower().strip()
        package = APP_PACKAGES.get(app_lower)
        if not package:
            return f"Unknown app '{app}'. Supported: {', '.join(sorted(APP_PACKAGES.keys()))}"
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        try:
            # remote.turn_on accepts `activity:` — for most apps the package
            # name alone is enough; HA resolves the launcher activity.
            get_client().call_service("remote", "turn_on", {
                "entity_id": remote,
                "activity": package,
            })
        except HAError as e:
            return f"Failed to launch {app}: {e}"
        return f"Launched {app} on TV"


class TvSkipCommand(Command):
    name = "tv_skip"
    description = (
        "Skip forward or back in the current video by a number of seconds. "
        "Each press skips ~5s. Default 30 seconds."
    )

    @property
    def parameters(self) -> dict:
        return {
            "direction": {"type": "string", "description": "'forward' or 'back'"},
            "seconds": {"type": "integer", "description": "Approximate seconds to skip (rounded to 5s steps)"},
        }

    @property
    def required_parameters(self) -> list:
        return ["direction"]

    def execute(self, direction: str, seconds: int = 30, _ctx=None) -> str:
        d = direction.lower().strip()
        if d in ("forward", "ahead", "right"):
            command = "DPAD_RIGHT"
        elif d in ("back", "backward", "rewind", "left"):
            command = "DPAD_LEFT"
        else:
            return f"Unknown direction '{direction}'. Use 'forward' or 'back'."
        try:
            seconds = int(seconds)
        except (ValueError, TypeError):
            return f"Invalid seconds value '{seconds}'. Use a number."
        presses = max(1, min(120, round(seconds / 5)))
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        # remote.send_command's num_repeats handles the loop natively
        err = _send_remote_command(remote, command, num_repeats=presses)
        return err or f"Skipped {d} ~{presses * 5}s on TV"


class TvSearchYouTubeCommand(Command):
    name = "tv_search_youtube"
    description = (
        "Open YouTube on the TV with a search query. "
        "Use when the user wants to find or watch something on YouTube."
    )

    @property
    def parameters(self) -> dict:
        return {"query": {"type": "string", "description": "Search query, e.g. 'lo-fi beats'"}}

    def execute(self, query: str, _ctx=None) -> str:
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        # YouTube TV deep link — opens the search page in the TV YouTube app
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        try:
            get_client().call_service("remote", "turn_on", {
                "entity_id": remote,
                "activity": url,
            })
        except HAError as e:
            return f"YouTube search failed: {e}"
        return f"Searching YouTube for '{query}'"


class TvCurrentAppCommand(Command):
    name = "tv_current_app"
    description = "Report what app or activity is currently shown on the TV."

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, _ctx=None, **_) -> str:
        remote = _resolve_tv_remote(_ctx)
        if not remote:
            return "No TV remote available in Home Assistant"
        try:
            state = get_client().get_state(remote)
        except HAError as e:
            return f"Failed to read TV state: {e}"
        attrs = state.get("attributes", {})
        activity = attrs.get("current_activity") or "(none)"
        # Try to give a friendly app name when we can map the package back
        for name, pkg in APP_PACKAGES.items():
            if pkg in activity:
                return f"TV is showing {name} ({activity})"
        return f"TV is showing {activity}"
