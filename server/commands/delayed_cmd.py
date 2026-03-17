"""Delayed command execution — schedule any command to run after a delay.

"Turn off the lights in 15 minutes" → delayed_command(command="set_light",
params='{"power": "off", "label": "office"}', delay="15 minutes").

Uses the existing timer/event loop infrastructure. The timer fires after
the delay and executes the command through the normal command dispatcher.
A TTS announcement plays when the delayed command executes.
"""
import json
import logging

from .base import Command
from .timer_cmd import parse_duration
from server.event_loop import get_event_loop

logger = logging.getLogger(__name__)

# Commands that are safe to run on a delay (no interactive confirmation needed)
_ALLOWED_DELAYED = frozenset({
    "set_light", "set_brightness", "set_color", "set_color_temp",
    "adjust_brightness", "adjust_color_temp", "shift_hue",
    "set_scene",
    "set_sonos_volume", "adjust_sonos_volume", "sonos_mute",
    "tv_power", "tv_key", "tv_launch", "tv_playback",
    "set_volume", "adjust_volume",
})


class DelayedCommandCommand(Command):
    name = "delayed_command"
    description = (
        "Schedule a command to execute after a delay. Use for requests like "
        "'turn off the lights in 15 minutes' or 'mute the TV in an hour'. "
        "The command runs automatically when the timer expires."
    )

    @property
    def parameters(self) -> dict:
        return {
            "command": {
                "type": "string",
                "description": "The command to execute later (e.g. 'set_light', 'tv_power')"
            },
            "params": {
                "type": "string",
                "description": 'JSON object of command parameters (e.g. \'{"power": "off", "label": "office"}\')'
            },
            "delay": {
                "type": "string",
                "description": "How long to wait (e.g. '15 minutes', '1 hour', '30 seconds')"
            },
        }

    def execute(self, command: str, params: str, delay: str, _ctx=None) -> str:
        command = command.strip()

        if command not in _ALLOWED_DELAYED:
            return f"Cannot delay '{command}' — only hardware/device commands can be delayed."

        # Parse the delay duration
        seconds = parse_duration(delay)
        if seconds is None or seconds <= 0:
            return f"Could not parse delay: '{delay}'"
        if seconds > 86400:
            return "Maximum delay is 24 hours"

        # Parse the command parameters
        try:
            cmd_params = json.loads(params) if params.strip() else {}
        except json.JSONDecodeError:
            return f"Invalid params JSON: {params}"

        # Build the callback that executes the command when timer fires
        from server.commands import execute as execute_command

        def _on_expire(timer_name: str):
            try:
                result = execute_command(command, _ctx=_ctx, **cmd_params)
                logger.info(f"Delayed command executed: {command}({cmd_params}) → {result}")
            except Exception as e:
                logger.error(f"Delayed command failed: {command}({cmd_params}) → {e}")

        # Create a timer with the callback
        timer_name = f"{command}:{delay}"
        event_loop = get_event_loop()
        room_id = _ctx.room.room_id if _ctx and hasattr(_ctx, 'room') else None

        if not event_loop.add_timer(timer_name, seconds, callback=_on_expire, room_id=room_id):
            return f"A delayed '{command}' is already scheduled. Cancel it first."

        # Format confirmation
        if seconds >= 3600:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            delay_str = f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minute{'s' if m != 1 else ''}" if m else "")
        elif seconds >= 60:
            m, s = int(seconds // 60), int(seconds % 60)
            delay_str = f"{m} minute{'s' if m != 1 else ''}" + (f" {s} second{'s' if s != 1 else ''}" if s else "")
        else:
            delay_str = f"{int(seconds)} second{'s' if int(seconds) != 1 else ''}"

        # Human-readable description of what will happen
        param_desc = ", ".join(f"{k}={v}" for k, v in cmd_params.items())
        return f"Scheduled: {command}({param_desc}) in {delay_str}"
