"""Private internals of wakeword. Cannot be imported from outside wakeword."""
import sys

_PREFIX = "wakeword"
_OWN = __name__


def _check_caller() -> None:
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if (name
            and name != _OWN
            and not name.startswith(("importlib", "_frozen_importlib"))):
            if not name.startswith(_PREFIX):
                raise ImportError(
                    f"wakeword._internal is private; "
                    f"importing from '{name}' is forbidden."
                )
            return
        frame = frame.f_back


_check_caller()
