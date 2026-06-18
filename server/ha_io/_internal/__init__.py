"""Private internals of ha_io. Cannot be imported from outside ha_io."""
import sys

_PREFIX = "server.ha_io"
_OWN = __name__


def _check_caller() -> None:
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if (name
            and name != _OWN
            and not name.startswith(("importlib", "_frozen_importlib"))):
            if not (name.startswith(_PREFIX) or name.startswith("tests.")):
                raise ImportError(
                    f"server.ha_io._internal is private; "
                    f"importing from '{name}' is forbidden. "
                    f"Use server.ha_io.contracts instead."
                )
            return
        frame = frame.f_back


_check_caller()
