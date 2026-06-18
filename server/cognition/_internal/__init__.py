"""Private internals of cognition. Cannot be imported from outside cognition."""
import sys

_PREFIX = "server.cognition"
_OWN = __name__  # "server.cognition._internal"


def _check_caller() -> None:
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        # Skip own frame and importlib machinery
        if (name
            and name != _OWN
            and not name.startswith(("importlib", "_frozen_importlib"))):
            if not name.startswith(_PREFIX):
                raise ImportError(
                    f"server.cognition._internal is private; "
                    f"importing from '{name}' is forbidden. "
                    f"Use server.cognition.contracts instead."
                )
            return
        frame = frame.f_back


_check_caller()
