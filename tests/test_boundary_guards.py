"""Verify that _internal modules cannot be imported from outside their context."""
import importlib
import sys
import pytest


def _import_from_fake_caller(target: str, caller_name: str = "evil.attacker"):
    """Simulate import from a non-tests, non-context caller."""
    sys.modules.pop(target, None)
    code = compile(
        f"import importlib; importlib.import_module({target!r})",
        f"<{caller_name}>", "exec",
    )
    exec(code, {"__name__": caller_name, "__builtins__": __builtins__})


def test_cognition_internal_blocked_from_outside():
    with pytest.raises(ImportError, match="private"):
        _import_from_fake_caller("server.cognition._internal")


def test_ha_io_internal_blocked_from_outside():
    with pytest.raises(ImportError, match="private"):
        _import_from_fake_caller("server.ha_io._internal")


def test_external_internal_blocked_from_outside():
    with pytest.raises(ImportError, match="private"):
        _import_from_fake_caller("server.external._internal")
