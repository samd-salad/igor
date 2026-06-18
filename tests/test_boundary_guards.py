"""Verify that _internal modules cannot be imported from outside their context."""
import importlib
import sys
import pytest


def test_cognition_internal_blocked_from_outside():
    sys.modules.pop("server.cognition._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.cognition._internal")


def test_ha_io_internal_blocked_from_outside():
    sys.modules.pop("server.ha_io._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.ha_io._internal")


def test_external_internal_blocked_from_outside():
    sys.modules.pop("server.external._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.external._internal")
