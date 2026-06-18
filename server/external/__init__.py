"""Public surface of the external context.

Re-exports HAClient so the composition root and (transitionally) server.commands
can reach it without touching _internal/ directly.
"""
from server.external._internal.ha_client import get_client, HAClient, HAError

__all__ = ["get_client", "HAClient", "HAError"]
