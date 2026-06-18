"""Background asyncio loop in a daemon thread, with a sync run(coro) facade.

Used by adapters (e.g. ha_mcp_executor) whose underlying libraries are async,
so the sync ToolExecutorPort can still call them without leaking asyncio into
cognition or ha_io."""
from __future__ import annotations
import asyncio
import threading
from typing import Any, Coroutine


class AsyncRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="AsyncRunner",
            daemon=True,
        )
        self._thread.start()
        self._stopped = False

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Submit `coro` to the background loop and block until done."""
        if self._stopped:
            raise RuntimeError("AsyncRunner is shut down")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
