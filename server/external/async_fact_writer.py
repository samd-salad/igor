"""Background writer for facts. The LLM tool-call returns within ~ms; the
encode + DB write happens on a single worker thread that drains a FIFO queue.

Matches Letta's primary/sleep-time split — the hot path doesn't block on
encoder + disk I/O.

The writer owns the encoder. Persistence stores whatever bytes Fact.embedding
carries; encoding is a cognition-layer concern that doesn't belong inside the
SQL adapter. When the writer is configured with an encoder, it encodes any
Fact arriving with embedding=None before handing it off to persistence.

Threading invariants (the tests rely on these):
- Every put() increments unfinished_tasks
- The worker calls task_done() for EVERY get() (including the sentinel)
- So queue.join() blocks until every enqueued item has been processed
"""
from __future__ import annotations
import logging
import queue
import threading
from dataclasses import replace
from typing import Optional, Protocol

from server.cognition.contracts import Fact

_log = logging.getLogger(__name__)
_SENTINEL = object()


class _Encoder(Protocol):
    def encode(self, text: str) -> bytes: ...


class AsyncFactWriter:
    def __init__(self, persistence, *, encoder: Optional[_Encoder] = None):
        self._persistence = persistence
        self._encoder = encoder
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="igor-fact-writer", daemon=True
        )
        self._thread.start()

    def enqueue(self, fact: Fact) -> None:
        self._queue.put(fact)

    def flush(self, timeout: float = 5.0) -> None:
        """Block until every enqueued fact has been processed.
        queue.Queue.join has no timeout — wrap it in a watcher thread + Event."""
        done = threading.Event()
        def waiter():
            self._queue.join()
            done.set()
        threading.Thread(target=waiter, daemon=True).start()
        if not done.wait(timeout):
            raise TimeoutError("AsyncFactWriter.flush timeout")

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                try:
                    fact = item
                    if self._encoder is not None and fact.embedding is None:
                        fact = replace(fact, embedding=self._encoder.encode(fact.value))
                    self._persistence.save_fact(fact)
                except Exception:
                    _log.exception("AsyncFactWriter: save_fact failed")
            finally:
                self._queue.task_done()
