import asyncio
import pytest

from server.external._internal.async_runner import AsyncRunner


def test_runs_a_coroutine_and_returns_its_value():
    r = AsyncRunner()
    try:
        async def hello():
            await asyncio.sleep(0)
            return 42
        assert r.run(hello()) == 42
    finally:
        r.shutdown()


def test_propagates_exceptions_from_coroutine():
    r = AsyncRunner()
    try:
        async def boom():
            raise ValueError("nope")
        with pytest.raises(ValueError, match="nope"):
            r.run(boom())
    finally:
        r.shutdown()


def test_runs_multiple_sequential_coroutines_on_same_loop():
    r = AsyncRunner()
    try:
        async def n(x):
            await asyncio.sleep(0)
            return x * 2
        assert r.run(n(1)) == 2
        assert r.run(n(5)) == 10
        assert r.run(n(7)) == 14
    finally:
        r.shutdown()


def test_shutdown_is_idempotent():
    r = AsyncRunner()
    r.shutdown()
    r.shutdown()
