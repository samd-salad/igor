# Igor Tool Registry — HA MCP Client + Cognition-Native Tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Igor's tool catalog by delegating all device/HA-side actions to Home Assistant's MCP Server (no per-device code in Igor) while keeping a small Igor-native registry for cognition-only tools (memory, feedback, weather, calc).

**Architecture:**
- `HAMCPToolExecutor` opens a streamable-HTTP MCP session to `http://10.0.40.5:8123/api/mcp` with the existing `HA_TOKEN`, caches HA's tool list at startup, translates each MCP tool's `inputSchema` into Anthropic's tool schema shape, and forwards `tool_use → tools/call → tool_result`.
- `IgorNativeToolExecutor` exposes a tiny in-process registry for tools that should never leave Igor: `save_memory`, `forget_memory`, `log_feedback`, `get_weather`, `calculate`.
- `CompositeToolExecutor` is the single `ToolExecutorPort` implementation Conversation sees — it concatenates schemas and routes `execute(name, …)` by tool name.
- MCP's async client lives behind a sync `AsyncRunner` (background asyncio loop in a daemon thread) so the existing sync conversation pipeline stays intact.

**Tech Stack:** Python 3.12+, `mcp` SDK (Anthropic), `anyio`/`asyncio`, existing FastAPI + SQLite stack.

## Global Constraints

- The `mcp` library is third-party — ONLY `server/external/` is allowed to import it. `tools/boundary_check.py` must enforce this.
- All HA device/area/entity awareness comes from HA. Igor never hardcodes a service name (`light.turn_on`), entity_id, or area name.
- HA-side blast radius is bounded by HA's existing "Expose to voice" config — that is the user's safety surface, not Igor's.
- Sync `ToolExecutorPort` interface is unchanged (Conversation is sync). Async happens entirely inside `server/external/`.
- VoiceTurn flows through `execute(name, args, turn)` as today — MCP tools don't need it (HA owns its own context) but Igor-native tools do.
- No new test dependencies. Use stub MCP clients + monkeypatch.
- Frequent commits — one per task.

---

## File Structure

**New files:**
- `server/external/_internal/async_runner.py` — daemon-thread asyncio loop with `run(coro)` blocking submit
- `server/external/_internal/mcp_session.py` — async helpers: `open_session(url, token)`, `list_remote_tools(session)`, `call_remote_tool(session, name, args)`
- `server/external/ha_mcp_executor.py` — sync `HAMCPToolExecutor` implementing `ToolExecutorPort`
- `server/external/igor_native_tools.py` — `IgorTool` dataclass + `IgorNativeToolExecutor` + factory `build_native_registry(memory, user_state, weather)`
- `server/external/composite_executor.py` — `CompositeToolExecutor` combining native + MCP
- `server/external/weather_open_meteo.py` — Open-Meteo client (lifted from legacy `weather_cmd.py`)
- `tests/external/test_async_runner.py`
- `tests/external/test_mcp_session.py`
- `tests/external/test_ha_mcp_executor.py`
- `tests/external/test_igor_native_tools.py`
- `tests/external/test_composite_executor.py`

**Modified files:**
- `requirements-server-text.txt` — add `mcp>=1.2.0`
- `tools/boundary_check.py` — add `mcp` to `THIRD_PARTY_LOCKED_TO_EXTERNAL`
- `server/main.py` — swap `HARestToolExecutor()` for composite assembly
- `server/external/ha_rest_adapter.py` — DELETED (Task 8)

---

### Task 1: Add MCP dependency and boundary lock

**Files:**
- Modify: `requirements-server-text.txt` (add `mcp>=1.2.0`)
- Modify: `tools/boundary_check.py:29-35` (extend `THIRD_PARTY_LOCKED_TO_EXTERNAL`)
- Modify: `tools/boundary_check.py:19-28` (extend `HA_IO_FORBIDDEN`, `COGNITION_FORBIDDEN`)

**Interfaces:**
- Produces: `mcp` package importable inside `server/external/` only; CI blocks cognition/ha_io imports.

- [ ] **Step 1: Add `mcp` to requirements**

Open `requirements-server-text.txt` and append:

```
mcp>=1.2.0
```

- [ ] **Step 2: Install in local venv**

```bash
.venv/Scripts/python.exe -m pip install -r requirements-server-text.txt
```
Expected: `mcp` and its transitive deps install cleanly.

- [ ] **Step 3: Lock `mcp` to `server/external/` only**

In `tools/boundary_check.py`, extend the forbidden sets and add `mcp` to the per-file allowlist:

```python
COGNITION_FORBIDDEN = {
    "server.external",
    "server.ha_io",
    "anthropic",
    "requests",
    "sqlite3",
    "fastapi",
    "mcp",
}
HA_IO_FORBIDDEN = {
    "server.external",
    "server.cognition.ports",
    "server.cognition.aggregates",
    "server.cognition.services",
    "server.cognition._internal",
    "anthropic",
    "requests",
    "sqlite3",
    "mcp",
}
THIRD_PARTY_LOCKED_TO_EXTERNAL = {
    "anthropic": ("server/external/claude_adapter.py",),
    "sqlite3":   ("server/external/sqlite_persistence.py",
                  "server/external/sqlite_retrieval.py",
                  "server/external/_internal/db.py",
                  "server/external/_internal/brain_json_migration.py"),
    "mcp":       ("server/external/_internal/mcp_session.py",
                  "server/external/ha_mcp_executor.py"),
}
```

- [ ] **Step 4: Run boundary check**

```bash
.venv/Scripts/python.exe -m tools.boundary_check
```
Expected: `Boundary check passed.` (no new imports yet — the lock is preventative.)

- [ ] **Step 5: Run full tests**

```bash
.venv/Scripts/python.exe -m pytest
```
Expected: all current tests still pass (e.g. 79+).

- [ ] **Step 6: Commit**

```bash
git add requirements-server-text.txt tools/boundary_check.py
git commit -m "tools/boundary_check: lock mcp library to server/external"
```

---

### Task 2: AsyncRunner — sync→async bridge

**Files:**
- Create: `server/external/_internal/async_runner.py`
- Create: `tests/external/test_async_runner.py`

**Interfaces:**
- Produces: `class AsyncRunner` with `__init__()`, `run(coro) -> Any` (sync, blocking), `shutdown() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_async_runner.py`:

```python
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
    r.shutdown()  # should not raise
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_async_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'server.external._internal.async_runner'`.

- [ ] **Step 3: Implement**

Create `server/external/_internal/async_runner.py`:

```python
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
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_async_runner.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/_internal/async_runner.py tests/external/test_async_runner.py
git commit -m "external/_internal: AsyncRunner — sync/async bridge for MCP client"
```

---

### Task 3: MCP session helpers

**Files:**
- Create: `server/external/_internal/mcp_session.py`
- Create: `tests/external/test_mcp_session.py`

**Interfaces:**
- Produces:
  - `async def fetch_tool_catalog(url: str, token: str) -> list[McpTool]` — opens session, calls `list_tools()`, returns list of dataclasses `McpTool(name, description, input_schema)`.
  - `async def invoke_tool(url: str, token: str, name: str, arguments: dict) -> str` — opens session, calls `call_tool`, returns a string suitable for `tool_result.content`.
  - `@dataclass(frozen=True) class McpTool: name: str; description: str; input_schema: dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_mcp_session.py`. We test the dataclass shape and the schema-translation seam; the actual transport is covered by integration smoke test (Task 8).

```python
import pytest

from server.external._internal.mcp_session import McpTool, _content_to_text


def test_mcp_tool_is_frozen_dataclass():
    t = McpTool(name="HassTurnOn", description="Turn something on",
                input_schema={"type": "object", "properties": {}})
    assert t.name == "HassTurnOn"
    with pytest.raises(Exception):
        t.name = "other"  # frozen


def test_content_to_text_concatenates_text_blocks():
    class _TB:
        type = "text"
        text = "Lights on."
    class _TB2:
        type = "text"
        text = " Done."
    out = _content_to_text([_TB(), _TB2()])
    assert out == "Lights on. Done."


def test_content_to_text_falls_back_to_str_for_unknown_block():
    class _Img:
        type = "image"
        def __str__(self): return "<image>"
    assert _content_to_text([_Img()]) == "<image>"


def test_content_to_text_handles_empty_list():
    assert _content_to_text([]) == ""
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_mcp_session.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `server/external/_internal/mcp_session.py`:

```python
"""Async helpers around the `mcp` SDK. Lives in _internal so the file-locked
third-party `mcp` import doesn't leak to siblings. Used by HAMCPToolExecutor
behind AsyncRunner."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict


def _content_to_text(content_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "".join(parts)


async def fetch_tool_catalog(url: str, token: str) -> list[McpTool]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                McpTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                for t in result.tools
            ]


async def invoke_tool(url: str, token: str, name: str, arguments: dict) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name=name, arguments=arguments)
            if result.isError:
                return f"Error from HA: {_content_to_text(result.content)}"
            return _content_to_text(result.content) or "(no output)"
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_mcp_session.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Boundary check**

```bash
.venv/Scripts/python.exe -m tools.boundary_check
```
Expected: `Boundary check passed.` (mcp is now imported from the allowed file).

- [ ] **Step 6: Commit**

```bash
git add server/external/_internal/mcp_session.py tests/external/test_mcp_session.py
git commit -m "external/_internal: mcp_session helpers — list_tools / call_tool over streamable HTTP"
```

---

### Task 4: HAMCPToolExecutor — sync façade implementing ToolExecutorPort

**Files:**
- Create: `server/external/ha_mcp_executor.py`
- Create: `tests/external/test_ha_mcp_executor.py`

**Interfaces:**
- Consumes: `AsyncRunner`, `fetch_tool_catalog`, `invoke_tool`, `McpTool`.
- Produces:
  - `class HAMCPToolExecutor:` constructor `__init__(url: str, token: str, runner: AsyncRunner)`.
  - `list_schemas() -> list[dict]` — Anthropic tool schemas.
  - `execute(name: str, args: dict, turn: VoiceTurn) -> str` — forwards to HA.
  - `handles(name: str) -> bool` — for routing by `CompositeToolExecutor`.
  - `refresh() -> None` — re-fetch tool catalog (called once at construction).

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_ha_mcp_executor.py`. We stub the network seam by monkeypatching the two async helpers.

```python
from datetime import datetime, UTC
from unittest.mock import patch

import pytest

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.external._internal.async_runner import AsyncRunner
from server.external._internal.mcp_session import McpTool
from server.external.ha_mcp_executor import HAMCPToolExecutor


def _turn(text: str = "irrelevant") -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1",
        started_at=datetime(2026, 6, 18, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


@pytest.fixture
def runner():
    r = AsyncRunner()
    yield r
    r.shutdown()


def test_caches_tool_catalog_on_construction(runner):
    catalog = [
        McpTool(name="HassTurnOn", description="Turn on a device",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}}),
        McpTool(name="HassTurnOff", description="Turn off a device",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}}),
    ]
    async def fake_fetch(url, token):
        return catalog
    with patch("server.external.ha_mcp_executor.fetch_tool_catalog", new=fake_fetch):
        ex = HAMCPToolExecutor("http://10.0.40.5:8123/api/mcp", "tok", runner)
    schemas = ex.list_schemas()
    names = [s["name"] for s in schemas]
    assert names == ["HassTurnOn", "HassTurnOff"]
    for s in schemas:
        assert "description" in s
        assert s["input_schema"]["type"] == "object"


def test_handles_returns_true_for_cached_tool_name(runner):
    async def fake_fetch(url, token):
        return [McpTool("HassTurnOn", "x", {"type": "object", "properties": {}})]
    with patch("server.external.ha_mcp_executor.fetch_tool_catalog", new=fake_fetch):
        ex = HAMCPToolExecutor("u", "t", runner)
    assert ex.handles("HassTurnOn")
    assert not ex.handles("save_memory")


def test_execute_forwards_to_invoke_tool(runner):
    async def fake_fetch(url, token):
        return [McpTool("HassTurnOn", "x", {"type": "object", "properties": {}})]
    captured: dict = {}
    async def fake_invoke(url, token, name, arguments):
        captured["url"] = url
        captured["name"] = name
        captured["args"] = arguments
        return "Lights on."
    with patch("server.external.ha_mcp_executor.fetch_tool_catalog", new=fake_fetch), \
         patch("server.external.ha_mcp_executor.invoke_tool", new=fake_invoke):
        ex = HAMCPToolExecutor("http://ha/api/mcp", "tok", runner)
        result = ex.execute("HassTurnOn", {"name": "Kitchen Lights"}, _turn())
    assert result == "Lights on."
    assert captured == {"url": "http://ha/api/mcp",
                        "name": "HassTurnOn",
                        "args": {"name": "Kitchen Lights"}}


def test_execute_returns_message_for_unknown_tool(runner):
    async def fake_fetch(url, token):
        return []
    with patch("server.external.ha_mcp_executor.fetch_tool_catalog", new=fake_fetch):
        ex = HAMCPToolExecutor("u", "t", runner)
    out = ex.execute("nonexistent", {}, _turn())
    assert "nonexistent" in out.lower()
    assert "unknown" in out.lower() or "not found" in out.lower()


def test_construction_survives_unreachable_ha(runner):
    """If HA is down at startup, the adapter must still construct (empty
    catalog). Otherwise Igor can't boot when HA is offline."""
    async def fake_fetch(url, token):
        raise ConnectionError("ha unreachable")
    with patch("server.external.ha_mcp_executor.fetch_tool_catalog", new=fake_fetch):
        ex = HAMCPToolExecutor("u", "t", runner)
    assert ex.list_schemas() == []
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_ha_mcp_executor.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `server/external/ha_mcp_executor.py`:

```python
"""ToolExecutorPort impl that delegates to Home Assistant's MCP Server.

Caches HA's tool catalog at startup, translates each MCP tool's inputSchema
into Anthropic's tool schema shape, and forwards tool_use → tools/call →
tool_result. No per-device code — every new HA integration the user installs
becomes a callable tool here automatically (subject to HA's expose-to-voice)."""
from __future__ import annotations
import logging
from typing import Optional

from server.cognition.contracts import VoiceTurn
from server.external._internal.async_runner import AsyncRunner
from server.external._internal.mcp_session import (
    McpTool, fetch_tool_catalog, invoke_tool,
)

logger = logging.getLogger(__name__)


class HAMCPToolExecutor:
    def __init__(self, url: str, token: str, runner: AsyncRunner):
        self._url = url
        self._token = token
        self._runner = runner
        self._catalog: list[McpTool] = []
        self._by_name: dict[str, McpTool] = {}
        self.refresh()

    def refresh(self) -> None:
        try:
            self._catalog = self._runner.run(fetch_tool_catalog(self._url, self._token))
        except Exception as e:
            logger.warning("HA MCP catalog fetch failed (HA offline?): %s", e)
            self._catalog = []
        self._by_name = {t.name: t for t in self._catalog}
        logger.info("HA MCP catalog: %d tool(s) cached", len(self._catalog))

    def handles(self, name: str) -> bool:
        return name in self._by_name

    def list_schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._catalog
        ]

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        if name not in self._by_name:
            return f"Unknown HA tool: {name}"
        try:
            return self._runner.run(invoke_tool(self._url, self._token, name, args))
        except Exception as e:
            logger.exception("MCP call_tool(%s) failed", name)
            return f"Error calling {name}: {e}"
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_ha_mcp_executor.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/ha_mcp_executor.py tests/external/test_ha_mcp_executor.py
git commit -m "external: HAMCPToolExecutor — Igor's tool registry comes from HA's MCP server"
```

---

### Task 5: Weather adapter (lifted from legacy weather_cmd)

**Files:**
- Create: `server/external/weather_open_meteo.py`
- Create: `tests/external/test_weather_open_meteo.py`

**Interfaces:**
- Produces: `class OpenMeteoWeather:` with `current(location: str) -> str` returning a spoken-friendly sentence ("65°F and partly cloudy in Arlington, VA"). Uses `requests` (already a dependency).

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_weather_open_meteo.py`:

```python
from unittest.mock import patch, MagicMock

from server.external.weather_open_meteo import OpenMeteoWeather


def _stub_response(json_payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_payload
    r.raise_for_status = MagicMock()
    return r


def test_current_returns_spoken_summary():
    geocode = _stub_response({
        "results": [{"latitude": 38.8, "longitude": -77.1, "name": "Arlington"}]
    })
    forecast = _stub_response({
        "current": {"temperature_2m": 18.3, "weather_code": 2},
    })
    with patch("server.external.weather_open_meteo.requests.get",
               side_effect=[geocode, forecast]):
        w = OpenMeteoWeather()
        out = w.current("Arlington, VA")
    assert "Arlington" in out
    assert "F" in out  # Fahrenheit
    assert any(word in out.lower() for word in ("cloud", "partly"))


def test_current_handles_unknown_location():
    geocode = _stub_response({"results": []})
    with patch("server.external.weather_open_meteo.requests.get",
               return_value=geocode):
        w = OpenMeteoWeather()
        out = w.current("Atlantis")
    assert "atlantis" in out.lower() or "couldn't find" in out.lower() \
        or "unknown" in out.lower()


def test_current_handles_api_failure_gracefully():
    geocode = _stub_response({"results": [{"latitude": 0, "longitude": 0,
                                           "name": "X"}]})
    failed = MagicMock()
    failed.raise_for_status.side_effect = Exception("502")
    with patch("server.external.weather_open_meteo.requests.get",
               side_effect=[geocode, failed]):
        w = OpenMeteoWeather()
        out = w.current("X")
    assert "weather" in out.lower()  # graceful fallback message
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_weather_open_meteo.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `server/external/weather_open_meteo.py`:

```python
"""Open-Meteo weather lookup. No API key required. Returns spoken-friendly
strings — never structured data — because the consumer is a voice tool."""
from __future__ import annotations
import logging

import requests

logger = logging.getLogger(__name__)


# https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
}


def _c_to_f(c: float) -> int:
    return round(c * 9 / 5 + 32)


class OpenMeteoWeather:
    def current(self, location: str) -> str:
        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
                timeout=5,
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return f"I couldn't find {location}."
            place = results[0]
            lat, lon, name = place["latitude"], place["longitude"], place["name"]
            forecast = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "current": "temperature_2m,weather_code"},
                timeout=5,
            )
            forecast.raise_for_status()
            cur = forecast.json().get("current", {})
            temp_c = cur.get("temperature_2m")
            code = cur.get("weather_code", 0)
            condition = _WEATHER_CODES.get(code, "unknown")
            return f"{_c_to_f(temp_c)}°F and {condition} in {name}."
        except Exception:
            logger.exception("Weather lookup failed for %s", location)
            return f"Weather lookup failed for {location}."
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_weather_open_meteo.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/weather_open_meteo.py tests/external/test_weather_open_meteo.py
git commit -m "external: weather_open_meteo — keyless weather adapter, voice-formatted output"
```

---

### Task 6: Igor-native tools registry

**Files:**
- Create: `server/external/igor_native_tools.py`
- Create: `tests/external/test_igor_native_tools.py`

**Interfaces:**
- Consumes: `MemoryStore`, `UserState`, `OpenMeteoWeather`, `VoiceTurn`.
- Produces:
  - `@dataclass(frozen=True) class IgorTool: name: str; description: str; input_schema: dict; handler: Callable[[dict, VoiceTurn], str]`
  - `class IgorNativeToolExecutor:` with `list_schemas() -> list[dict]`, `execute(name, args, turn) -> str`, `handles(name) -> bool`.
  - `def build_native_registry(memory: MemoryStore, user_state: UserState, weather: OpenMeteoWeather, default_location: str) -> IgorNativeToolExecutor`.
  - Tools registered: `save_memory`, `forget_memory`, `log_feedback`, `get_weather`, `calculate`.

- [ ] **Step 1: Inspect MemoryStore / UserState surface**

```bash
.venv/Scripts/python.exe -c "from server.cognition.aggregates.memory import MemoryStore; help(MemoryStore)"
.venv/Scripts/python.exe -c "from server.cognition.aggregates.user_state import UserState; help(UserState)"
```
Confirm method signatures before referencing them. If method names differ from `save_fact` / `invalidate_fact` / `log_feedback`, adjust the code in Step 3 to match.

- [ ] **Step 2: Write the failing test**

Create `tests/external/test_igor_native_tools.py`:

```python
from datetime import datetime, UTC
from unittest.mock import MagicMock

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.external.igor_native_tools import IgorTool, IgorNativeToolExecutor, build_native_registry


def _turn(text: str = "x") -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1",
        started_at=datetime(2026, 6, 18, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_executor_lists_registered_tool_schemas():
    tool = IgorTool(
        name="ping",
        description="say pong",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, turn: "pong",
    )
    ex = IgorNativeToolExecutor([tool])
    schemas = ex.list_schemas()
    assert schemas == [{
        "name": "ping",
        "description": "say pong",
        "input_schema": {"type": "object", "properties": {}},
    }]
    assert ex.handles("ping")
    assert not ex.handles("unknown")


def test_executor_routes_execute_to_handler_with_args_and_turn():
    captured = {}
    def handler(args, turn):
        captured["args"] = args
        captured["turn"] = turn
        return "ok"
    tool = IgorTool("echo", "echo", {"type": "object", "properties": {}}, handler)
    ex = IgorNativeToolExecutor([tool])
    out = ex.execute("echo", {"hello": "world"}, _turn("hi"))
    assert out == "ok"
    assert captured["args"] == {"hello": "world"}
    assert captured["turn"].input_text == "hi"


def test_executor_returns_message_for_unknown_tool():
    ex = IgorNativeToolExecutor([])
    out = ex.execute("nope", {}, _turn())
    assert "unknown" in out.lower() or "not found" in out.lower()


def test_build_native_registry_includes_expected_tools():
    memory = MagicMock()
    user_state = MagicMock()
    weather = MagicMock()
    weather.current.return_value = "65°F and clear in Arlington."
    ex = build_native_registry(memory=memory, user_state=user_state,
                               weather=weather, default_location="Arlington, VA")
    names = {s["name"] for s in ex.list_schemas()}
    assert {"save_memory", "forget_memory", "log_feedback",
            "get_weather", "calculate"} <= names


def test_get_weather_uses_default_when_no_location_argument():
    weather = MagicMock()
    weather.current.return_value = "70°F and sunny in Arlington."
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=weather, default_location="Arlington, VA")
    out = ex.execute("get_weather", {}, _turn())
    weather.current.assert_called_once_with("Arlington, VA")
    assert "Arlington" in out


def test_get_weather_uses_argument_when_provided():
    weather = MagicMock()
    weather.current.return_value = "x"
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=weather, default_location="Arlington, VA")
    ex.execute("get_weather", {"location": "Tokyo"}, _turn())
    weather.current.assert_called_once_with("Tokyo")


def test_calculate_evaluates_simple_arithmetic():
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("calculate", {"expression": "15 * 0.18"}, _turn())
    assert "2.7" in out


def test_calculate_rejects_dangerous_expressions():
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("calculate", {"expression": "__import__('os').system('echo bad')"},
                     _turn())
    assert "can't" in out.lower() or "invalid" in out.lower() or "error" in out.lower()
```

- [ ] **Step 3: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_igor_native_tools.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Create `server/external/igor_native_tools.py`. Note: `calculate` uses a restricted AST evaluator — never `eval()` or `exec()` (security rule from CLAUDE.md).

```python
"""Igor-native tool registry. These tools live in Igor (not HA) because they
operate on cognition aggregates or external services HA doesn't expose:
memory writes, feedback logging, weather, arithmetic.

Each tool takes (args: dict, turn: VoiceTurn) and returns a string suitable
for the LLM's tool_result.content."""
from __future__ import annotations
import ast
import logging
import operator
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import Fact, FeedbackEntry, VoiceTurn
from server.external.weather_open_meteo import OpenMeteoWeather

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IgorTool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, VoiceTurn], str]


class IgorNativeToolExecutor:
    def __init__(self, tools: list[IgorTool]):
        self._tools = list(tools)
        self._by_name = {t.name: t for t in self._tools}

    def handles(self, name: str) -> bool:
        return name in self._by_name

    def list_schemas(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema}
            for t in self._tools
        ]

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        tool = self._by_name.get(name)
        if tool is None:
            return f"Unknown native tool: {name}"
        try:
            return tool.handler(args, turn)
        except Exception as e:
            logger.exception("Native tool %s failed", name)
            return f"Error in {name}: {e}"


# ---------- Tool factories (closures over aggregates) ----------

def _save_memory(memory: MemoryStore) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        category = args["category"]
        key = args["key"]
        value = args["value"]
        tags = args.get("tags") or []
        now = datetime.now(UTC)
        memory.save_fact(Fact(
            fact_id=str(uuid.uuid4()),
            category=category, key=key, value=value,
            tags=list(tags),
            source_episode_id=turn.correlation_id,
            embedding=None,
            valid_at=now, invalid_at=None, created_at=now,
        ))
        return f"Remembered {category}/{key}."
    return IgorTool(
        name="save_memory",
        description="Save a fact about the user. Use when learning names, "
                    "preferences, schedules, relationships, or corrections.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "description": "preferences | schedule | people | personal | home | other"},
                "key": {"type": "string",
                        "description": "short lowercase underscored identifier"},
                "value": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["category", "key", "value"],
        },
        handler=handler,
    )


def _forget_memory(memory: MemoryStore) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        category = args["category"]
        key = args["key"]
        existing = memory.find_fact(category, key)
        if existing is None:
            return f"No memory at {category}/{key}."
        memory.invalidate_fact(existing.fact_id, at=datetime.now(UTC))
        return f"Forgot {category}/{key}."
    return IgorTool(
        name="forget_memory",
        description="Forget a previously-saved fact about the user.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "key": {"type": "string"},
            },
            "required": ["category", "key"],
        },
        handler=handler,
    )


def _log_feedback(user_state: UserState) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        issue = args["issue"]
        entry = FeedbackEntry(
            feedback_id=str(uuid.uuid4()),
            occurred_at=datetime.now(UTC),
            issue=issue,
            status="open",
            source_episode_id=turn.correlation_id,
        )
        user_state.add_feedback(entry)
        return "Logged."
    return IgorTool(
        name="log_feedback",
        description="Log a change request or correction from the user. "
                    "Use when the user says something went wrong or asks for a behavioral change.",
        input_schema={
            "type": "object",
            "properties": {
                "issue": {"type": "string"},
            },
            "required": ["issue"],
        },
        handler=handler,
    )


def _get_weather(weather: OpenMeteoWeather, default_location: str) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        loc = args.get("location") or default_location
        return weather.current(loc)
    return IgorTool(
        name="get_weather",
        description="Get current weather conditions for a location. "
                    "If the user doesn't specify a location, the default is used.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": [],
        },
        handler=handler,
    )


# Safe arithmetic via AST — operators allowlisted, no names, no calls.
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric literal")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _calculate() -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        expression = args.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval_arith(tree)
        except Exception:
            return "I can't evaluate that expression."
        # Round to 4 dp; strip trailing zeros for voice
        formatted = f"{result:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return IgorTool(
        name="calculate",
        description="Evaluate a simple arithmetic expression (+, -, *, /, %, **).",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
        handler=handler,
    )


def build_native_registry(memory: MemoryStore, user_state: UserState,
                          weather: OpenMeteoWeather,
                          default_location: str) -> IgorNativeToolExecutor:
    return IgorNativeToolExecutor([
        _save_memory(memory),
        _forget_memory(memory),
        _log_feedback(user_state),
        _get_weather(weather, default_location),
        _calculate(),
    ])
```

- [ ] **Step 5: Verify against actual aggregate signatures**

If Step 1 revealed that `MemoryStore.save_fact` / `find_fact` / `invalidate_fact` or `UserState.add_feedback` are named differently, update the closures accordingly and re-run the test.

- [ ] **Step 6: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_igor_native_tools.py -v
```
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add server/external/igor_native_tools.py tests/external/test_igor_native_tools.py
git commit -m "external: igor_native_tools — save/forget memory, feedback, weather, calc"
```

---

### Task 7: CompositeToolExecutor

**Files:**
- Create: `server/external/composite_executor.py`
- Create: `tests/external/test_composite_executor.py`

**Interfaces:**
- Consumes: any object with `list_schemas() -> list[dict]`, `execute(name, args, turn) -> str`, `handles(name) -> bool`.
- Produces: `class CompositeToolExecutor:` implementing `ToolExecutorPort`. Constructor takes `*executors`.

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_composite_executor.py`:

```python
from datetime import datetime, UTC

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.external.composite_executor import CompositeToolExecutor


class _Stub:
    def __init__(self, schemas, results):
        self._schemas = schemas
        self._results = results
    def list_schemas(self):
        return self._schemas
    def handles(self, name):
        return name in self._results
    def execute(self, name, args, turn):
        return self._results[name]


def _turn() -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1", started_at=datetime(2026, 6, 18, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text="x", speaker_id=None, metadata={},
    )


def test_list_schemas_concatenates_in_order():
    a = _Stub([{"name": "a1"}, {"name": "a2"}], {})
    b = _Stub([{"name": "b1"}], {})
    c = CompositeToolExecutor(a, b)
    assert [s["name"] for s in c.list_schemas()] == ["a1", "a2", "b1"]


def test_execute_routes_to_first_executor_that_handles():
    a = _Stub([], {"shared": "from_a"})
    b = _Stub([], {"shared": "from_b", "only_b": "from_b_only"})
    c = CompositeToolExecutor(a, b)
    assert c.execute("shared", {}, _turn()) == "from_a"
    assert c.execute("only_b", {}, _turn()) == "from_b_only"


def test_execute_returns_unknown_when_no_executor_handles():
    a = _Stub([], {})
    b = _Stub([], {})
    c = CompositeToolExecutor(a, b)
    out = c.execute("phantom", {}, _turn())
    assert "phantom" in out
    assert "unknown" in out.lower() or "not found" in out.lower()
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_composite_executor.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `server/external/composite_executor.py`:

```python
"""CompositeToolExecutor — Conversation's single ToolExecutorPort.
Routes tool execution by name to the first executor that claims it; lists
the union of every executor's schemas in registration order."""
from __future__ import annotations
from typing import Protocol

from server.cognition.contracts import VoiceTurn


class _ChildExecutor(Protocol):
    def list_schemas(self) -> list[dict]: ...
    def handles(self, name: str) -> bool: ...
    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str: ...


class CompositeToolExecutor:
    def __init__(self, *executors: _ChildExecutor):
        self._executors = list(executors)

    def list_schemas(self) -> list[dict]:
        out: list[dict] = []
        for ex in self._executors:
            out.extend(ex.list_schemas())
        return out

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        for ex in self._executors:
            if ex.handles(name):
                return ex.execute(name, args, turn)
        return f"Unknown tool: {name}"
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/external/test_composite_executor.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/external/composite_executor.py tests/external/test_composite_executor.py
git commit -m "external: CompositeToolExecutor — single ToolExecutorPort over many adapters"
```

---

### Task 8: Wire composition root + retire HARestToolExecutor

**Files:**
- Modify: `server/main.py` (build_app wiring)
- Delete: `server/external/ha_rest_adapter.py`
- Modify: `tools/boundary_check.py` (if it referenced ha_rest_adapter)
- Modify: `tests/test_main_composition.py` (ensure composition still builds)

**Interfaces:**
- Consumes: `HAMCPToolExecutor`, `build_native_registry`, `CompositeToolExecutor`, `AsyncRunner`, `OpenMeteoWeather`.
- Produces: a `Conversation` whose `tools` is a `CompositeToolExecutor` of native + HA-MCP.

- [ ] **Step 1: Update `server/main.py`**

Replace the existing tool-executor line. New imports + composition:

```python
# add at top of file
from server.external._internal.async_runner import AsyncRunner
from server.external.composite_executor import CompositeToolExecutor
from server.external.ha_mcp_executor import HAMCPToolExecutor
from server.external.igor_native_tools import build_native_registry
from server.external.weather_open_meteo import OpenMeteoWeather

# remove the import of HARestToolExecutor
# (and the line `tools = HARestToolExecutor()` further down)
```

In `build()`, replace `tools = HARestToolExecutor()` with:

```python
    ha_url = os.environ.get("HA_URL", "http://10.0.40.5:8123").rstrip("/")
    ha_token = os.environ.get("HA_TOKEN", "")
    mcp_url = f"{ha_url}/api/mcp"
    default_location = os.environ.get("DEFAULT_LOCATION", "Arlington, VA")

    async_runner = AsyncRunner()
    weather = OpenMeteoWeather()
    native_tools = build_native_registry(
        memory=memory, user_state=user_state,
        weather=weather, default_location=default_location,
    )
    ha_tools = HAMCPToolExecutor(mcp_url, ha_token, async_runner)
    tools = CompositeToolExecutor(native_tools, ha_tools)
```

(Note: `native_tools` is registered *first* so a future name collision falls in Igor's favor.)

- [ ] **Step 2: Delete legacy adapter**

```bash
git rm server/external/ha_rest_adapter.py
```

- [ ] **Step 3: Run the composition test**

```bash
.venv/Scripts/python.exe -m pytest tests/test_main_composition.py -v
```
Expected: PASS. The test stubs HA_TOKEN — `HAMCPToolExecutor.refresh()` will catch the connection failure and log a warning, then return an empty catalog. App still builds.

If it fails because `import HARestToolExecutor` lingers, search and remove:
```bash
.venv/Scripts/python.exe -m pytest tests/test_main_composition.py -v 2>&1 | tail -20
```

- [ ] **Step 4: Full test suite**

```bash
.venv/Scripts/python.exe -m pytest
```
Expected: all pass.

- [ ] **Step 5: Boundary check**

```bash
.venv/Scripts/python.exe -m tools.boundary_check
```
Expected: `Boundary check passed.`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "main: wire MCP+native tool registry; retire HARestToolExecutor"
```

---

### Task 9: HA-side setup notes + smoke-test playbook

**Files:**
- Create: `docs/ha-mcp-setup.md` (short — read in 2 min)

**Interfaces:**
- Produces: documentation only.

- [ ] **Step 1: Write the doc**

Create `docs/ha-mcp-setup.md`:

```markdown
# Home Assistant MCP Server — One-Time Setup

Igor reaches HA via HA's built-in MCP Server integration. Set this up ONCE
on the HA box, then never touch it.

## 1. Install the integration

HA UI → Settings → Devices & Services → Add Integration → search "Model
Context Protocol Server" → install.

When prompted for auth, choose **Long-lived access token** and paste the
existing `HA_TOKEN` value (the one already in Portainer's Igor stack env).

The endpoint that Igor connects to is automatically:
`http://10.0.40.5:8123/api/mcp`

## 2. Expose the entities Igor should control

Igor sees only entities that HA flags as exposed-to-voice. This is HA's
expose toggle, NOT Igor's allowlist.

HA UI → Settings → Voice assistants → Expose. Toggle on every light,
switch, media_player, climate, todo, etc. that Igor should be able to act on.

## 3. Verify Igor can see them

After redeploying Igor's container, on the Pi:

    docker logs igor | grep "HA MCP catalog"

Expected: a line like `HA MCP catalog: 17 tool(s) cached`. If it shows `0`,
HA's MCP integration isn't reachable — check the endpoint URL and the
exposure list.

## 4. Smoke tests

Say:
- "Okay Nabu, turn off the kitchen lights"
- "Okay Nabu, what's the weather?"  (Igor-native, doesn't need HA)
- "Okay Nabu, remember I prefer dark roast coffee"  (Igor-native)

Then on the Pi:

    docker exec igor python -m server.tools.recent_episodes 5

Should show all three with `tools: HassTurnOff` / `get_weather` / `save_memory`
in the tool-calls line.
```

- [ ] **Step 2: Commit**

```bash
git add docs/ha-mcp-setup.md
git commit -m "docs: HA MCP Server one-time setup + smoke test playbook"
```

- [ ] **Step 3: Push the whole feature**

```bash
git push
```

Expected: Portainer GitOps pulls and redeploys.

---

## Self-Review

**Spec coverage:**
- ✅ Delegate HA actions to MCP — Task 4 (HAMCPToolExecutor) + Task 8 (wiring).
- ✅ Igor-native cognition tools preserved — Task 5 (weather) + Task 6 (memory/feedback/calc).
- ✅ Boundary check enforces `mcp` lock to external/ — Task 1.
- ✅ Sync ToolExecutorPort interface preserved — Task 2 (AsyncRunner), Task 4.
- ✅ No per-device code remains — Task 8 deletes `ha_rest_adapter.py`.
- ✅ HA-side setup documented — Task 9.

**Placeholder scan:** None.

**Type consistency:** `IgorTool` and `McpTool` field names match across tasks. `list_schemas() -> list[dict]`, `execute(name, args, turn) -> str`, and `handles(name) -> bool` are the same triple across `HAMCPToolExecutor`, `IgorNativeToolExecutor`, and `CompositeToolExecutor`. `AsyncRunner.run(coro)` and `shutdown()` are consistent in Tasks 2, 4, 8.
