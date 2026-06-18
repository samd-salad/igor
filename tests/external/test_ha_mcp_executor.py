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
