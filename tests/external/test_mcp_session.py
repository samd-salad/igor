from contextlib import asynccontextmanager
from unittest.mock import patch

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


def test_fetch_tool_catalog_uses_http_client_kwarg_not_headers(monkeypatch):
    """Regression: in mcp 1.x the new streamable_http_client takes a
    pre-configured httpx.AsyncClient, not a headers= kwarg. Earlier code
    passed headers= and crashed at runtime ("unexpected keyword argument
    'headers'"). Pin the call shape so we don't regress."""
    import asyncio
    from server.external._internal import mcp_session

    captured: dict = {}

    @asynccontextmanager
    async def fake_streamable(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        class _RW:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
        yield (_RW(), _RW(), lambda: None)

    class _FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def initialize(self): pass
        async def list_tools(self):
            class R:
                tools = []
            return R()

    monkeypatch.setattr(mcp_session, "streamable_http_client", fake_streamable)
    monkeypatch.setattr(mcp_session, "ClientSession", _FakeSession)

    asyncio.run(mcp_session.fetch_tool_catalog("http://x/api/mcp", "tok"))

    assert "http_client" in captured["kwargs"], \
        "must pass http_client= (mcp 1.x), not headers="
    assert "headers" not in captured["kwargs"]
