"""Async helpers around the `mcp` SDK. Lives in _internal so the file-locked
third-party `mcp` import doesn't leak to siblings. Used by HAMCPToolExecutor
behind AsyncRunner."""
from __future__ import annotations
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


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


@asynccontextmanager
async def _session(url: str, token: str):
    """Open a ClientSession to `url` with Bearer auth carried by a dedicated
    httpx client. New SDK (1.x) wants headers on the httpx client, not on
    streamable_http_client itself.

    verify=False when talking to https:// — HA on the homelab uses a
    self-signed cert. If deploying off-net, pin the cert fingerprint instead."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    verify = not url.lower().startswith("https://")
    async with httpx.AsyncClient(headers=headers, verify=verify) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def fetch_tool_catalog(url: str, token: str) -> list[McpTool]:
    async with _session(url, token) as session:
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
    async with _session(url, token) as session:
        result = await session.call_tool(name=name, arguments=arguments)
        if result.isError:
            return f"Error from HA: {_content_to_text(result.content)}"
        return _content_to_text(result.content) or "(no output)"
