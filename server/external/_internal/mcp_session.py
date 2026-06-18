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
