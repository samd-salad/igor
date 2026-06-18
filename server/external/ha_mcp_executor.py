"""ToolExecutorPort impl that delegates to Home Assistant's MCP Server.

Caches HA's tool catalog at startup, translates each MCP tool's inputSchema
into Anthropic's tool schema shape, and forwards tool_use -> tools/call ->
tool_result. No per-device code — every new HA integration the user installs
becomes a callable tool here automatically (subject to HA's expose-to-voice)."""
from __future__ import annotations
import logging

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
