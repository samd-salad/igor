"""ToolRegistry — exposes ToolExecutorPort.list_schemas() as cognition-shaped data."""
from __future__ import annotations
from server.cognition.contracts import ToolSchema
from server.cognition.ports.tools import ToolExecutorPort


class ToolRegistry:
    def __init__(self, executor: ToolExecutorPort):
        self._exec = executor

    @property
    def schemas(self) -> list[ToolSchema]:
        return self._exec.list_schemas()
