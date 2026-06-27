from server.cognition.contracts import ToolSchema
from server.cognition.services.tool_registry import ToolRegistry


class _FakeExecutor:
    def list_schemas(self):
        return [ToolSchema(name="fake_say", description="echo", input_schema={})]

    def execute(self, name, args, turn):
        return ""


def test_schemas_pass_through():
    reg = ToolRegistry(_FakeExecutor())
    assert reg.schemas == [ToolSchema(name="fake_say", description="echo", input_schema={})]
