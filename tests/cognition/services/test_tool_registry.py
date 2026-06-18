from server.cognition.services.tool_registry import ToolRegistry


class _FakeExecutor:
    def list_schemas(self):
        return [{"name": "fake_say", "description": "echo", "input_schema": {}}]

    def execute(self, name, args, turn):
        return ""


def test_schemas_pass_through():
    reg = ToolRegistry(_FakeExecutor())
    assert reg.schemas == [{"name": "fake_say", "description": "echo", "input_schema": {}}]
