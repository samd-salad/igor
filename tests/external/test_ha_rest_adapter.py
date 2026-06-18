from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.external.ha_rest_adapter import HARestToolExecutor


class _FakeCommand:
    name = "fake_say"
    description = "Echo whatever you pass"

    @property
    def parameters(self):
        return {"text": {"type": "string", "description": "what to echo"}}

    required_parameters = ["text"]

    def to_tool(self):
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": self.parameters,
                                 "required": self.required_parameters}}

    def execute(self, text: str):
        return f"echo: {text}"


def _turn() -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text="x", speaker_id=None, metadata={},
    )


def test_list_schemas_returns_registered_tools():
    executor = HARestToolExecutor(commands={"fake_say": _FakeCommand()})
    schemas = executor.list_schemas()
    assert any(s["name"] == "fake_say" for s in schemas)


def test_execute_invokes_command():
    executor = HARestToolExecutor(commands={"fake_say": _FakeCommand()})
    assert executor.execute("fake_say", {"text": "hi"}, _turn()) == "echo: hi"


def test_unknown_tool_returns_error_string():
    executor = HARestToolExecutor(commands={})
    assert "unknown" in executor.execute("nope", {}, _turn()).lower()
