from server.cognition.contracts import ToolSchema
from server.external.claude_adapter import ClaudeAdapter


class _FakeAnthropic:
    def __init__(self):
        self.last_kwargs: dict = {}
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, outer):
            self._outer = outer
        def create(self, **kwargs):
            self._outer.last_kwargs = kwargs
            class _R:
                content = [type("X", (), {"type": "text", "text": "Hello, world."})]
                usage = type("U", (), {"input_tokens": 12, "output_tokens": 5})
                stop_reason = "end_turn"
            return _R()


def test_chat_returns_text_and_tokens():
    adapter = ClaudeAdapter(client=_FakeAnthropic(), model="claude-haiku-4-5-20251001")
    result = adapter.chat(
        system_prompt="be brief",
        user_text="hello",
        tool_schemas=[],
        tool_executor=lambda name, args: "",
    )
    assert result.text == "Hello, world."
    assert result.input_tokens == 12
    assert result.output_tokens == 5


def test_chat_translates_tool_schemas_to_anthropic_dict_shape():
    """ClaudeAdapter is the ONLY place that knows Anthropic's tool shape.
    cognition hands it ToolSchema value objects; the adapter translates."""
    fake = _FakeAnthropic()
    adapter = ClaudeAdapter(client=fake, model="claude-haiku-4-5-20251001")
    schemas = [
        ToolSchema(name="ping", description="say pong",
                   input_schema={"type": "object", "properties": {}}),
        ToolSchema(name="add", description="add two numbers",
                   input_schema={"type": "object",
                                 "properties": {"a": {"type": "number"}}}),
    ]
    adapter.chat(
        system_prompt="x", user_text="x",
        tool_schemas=schemas,
        tool_executor=lambda name, args: "",
    )
    sent = fake.last_kwargs["tools"]
    assert sent == [
        {"name": "ping", "description": "say pong",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "add", "description": "add two numbers",
         "input_schema": {"type": "object",
                          "properties": {"a": {"type": "number"}}}},
    ]
