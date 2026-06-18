from server.external.claude_adapter import ClaudeAdapter


class _FakeAnthropic:
    class _Messages:
        def create(self, **kwargs):
            class _R:
                content = [type("X", (), {"type": "text", "text": "Hello, world."})]
                usage = type("U", (), {"input_tokens": 12, "output_tokens": 5})
                stop_reason = "end_turn"
            return _R()

    messages = _Messages()


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
