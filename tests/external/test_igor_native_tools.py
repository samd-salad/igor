from datetime import datetime, UTC
from unittest.mock import MagicMock

from server.cognition.contracts import RoomConfig, ToolSchema, VoiceTurn
from server.external.igor_native_tools import (
    IgorTool, IgorNativeToolExecutor, build_native_registry,
)


def _turn(text: str = "x") -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1",
        started_at=datetime(2026, 6, 18, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_executor_lists_registered_tool_schemas():
    tool = IgorTool(
        name="ping",
        description="say pong",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, turn: "pong",
    )
    ex = IgorNativeToolExecutor([tool])
    schemas = ex.list_schemas()
    assert schemas == [ToolSchema(
        name="ping",
        description="say pong",
        input_schema={"type": "object", "properties": {}},
    )]
    assert ex.handles("ping")
    assert not ex.handles("unknown")


def test_executor_routes_execute_to_handler_with_args_and_turn():
    captured = {}
    def handler(args, turn):
        captured["args"] = args
        captured["turn"] = turn
        return "ok"
    tool = IgorTool("echo", "echo", {"type": "object", "properties": {}}, handler)
    ex = IgorNativeToolExecutor([tool])
    out = ex.execute("echo", {"hello": "world"}, _turn("hi"))
    assert out == "ok"
    assert captured["args"] == {"hello": "world"}
    assert captured["turn"].input_text == "hi"


def test_executor_returns_message_for_unknown_tool():
    ex = IgorNativeToolExecutor([])
    out = ex.execute("nope", {}, _turn())
    assert "unknown" in out.lower() or "not found" in out.lower()


def test_build_native_registry_includes_expected_tools():
    memory = MagicMock()
    user_state = MagicMock()
    weather = MagicMock()
    weather.current.return_value = "65°F and clear in Arlington."
    ex = build_native_registry(memory=memory, user_state=user_state,
                               weather=weather, default_location="Arlington, VA")
    names = {s.name for s in ex.list_schemas()}
    assert {"save_memory", "forget_memory", "log_feedback",
            "get_weather", "calculate"} <= names


def test_save_memory_calls_memory_store_with_correct_args():
    memory = MagicMock()
    ex = build_native_registry(memory=memory, user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("save_memory", {
        "category": "preferences", "key": "coffee", "value": "dark roast",
        "tags": ["beverage"],
    }, _turn())
    memory.save_fact.assert_called_once()
    kwargs = memory.save_fact.call_args.kwargs
    args = memory.save_fact.call_args.args
    # Accept either positional or keyword binding
    if kwargs:
        assert kwargs["category"] == "preferences"
        assert kwargs["key"] == "coffee"
        assert kwargs["value"] == "dark roast"
        assert kwargs["tags"] == ["beverage"]
    else:
        assert args[0] == "preferences"
        assert args[1] == "coffee"
        assert args[2] == "dark roast"
    assert "preferences" in out
    assert "coffee" in out


def test_forget_memory_invokes_forget_and_reports_outcome():
    memory = MagicMock()
    memory.forget_fact.return_value = True
    ex = build_native_registry(memory=memory, user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("forget_memory", {"category": "preferences", "key": "coffee"},
                     _turn())
    memory.forget_fact.assert_called_once()
    assert "coffee" in out.lower() or "forgot" in out.lower()


def test_forget_memory_returns_no_op_message_when_nothing_to_forget():
    memory = MagicMock()
    memory.forget_fact.return_value = False
    ex = build_native_registry(memory=memory, user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("forget_memory", {"category": "preferences", "key": "coffee"},
                     _turn())
    assert "no" in out.lower() or "nothing" in out.lower() or "not" in out.lower()


def test_log_feedback_calls_user_state():
    user_state = MagicMock()
    ex = build_native_registry(memory=MagicMock(), user_state=user_state,
                               weather=MagicMock(), default_location="x")
    out = ex.execute("log_feedback", {"issue": "stop saying 'sure'"}, _turn())
    user_state.log_feedback.assert_called_once()
    assert out


def test_get_weather_uses_default_when_no_location_argument():
    weather = MagicMock()
    weather.current.return_value = "70°F and sunny in Arlington."
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=weather, default_location="Arlington, VA")
    out = ex.execute("get_weather", {}, _turn())
    weather.current.assert_called_once_with("Arlington, VA")
    assert "Arlington" in out


def test_get_weather_uses_argument_when_provided():
    weather = MagicMock()
    weather.current.return_value = "x"
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=weather, default_location="Arlington, VA")
    ex.execute("get_weather", {"location": "Tokyo"}, _turn())
    weather.current.assert_called_once_with("Tokyo")


def test_calculate_evaluates_simple_arithmetic():
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("calculate", {"expression": "15 * 0.18"}, _turn())
    assert "2.7" in out


def test_calculate_rejects_dangerous_expressions():
    ex = build_native_registry(memory=MagicMock(), user_state=MagicMock(),
                               weather=MagicMock(), default_location="x")
    out = ex.execute("calculate", {"expression": "__import__('os').system('echo bad')"},
                     _turn())
    assert "can't" in out.lower() or "invalid" in out.lower() or "error" in out.lower()
