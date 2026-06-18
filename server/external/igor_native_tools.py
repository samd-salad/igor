"""Igor-native tool registry. These tools live in Igor (not HA) because they
operate on cognition aggregates or external services HA doesn't expose:
memory writes, feedback logging, weather, arithmetic.

Each tool takes (args: dict, turn: VoiceTurn) and returns a string suitable
for the LLM's tool_result.content."""
from __future__ import annotations
import ast
import logging
import operator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import VoiceTurn
from server.external.weather_open_meteo import OpenMeteoWeather

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IgorTool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, VoiceTurn], str]


class IgorNativeToolExecutor:
    def __init__(self, tools: list[IgorTool]):
        self._tools = list(tools)
        self._by_name = {t.name: t for t in self._tools}

    def handles(self, name: str) -> bool:
        return name in self._by_name

    def list_schemas(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema}
            for t in self._tools
        ]

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        tool = self._by_name.get(name)
        if tool is None:
            return f"Unknown native tool: {name}"
        try:
            return tool.handler(args, turn)
        except Exception as e:
            logger.exception("Native tool %s failed", name)
            return f"Error in {name}: {e}"


def _save_memory(memory: MemoryStore) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        category = args["category"]
        key = args["key"]
        value = args["value"]
        tags = list(args.get("tags") or [])
        memory.save_fact(
            category=category, key=key, value=value, tags=tags,
            source_episode_id=turn.correlation_id, now=datetime.now(UTC),
        )
        return f"Remembered {category}/{key}."
    return IgorTool(
        name="save_memory",
        description="Save a fact about the user. Use when learning names, "
                    "preferences, schedules, relationships, or corrections.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "description": "preferences | schedule | people | personal | home | other"},
                "key": {"type": "string",
                        "description": "short lowercase underscored identifier"},
                "value": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["category", "key", "value"],
        },
        handler=handler,
    )


def _forget_memory(memory: MemoryStore) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        category = args["category"]
        key = args["key"]
        forgotten = memory.forget_fact(category, key, datetime.now(UTC))
        if not forgotten:
            return f"Nothing to forget at {category}/{key}."
        return f"Forgot {category}/{key}."
    return IgorTool(
        name="forget_memory",
        description="Forget a previously-saved fact about the user.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "key": {"type": "string"},
            },
            "required": ["category", "key"],
        },
        handler=handler,
    )


def _log_feedback(user_state: UserState) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        issue = args["issue"]
        user_state.log_feedback(
            issue=issue, at=datetime.now(UTC),
            source_episode_id=turn.correlation_id,
        )
        return "Logged."
    return IgorTool(
        name="log_feedback",
        description="Log a change request or correction from the user. "
                    "Use when the user says something went wrong or asks "
                    "for a behavioral change.",
        input_schema={
            "type": "object",
            "properties": {
                "issue": {"type": "string"},
            },
            "required": ["issue"],
        },
        handler=handler,
    )


def _get_weather(weather: OpenMeteoWeather, default_location: str) -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        loc = args.get("location") or default_location
        return weather.current(loc)
    return IgorTool(
        name="get_weather",
        description="Get current weather conditions for a location. "
                    "If the user doesn't specify a location, the default is used.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": [],
        },
        handler=handler,
    )


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric literal")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _calculate() -> IgorTool:
    def handler(args: dict, turn: VoiceTurn) -> str:
        expression = args.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval_arith(tree)
        except Exception:
            return "I can't evaluate that expression."
        formatted = f"{result:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return IgorTool(
        name="calculate",
        description="Evaluate a simple arithmetic expression (+, -, *, /, %, **).",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
        handler=handler,
    )


def build_native_registry(memory: MemoryStore, user_state: UserState,
                          weather: OpenMeteoWeather,
                          default_location: str) -> IgorNativeToolExecutor:
    return IgorNativeToolExecutor([
        _save_memory(memory),
        _forget_memory(memory),
        _log_feedback(user_state),
        _get_weather(weather, default_location),
        _calculate(),
    ])
