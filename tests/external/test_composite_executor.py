from datetime import datetime, UTC

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.external.composite_executor import CompositeToolExecutor


class _Stub:
    def __init__(self, schemas, results):
        self._schemas = schemas
        self._results = results
    def list_schemas(self):
        return self._schemas
    def handles(self, name):
        return name in self._results
    def execute(self, name, args, turn):
        return self._results[name]


def _turn() -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1", started_at=datetime(2026, 6, 18, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text="x", speaker_id=None, metadata={},
    )


def test_list_schemas_concatenates_in_order():
    a = _Stub([{"name": "a1"}, {"name": "a2"}], {})
    b = _Stub([{"name": "b1"}], {})
    c = CompositeToolExecutor(a, b)
    assert [s["name"] for s in c.list_schemas()] == ["a1", "a2", "b1"]


def test_execute_routes_to_first_executor_that_handles():
    a = _Stub([], {"shared": "from_a"})
    b = _Stub([], {"shared": "from_b", "only_b": "from_b_only"})
    c = CompositeToolExecutor(a, b)
    assert c.execute("shared", {}, _turn()) == "from_a"
    assert c.execute("only_b", {}, _turn()) == "from_b_only"


def test_execute_returns_unknown_when_no_executor_handles():
    a = _Stub([], {})
    b = _Stub([], {})
    c = CompositeToolExecutor(a, b)
    out = c.execute("phantom", {}, _turn())
    assert "phantom" in out
    assert "unknown" in out.lower() or "not found" in out.lower()
