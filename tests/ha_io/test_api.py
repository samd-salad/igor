from fastapi.testclient import TestClient
from server.cognition.contracts import ConversationResult, RoomConfig
from server.ha_io.api import build_app


class _StubConversation:
    def process(self, turn):
        return ConversationResult(
            correlation_id=turn.correlation_id,
            response_text=f"echo: {turn.input_text}",
            commands_executed=[], end_conversation=True,
        )


class _NoopHAClient:
    def area_of_device(self, device_id):
        return ""


def _client():
    app = build_app(
        conversation=_StubConversation(),
        ha_client=_NoopHAClient(),
        known_rooms={"default": RoomConfig("default", "Default")},
    )
    return TestClient(app)


def test_health_ok():
    r = _client().get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_conversation_endpoint_round_trip():
    r = _client().post("/conversation/process",
                       json={"text": "hi", "device_id": None, "language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "echo: hi"
    assert body["end_conversation"] is True


def test_token_enforced(monkeypatch):
    monkeypatch.setenv("IGOR_API_TOKEN", "secret")
    r = _client().post("/conversation/process",
                       json={"text": "hi"})
    assert r.status_code == 401
    r = _client().post("/conversation/process",
                       json={"text": "hi"},
                       headers={"X-Igor-Token": "secret"})
    assert r.status_code == 200
