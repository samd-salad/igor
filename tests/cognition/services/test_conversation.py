from datetime import datetime, timedelta, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.ports.llm import ChatResult
from server.cognition.services.conversation import Conversation
from server.cognition.services.intent_router import IntentRouter, Tier1Match
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval
from server.external.system_clock import SystemClock


class _StubLLM:
    def __init__(self, text: str = "hi back"):
        self._text = text
        self.calls = 0

    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        self.calls += 1
        return ChatResult(text=self._text, commands_executed=[],
                          input_tokens=10, output_tokens=2)


class _StubExecutor:
    def list_schemas(self):
        return []
    def execute(self, name, args, turn):
        return ""


class _FixedClock:
    def __init__(self, t: datetime):
        self._t = t
    def now(self) -> datetime:
        return self._t


def _turn(text: str, correlation_id: str = "t-1",
          at: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> VoiceTurn:
    return VoiceTurn(
        correlation_id=correlation_id, started_at=at,
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def _build_conversation(sp, llm=None, clock=None, tools=None, intent_router=None):
    return Conversation(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), user_state=UserState(sp),
        retrieval=TagRetrieval(sp), llm=llm or _StubLLM(),
        tools=tools or _StubExecutor(), clock=clock or SystemClock(),
        intent_router=intent_router,
    )


def test_conversation_writes_episode_with_correlation_id(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = _build_conversation(sp)
    result = conv.process(_turn("hello there", correlation_id="t-hello"))
    assert result.correlation_id == "t-hello"
    assert result.response_text == "hi back"
    ep = sp.load_episode("t-hello")
    assert ep is not None
    assert ep.raw_utterance == "hello there"


def test_pause_now_falls_through_to_llm(tmp_path):
    """Regression: Tier 1 used to short-circuit 'pause' to a non-existent
    `play_pause` tool, returning a canned 'Paused.' while nothing happened.
    Patterns are gone; 'pause' must reach the LLM so it can pick the real
    HA media tool."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="paused for real this time")
    conv = _build_conversation(sp, llm=llm)
    result = conv.process(_turn("pause", correlation_id="t-pause"))
    assert llm.calls == 1
    assert result.response_text == "paused for real this time"


def test_conversation_persists_response_text_on_llm_turn(tmp_path):
    """Igor's literal response must be queryable from brain.db after the turn —
    otherwise the user has no way to audit what he said when something goes weird."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = _build_conversation(sp, llm=_StubLLM(text="something witty"))
    conv.process(_turn("good morning igor", correlation_id="t-morn"))
    ep = sp.load_episode("t-morn")
    assert ep is not None
    assert ep.response_text == "something witty"


def test_tier1_dispatch_failure_falls_through_to_llm(tmp_path):
    """If a Tier 1 pattern ever matches a tool no executor handles, we must
    NOT serve the canned tier1 response (that's the bug that hid the
    HA-MCP migration mismatch for months). Detect the executor's
    'Unknown tool:' / 'Error ' return, log it, and hand the turn to the LLM."""

    class _AlwaysMatchRouter:
        def route(self, turn):
            return Tier1Match("frob", {}, "Frobbed.")

    class _FailingExecutor:
        def list_schemas(self):
            return []
        def execute(self, name, args, turn):
            return f"Unknown tool: {name}"

    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="i don't actually know how to frob")
    conv = _build_conversation(
        sp, llm=llm, tools=_FailingExecutor(),
        intent_router=_AlwaysMatchRouter(),
    )
    result = conv.process(_turn("frobnicate the widget", correlation_id="t-frob"))

    assert llm.calls == 1, "LLM must run when Tier 1 dispatch fails"
    assert result.response_text == "i don't actually know how to frob"
    assert result.response_text != "Frobbed.", "must not serve canned tier1 response on dispatch failure"


def test_conversation_suppresses_self_echo_within_window(tmp_path):
    """When the transcript arriving moments after Igor speaks overlaps
    heavily with his last response, treat it as the mic catching his own
    voice. Stay silent — don't run the LLM, don't TTS back."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    igor_response = "Sorry, Igor is unreachable right now"
    llm = _StubLLM(text=igor_response)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    conv = _build_conversation(sp, llm=llm, clock=_FixedClock(t0))

    first = conv.process(_turn("what time is it", correlation_id="t-1", at=t0))
    assert first.silent is False
    assert first.response_text == igor_response
    assert llm.calls == 1

    # Mic picks up Igor's own response ~2s later
    echo = conv.process(_turn(
        "sorry igor is unreachable right now",
        correlation_id="t-echo",
        at=t0 + timedelta(seconds=2),
    ))
    assert echo.silent is True
    assert echo.response_text == ""
    assert llm.calls == 1, "LLM should NOT be called when echo detected"


def test_conversation_does_not_suppress_after_echo_window(tmp_path):
    """A real user reply that happens to share words with Igor's last
    response, but arrives after the echo window, must NOT be suppressed."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="the kitchen lights are dim and warm tonight")
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    conv = _build_conversation(sp, llm=llm, clock=_FixedClock(t0))

    conv.process(_turn("how are the kitchen lights doing", correlation_id="t-a", at=t0))
    later = conv.process(_turn(
        "the kitchen lights are too dim and warm tonight",
        correlation_id="t-b",
        at=t0 + timedelta(seconds=30),
    ))
    assert later.silent is False
    assert llm.calls == 2


def test_conversation_does_not_suppress_unrelated_input(tmp_path):
    """A reply within the echo window but with low token overlap is a real
    user turn, not echo — must go through normally."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="the kitchen lights are dim and warm tonight")
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    conv = _build_conversation(sp, llm=llm, clock=_FixedClock(t0))

    conv.process(_turn("how are the kitchen lights doing", correlation_id="t-a", at=t0))
    reply = conv.process(_turn(
        "what's the weather forecast",
        correlation_id="t-b",
        at=t0 + timedelta(seconds=3),
    ))
    assert reply.silent is False
    assert llm.calls == 2


def test_conversation_stays_silent_when_llm_returns_ambient_sentinel(tmp_path):
    """Claude returns '[silent]' when it judges the transcript as ambient
    room audio. Conversation must mark the result silent and store an
    ambient_silent episode — no echo, no TTS, no follow-up cascade."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="[silent]")
    conv = _build_conversation(sp, llm=llm)

    result = conv.process(_turn("they don't rouse me sexually",
                                correlation_id="t-amb"))
    assert result.silent is True
    assert result.response_text == ""
    ep = sp.load_episode("t-amb")
    assert ep is not None
    assert ep.intent == "ambient_silent"


def test_conversation_silent_sentinel_matcher_is_tolerant_of_whitespace(tmp_path):
    """Whitespace and TTS-style trailing period must still trigger silence —
    Claude won't always return the exact 7-character string."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    for variant in ("[silent]", "  [silent]  ", "[silent].", "[Silent]"):
        llm = _StubLLM(text=variant)
        conv = _build_conversation(sp, llm=llm)
        result = conv.process(_turn("come on come on", correlation_id=f"t-{variant[:3]}"))
        assert result.silent is True, f"variant {variant!r} did not trigger silence"


def test_conversation_does_not_silence_real_responses_containing_silent_word(tmp_path):
    """The matcher must NOT trigger on natural responses that happen to
    contain the word 'silent' (e.g. 'I'll keep silent about that.')."""
    sp = SqlitePersistence(tmp_path / "brain.db")
    llm = _StubLLM(text="I'll keep silent about that.")
    conv = _build_conversation(sp, llm=llm)
    result = conv.process(_turn("don't mention dad's birthday", correlation_id="t-keep"))
    assert result.silent is False
    assert "silent" in result.response_text.lower()
