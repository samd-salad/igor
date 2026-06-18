from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import Episode
from server.cognition.ports.llm import ChatResult
from server.cognition.services.consolidator import Consolidator
from server.external.sqlite_persistence import SqlitePersistence


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        return ChatResult(text="Sam is a homelab nerd who likes dark roast coffee.",
                          commands_executed=[], input_tokens=10, output_tokens=5)


class _StubClock:
    def now(self):
        return datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


def _seed_unconsolidated(sp, n=5):
    for i in range(n):
        sp.save_episode(Episode(
            episode_id=f"ep-{i}",
            occurred_at=datetime(2026, 1, 1, 10, i, tzinfo=UTC),
            speaker_id=None, participants=[], intent="llm",
            raw_utterance=f"turn {i}", tool_calls=[], emotional_tone=None,
            summary=None, consolidated_at=None,
        ))


def _build(sp, n: int = 5) -> Consolidator:
    return Consolidator(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), llm=_StubLLM(), clock=_StubClock(),
        episodes_per_run=n,
    )


def test_consolidate_now_regenerates_identity_and_marks_consolidated(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_unconsolidated(sp, 5)
    _build(sp, n=5).run_once()
    assert IdentityStore(sp).get_narrative().startswith("Sam is a homelab nerd")
    assert len(EpisodeStore(sp).get_unconsolidated()) == 0


def test_replay_on_startup_when_unconsolidated_exist(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_unconsolidated(sp, 6)
    _build(sp, n=5).replay_if_pending()
    assert len(EpisodeStore(sp).get_unconsolidated()) <= 1
