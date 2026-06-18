from datetime import datetime, UTC
from server.cognition.contracts import Reflection, FeedbackEntry, Reminder
from server.external.sqlite_persistence import SqlitePersistence


def test_identity_round_trip(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    assert sp.get_identity_narrative() is None
    sp.save_identity_narrative("Sam is a homelab nerd.", datetime(2026, 1, 1, tzinfo=UTC), "ep-9")
    assert sp.get_identity_narrative() == "Sam is a homelab nerd."
    assert sp.get_last_consolidated_episode_id() == "ep-9"


def test_reflections_and_feedback_and_reminders(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sp.save_reflection(Reflection(reflection_id="r-1", occurred_at=now,
                                  note="long preambles annoy user", source_episode_id=None))
    assert len(sp.list_recent_reflections(5)) == 1

    sp.save_feedback(FeedbackEntry(feedback_id="fb-1", occurred_at=now,
                                   issue="use 24h time", status="open",
                                   source_episode_id=None))
    open_items = sp.list_feedback("open")
    assert len(open_items) == 1
    sp.resolve_feedback("fb-1")
    assert len(sp.list_feedback("open")) == 0

    sp.save_reminder(Reminder(reminder_id="rm-1", name="pasta", fire_at=now,
                              room_id="kitchen", status="pending", source_episode_id=None))
    pending = sp.list_pending_reminders()
    assert len(pending) == 1
    sp.update_reminder_status("rm-1", "fired")
    assert len(sp.list_pending_reminders()) == 0
