from datetime import datetime, UTC, timedelta
from server.cognition.aggregates.user_state import UserState
from server.external.sqlite_persistence import SqlitePersistence


def test_feedback_lifecycle(tmp_path):
    us = UserState(SqlitePersistence(tmp_path / "brain.db"))
    fb = us.log_feedback(issue="use 24h time",
                         at=datetime(2026, 1, 1, tzinfo=UTC),
                         source_episode_id=None)
    open_items = us.list_open_feedback()
    assert len(open_items) == 1
    us.resolve_feedback(fb.feedback_id)
    assert len(us.list_open_feedback()) == 0


def test_reminder_lifecycle(tmp_path):
    us = UserState(SqlitePersistence(tmp_path / "brain.db"))
    fire = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5)
    rm = us.add_reminder(name="pasta", fire_at=fire, room_id="kitchen",
                         source_episode_id=None)
    assert any(r.reminder_id == rm.reminder_id for r in us.list_pending())
    us.fire_reminder(rm.reminder_id)
    assert all(r.reminder_id != rm.reminder_id for r in us.list_pending())
