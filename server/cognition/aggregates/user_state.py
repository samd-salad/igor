"""UserState — feedback + reminders. Merged aggregate per spec validation."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from server.cognition.contracts import FeedbackEntry, Reminder
from server.cognition.ports.persistence import PersistencePort


class UserState:
    def __init__(self, persistence: PersistencePort):
        self._p = persistence

    # ---- Feedback ----
    def log_feedback(self, issue: str, at: datetime,
                     source_episode_id: Optional[str]) -> FeedbackEntry:
        fb = FeedbackEntry(
            feedback_id=str(uuid.uuid4()), occurred_at=at,
            issue=issue, status="open", source_episode_id=source_episode_id,
        )
        self._p.save_feedback(fb)
        return fb

    def list_open_feedback(self) -> list[FeedbackEntry]:
        return self._p.list_feedback("open")

    def resolve_feedback(self, feedback_id: str) -> None:
        self._p.resolve_feedback(feedback_id)

    # ---- Reminders ----
    def add_reminder(self, name: str, fire_at: datetime, room_id: Optional[str],
                     source_episode_id: Optional[str]) -> Reminder:
        rm = Reminder(
            reminder_id=str(uuid.uuid4()), name=name, fire_at=fire_at,
            room_id=room_id, status="pending", source_episode_id=source_episode_id,
        )
        self._p.save_reminder(rm)
        return rm

    def list_pending(self) -> list[Reminder]:
        return self._p.list_pending_reminders()

    def fire_reminder(self, reminder_id: str) -> None:
        self._p.update_reminder_status(reminder_id, "fired")

    def cancel_reminder(self, reminder_id: str) -> None:
        self._p.update_reminder_status(reminder_id, "cancelled")
