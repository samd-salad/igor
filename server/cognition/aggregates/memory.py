"""MemoryStore aggregate — owns Facts. Updates create a new fact and invalidate the old."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from server.cognition.contracts import Fact
from server.cognition.ports.persistence import PersistencePort


class MemoryStore:
    def __init__(self, persistence: PersistencePort):
        self._p = persistence

    def save_fact(
        self, category: str, key: str, value: str,
        tags: list[str], source_episode_id: Optional[str],
        now: datetime,
    ) -> Fact:
        fact = Fact(
            fact_id=str(uuid.uuid4()), category=category, key=key, value=value,
            tags=tags, source_episode_id=source_episode_id,
            embedding=None,
            valid_at=now, invalid_at=None, created_at=now,
        )
        self._p.save_fact(fact)
        return fact

    def update_fact(
        self, category: str, key: str, new_value: str,
        tags: list[str], source_episode_id: Optional[str],
        now: datetime,
    ) -> Fact:
        existing = self._p.find_fact(category, key)
        if existing is not None:
            self._p.invalidate_fact(existing.fact_id, now)
        return self.save_fact(category, key, new_value, tags, source_episode_id, now)

    def find_fact(self, category: str, key: str) -> Optional[Fact]:
        return self._p.find_fact(category, key)

    def forget_fact(self, category: str, key: str, now: datetime) -> bool:
        """Invalidate the active fact at (category, key). Returns whether
        anything was forgotten (False if no such active fact)."""
        existing = self._p.find_fact(category, key)
        if existing is None:
            return False
        self._p.invalidate_fact(existing.fact_id, now)
        return True

    def list_active(self) -> list[Fact]:
        return self._p.list_active_facts()
