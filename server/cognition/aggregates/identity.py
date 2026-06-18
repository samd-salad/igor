"""IdentityStore — single-row narrative + reflections sub-collection."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from server.cognition.contracts import Reflection
from server.cognition.ports.persistence import PersistencePort


class IdentityStore:
    def __init__(self, persistence: PersistencePort):
        self._p = persistence

    def get_narrative(self) -> str:
        return self._p.get_identity_narrative() or ""

    def replace_narrative(self, narrative: str, last_consolidated_at: datetime,
                          last_consolidated_episode_id: Optional[str]) -> None:
        self._p.save_identity_narrative(narrative, last_consolidated_at,
                                        last_consolidated_episode_id)

    def get_last_consolidated_episode_id(self) -> Optional[str]:
        return self._p.get_last_consolidated_episode_id()

    def log_reflection(self, note: str, at: datetime,
                       source_episode_id: Optional[str]) -> Reflection:
        r = Reflection(
            reflection_id=str(uuid.uuid4()), occurred_at=at,
            note=note, source_episode_id=source_episode_id,
        )
        self._p.save_reflection(r)
        return r

    def list_recent_reflections(self, limit: int) -> list[Reflection]:
        return self._p.list_recent_reflections(limit)
