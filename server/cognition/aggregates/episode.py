"""EpisodeStore — owns Episodes. Each Episode is also the provenance anchor."""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from server.cognition.contracts import Episode
from server.cognition.ports.persistence import PersistencePort


class EpisodeStore:
    def __init__(self, persistence: PersistencePort):
        self._p = persistence

    def add(self, episode: Episode) -> None:
        self._p.save_episode(episode)

    def load(self, episode_id: str) -> Optional[Episode]:
        return self._p.load_episode(episode_id)

    def get_recent(self, n: int) -> list[Episode]:
        return self._p.list_recent_episodes(n)

    def get_unconsolidated(self) -> list[Episode]:
        return self._p.list_unconsolidated_episodes()

    def mark_consolidated(self, episode_ids: list[str], at: datetime) -> None:
        self._p.mark_episodes_consolidated(episode_ids, at)
