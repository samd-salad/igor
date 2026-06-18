"""Initial RetrievalPort impl: tag overlap + recency decay.
Swap for HybridRetrieval (semantic + tag + recency) when crossing ~150 conversations."""
from __future__ import annotations
import math
import re
from datetime import datetime, UTC

from server.cognition.contracts import VoiceTurn, Fact
from server.external.sqlite_persistence import SqlitePersistence


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


class TagRetrieval:
    def __init__(self, persistence: SqlitePersistence):
        self._p = persistence

    def query(self, turn: VoiceTurn, k: int = 10) -> list[Fact]:
        qtokens = _tokens(turn.input_text)
        if not qtokens:
            return []
        now = datetime.now(UTC)
        scored: list[tuple[float, Fact]] = []
        for f in self._p.list_active_facts():
            tag_overlap = len(qtokens & _tokens(" ".join(f.tags)))
            value_overlap = len(qtokens & _tokens(f.value))
            key_overlap = len(qtokens & _tokens(f.key))
            recency = math.exp(-((now - f.created_at).days) / 30.0)
            score = 1.0 * tag_overlap + 0.5 * key_overlap + 0.3 * value_overlap + 0.2 * recency
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:k]]
