"""RetrievalPort — return relevant facts given a query."""
from __future__ import annotations
from typing import Protocol
from server.cognition.contracts import VoiceTurn, Fact


class RetrievalPort(Protocol):
    def query(self, turn: VoiceTurn, k: int = 10) -> list[Fact]: ...
