"""SQLite implementation of cognition.ports.PersistencePort."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from server.cognition.contracts import (
    Episode, Fact, Reflection, FeedbackEntry, Reminder, ToolCallRecord,
)
from server.external._internal.db import open_db
from server.external.embedding_encoder import EmbeddingEncoder
from server.external.vector_store import VectorStore


def _dt_to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _row_to_episode(row: sqlite3.Row) -> Episode:
    tcs = json.loads(row["tool_calls"]) if row["tool_calls"] else []
    keys = row.keys()
    return Episode(
        episode_id=row["episode_id"],
        occurred_at=_iso_to_dt(row["occurred_at"]),
        speaker_id=row["speaker_id"],
        participants=json.loads(row["participants"]) if row["participants"] else [],
        intent=row["intent"],
        raw_utterance=row["raw_utterance"],
        tool_calls=[ToolCallRecord(**tc) for tc in tcs],
        emotional_tone=row["emotional_tone"],
        summary=row["summary"],
        consolidated_at=_iso_to_dt(row["consolidated_at"]),
        response_text=row["response_text"] if "response_text" in keys else None,
    )


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        fact_id=row["fact_id"], category=row["category"], key=row["key"],
        value=row["value"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        source_episode_id=row["source_episode_id"],
        embedding=row["embedding"],
        valid_at=_iso_to_dt(row["valid_at"]),
        invalid_at=_iso_to_dt(row["invalid_at"]),
        created_at=_iso_to_dt(row["created_at"]),
    )


class SqlitePersistence:
    """Concrete PersistencePort. Single SQLite file."""

    def __init__(self, db_path: Path, *, encoder: Optional[EmbeddingEncoder] = None):
        self._conn = open_db(db_path)
        self._encoder = encoder
        self._vec = VectorStore(self._conn)

    # ---- Episodes ----
    def save_episode(self, episode: Episode) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, occurred_at, speaker_id, participants, intent,
                raw_utterance, tool_calls, emotional_tone, summary,
                consolidated_at, response_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.episode_id,
                _dt_to_iso(episode.occurred_at),
                episode.speaker_id,
                json.dumps(episode.participants),
                episode.intent,
                episode.raw_utterance,
                json.dumps([tc.__dict__ for tc in episode.tool_calls]),
                episode.emotional_tone,
                episode.summary,
                _dt_to_iso(episode.consolidated_at) if episode.consolidated_at else None,
                episode.response_text,
            ),
        )

    def load_episode(self, episode_id: str) -> Optional[Episode]:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return _row_to_episode(row) if row else None

    def list_recent_episodes(self, limit: int) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY occurred_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def list_unconsolidated_episodes(self) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE consolidated_at IS NULL ORDER BY occurred_at ASC"
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def mark_episodes_consolidated(self, episode_ids: list[str], at: datetime) -> None:
        with self._conn:
            for eid in episode_ids:
                self._conn.execute(
                    "UPDATE episodes SET consolidated_at = ? WHERE episode_id = ?",
                    (_dt_to_iso(at), eid),
                )

    # ---- Facts ----
    def save_fact(self, fact: Fact) -> None:
        embedding = fact.embedding
        if embedding is None and self._encoder is not None:
            embedding = self._encoder.encode(fact.value)
            fact = replace(fact, embedding=embedding)
        self._conn.execute(
            """INSERT OR REPLACE INTO facts
               (fact_id, category, key, value, tags, source_episode_id,
                embedding, valid_at, invalid_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.fact_id, fact.category, fact.key, fact.value,
                json.dumps(fact.tags), fact.source_episode_id,
                embedding,
                _dt_to_iso(fact.valid_at),
                _dt_to_iso(fact.invalid_at) if fact.invalid_at else None,
                _dt_to_iso(fact.created_at),
            ),
        )
        if embedding is not None:
            self._vec.upsert(fact.fact_id, embedding)

    def find_fact(self, category: str, key: str) -> Optional[Fact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE category=? AND key=? AND invalid_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (category, key),
        ).fetchone()
        return _row_to_fact(row) if row else None

    def find_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE fact_id = ? LIMIT 1", (fact_id,)
        ).fetchone()
        return _row_to_fact(row) if row else None

    def list_active_facts(self) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE invalid_at IS NULL"
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def invalidate_fact(self, fact_id: str, at: datetime) -> None:
        self._conn.execute(
            "UPDATE facts SET invalid_at = ? WHERE fact_id = ?",
            (_dt_to_iso(at), fact_id),
        )

    # ---- Identity ----
    def get_identity_narrative(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT narrative FROM identity WHERE id = 1"
        ).fetchone()
        return row["narrative"] if row else None

    def save_identity_narrative(self, narrative: str, last_consolidated_at: datetime,
                                last_consolidated_episode_id: Optional[str]) -> None:
        self._conn.execute(
            """INSERT INTO identity (id, narrative, last_consolidated_at,
                                     last_consolidated_episode_id)
               VALUES (1, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   narrative=excluded.narrative,
                   last_consolidated_at=excluded.last_consolidated_at,
                   last_consolidated_episode_id=excluded.last_consolidated_episode_id""",
            (narrative, _dt_to_iso(last_consolidated_at), last_consolidated_episode_id),
        )

    def get_last_consolidated_episode_id(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT last_consolidated_episode_id FROM identity WHERE id = 1"
        ).fetchone()
        return row["last_consolidated_episode_id"] if row else None

    # ---- Reflections ----
    def save_reflection(self, reflection: Reflection) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO reflections "
            "(reflection_id, occurred_at, note, source_episode_id) "
            "VALUES (?, ?, ?, ?)",
            (reflection.reflection_id, _dt_to_iso(reflection.occurred_at),
             reflection.note, reflection.source_episode_id),
        )

    def list_recent_reflections(self, limit: int) -> list[Reflection]:
        rows = self._conn.execute(
            "SELECT * FROM reflections ORDER BY occurred_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Reflection(
            reflection_id=r["reflection_id"],
            occurred_at=_iso_to_dt(r["occurred_at"]),
            note=r["note"],
            source_episode_id=r["source_episode_id"],
        ) for r in rows]

    # ---- Feedback ----
    def save_feedback(self, entry: FeedbackEntry) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO feedback "
            "(feedback_id, occurred_at, issue, status, source_episode_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry.feedback_id, _dt_to_iso(entry.occurred_at),
             entry.issue, entry.status, entry.source_episode_id),
        )

    def list_feedback(self, status: Optional[str] = None) -> list[FeedbackEntry]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM feedback ORDER BY occurred_at DESC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM feedback WHERE status = ? ORDER BY occurred_at DESC", (status,)
            ).fetchall()
        return [FeedbackEntry(
            feedback_id=r["feedback_id"], occurred_at=_iso_to_dt(r["occurred_at"]),
            issue=r["issue"], status=r["status"], source_episode_id=r["source_episode_id"],
        ) for r in rows]

    def resolve_feedback(self, feedback_id: str) -> None:
        self._conn.execute(
            "UPDATE feedback SET status='resolved' WHERE feedback_id=?", (feedback_id,)
        )

    # ---- Reminders ----
    def save_reminder(self, reminder: Reminder) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO reminders "
            "(reminder_id, name, fire_at, room_id, status, source_episode_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (reminder.reminder_id, reminder.name, _dt_to_iso(reminder.fire_at),
             reminder.room_id, reminder.status, reminder.source_episode_id),
        )

    def list_pending_reminders(self) -> list[Reminder]:
        rows = self._conn.execute(
            "SELECT * FROM reminders WHERE status='pending' ORDER BY fire_at ASC"
        ).fetchall()
        return [Reminder(
            reminder_id=r["reminder_id"], name=r["name"],
            fire_at=_iso_to_dt(r["fire_at"]), room_id=r["room_id"],
            status=r["status"], source_episode_id=r["source_episode_id"],
        ) for r in rows]

    def update_reminder_status(self, reminder_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE reminders SET status=? WHERE reminder_id=?", (status, reminder_id)
        )
