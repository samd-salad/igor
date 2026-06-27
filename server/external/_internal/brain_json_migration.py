"""One-shot brain.json -> SQLite migration. Idempotent (skips if brain.json gone).

Safety contract:
1. Snapshot brain.json to `.pre-migration-<stamp>.bak` BEFORE touching the DB.
2. Run all per-entry writes. Any exception propagates without renaming the source.
3. Only after every write succeeds, rename brain.json to `.imported-<stamp>.bak`.

Entry-type mapping:
- memory   -> Fact
- episode  -> Episode (unconsolidated)
- identity -> identity narrative
- summary  -> Episode with raw_utterance=text, marked consolidated (the existing
              identity narrative was already built on top of these; re-consolidating
              would burn Haiku calls on digested history)
- reminder -> Reminder (data.fire_at is a unix-epoch float)
- feedback -> FeedbackEntry
- routine  -> DROPPED (tool-call frequency analytics; no schema table, low value)
"""
from __future__ import annotations
import json
import shutil
import uuid
from datetime import datetime, UTC
from pathlib import Path

from server.cognition.contracts import (
    Fact, Episode, FeedbackEntry, Reminder, ToolCallRecord,
)
from server.external.sqlite_persistence import SqlitePersistence


def _parse_created(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        dt = datetime.fromisoformat(raw.rstrip("Z"))
    except ValueError:
        return datetime.now(UTC)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _fire_at_to_datetime(fire_at) -> datetime:
    """Reminder.data.fire_at is historically a unix-epoch float."""
    if isinstance(fire_at, (int, float)):
        return datetime.fromtimestamp(float(fire_at), tz=UTC)
    if isinstance(fire_at, str):
        return _parse_created(fire_at)
    return datetime.now(UTC)


def _import_memory(sp: SqlitePersistence, entry: dict, created_dt: datetime) -> None:
    data = entry.get("data") or {}
    sp.save_fact(Fact(
        fact_id=str(uuid.uuid4()),
        category=data.get("category", "unknown"),
        key=data.get("key", str(entry.get("id"))),
        value=str(data.get("value", "")),
        tags=entry.get("tags", []) or data.get("tags", []),
        source_episode_id=None,
        embedding=None,
        valid_at=created_dt,
        invalid_at=None,
        created_at=created_dt,
    ))


def _import_episode(sp: SqlitePersistence, entry: dict, created_dt: datetime,
                    *, consolidated: bool) -> None:
    data = entry.get("data") or {}
    sp.save_episode(Episode(
        episode_id=str(uuid.uuid4()),
        occurred_at=created_dt,
        speaker_id=None,
        participants=data.get("participants", []),
        intent=data.get("intent"),
        raw_utterance=data.get("raw_utterance") or data.get("text") or data.get("summary", ""),
        tool_calls=[ToolCallRecord(**tc) for tc in data.get("tool_calls", [])],
        emotional_tone=data.get("emotional_tone"),
        summary=data.get("summary"),
        consolidated_at=created_dt if consolidated else None,
    ))


def _import_identity(sp: SqlitePersistence, entry: dict, created_dt: datetime) -> None:
    data = entry.get("data") or {}
    sp.save_identity_narrative(data.get("narrative", ""), created_dt, None)


def _import_reminder(sp: SqlitePersistence, entry: dict, created_dt: datetime) -> None:
    data = entry.get("data") or {}
    sp.save_reminder(Reminder(
        reminder_id=str(uuid.uuid4()),
        name=data.get("name", "reminder"),
        fire_at=_fire_at_to_datetime(data.get("fire_at", created_dt.timestamp())),
        room_id=data.get("room_id"),
        status=entry.get("status", "pending"),
        source_episode_id=None,
    ))


def _import_feedback(sp: SqlitePersistence, entry: dict, created_dt: datetime) -> None:
    data = entry.get("data") or {}
    sp.save_feedback(FeedbackEntry(
        feedback_id=str(uuid.uuid4()),
        occurred_at=created_dt,
        issue=data.get("issue", ""),
        status=entry.get("status", "open"),
        source_episode_id=None,
    ))


def migrate_brain_json_if_needed(brain_json_path: Path, db_path: Path) -> None:
    if not brain_json_path.exists():
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    # SAFETY: snapshot the source verbatim before any DB writes. Survives even
    # if the rename-on-success step doesn't reach because of an exception.
    pre_bak = brain_json_path.with_suffix(f".json.pre-migration-{stamp}.bak")
    shutil.copy2(brain_json_path, pre_bak)

    data = json.loads(brain_json_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    sp = SqlitePersistence(db_path)

    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}

    for entry in entries:
        etype = entry.get("type")
        created_dt = _parse_created(entry.get("created"))

        if etype == "memory":
            _import_memory(sp, entry, created_dt)
        elif etype == "episode":
            _import_episode(sp, entry, created_dt, consolidated=False)
        elif etype == "summary":
            _import_episode(sp, entry, created_dt, consolidated=True)
        elif etype == "identity":
            _import_identity(sp, entry, created_dt)
        elif etype == "reminder":
            _import_reminder(sp, entry, created_dt)
        elif etype == "feedback":
            _import_feedback(sp, entry, created_dt)
        elif etype == "routine":
            dropped[etype] = dropped.get(etype, 0) + 1
            continue
        else:
            dropped[etype or "unknown"] = dropped.get(etype or "unknown", 0) + 1
            continue
        counts[etype] = counts.get(etype, 0) + 1

    if dropped:
        print(f"brain.json migration: imported {counts}, dropped {dropped}")
    else:
        print(f"brain.json migration: imported {counts}")

    # All writes succeeded — rename source so subsequent boots no-op.
    brain_json_path.rename(brain_json_path.with_suffix(f".json.imported-{stamp}.bak"))
