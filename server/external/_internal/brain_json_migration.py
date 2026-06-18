"""One-shot brain.json → SQLite migration. Idempotent (skips if brain.json gone)."""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

from server.cognition.contracts import Fact, Episode, ToolCallRecord
from server.external.sqlite_persistence import SqlitePersistence


def migrate_brain_json_if_needed(brain_json_path: Path, db_path: Path) -> None:
    if not brain_json_path.exists():
        return
    data = json.loads(brain_json_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    sp = SqlitePersistence(db_path)

    for e in entries:
        etype = e.get("type")
        edata = e.get("data") or {}
        created = e.get("created") or datetime.utcnow().isoformat()
        try:
            created_dt = datetime.fromisoformat(created.rstrip("Z"))
        except ValueError:
            created_dt = datetime.utcnow()

        if etype == "memory":
            sp.save_fact(Fact(
                fact_id=str(uuid.uuid4()),
                category=edata.get("category", "unknown"),
                key=edata.get("key", str(e.get("id"))),
                value=str(edata.get("value", "")),
                tags=e.get("tags", []) or edata.get("tags", []),
                source_episode_id=None,
                embedding=None,
                valid_at=created_dt,
                invalid_at=None,
                created_at=created_dt,
            ))
        elif etype == "episode":
            sp.save_episode(Episode(
                episode_id=str(uuid.uuid4()),
                occurred_at=created_dt,
                speaker_id=None,
                participants=edata.get("participants", []),
                intent=edata.get("intent"),
                raw_utterance=edata.get("raw_utterance") or edata.get("summary", ""),
                tool_calls=[ToolCallRecord(**tc) for tc in edata.get("tool_calls", [])],
                emotional_tone=edata.get("emotional_tone"),
                summary=edata.get("summary"),
                consolidated_at=None,
            ))
        elif etype == "identity":
            sp.save_identity_narrative(
                edata.get("narrative", ""),
                created_dt, None,
            )

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    brain_json_path.rename(brain_json_path.with_suffix(f".json.imported-{stamp}.bak"))
