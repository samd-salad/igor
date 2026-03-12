"""Unified Brain Store — single typed event store for memory, routines, feedback, reminders, and summaries.

All persistent knowledge lives in data/brain.json with a unified schema:
  - type: memory | routine | feedback | reminder | summary
  - tags: primary retrieval key (keyword matching against transcription)
  - status: lifecycle (active, open, resolved, pending, fired, cancelled, archived)
  - data: type-specific payload

One threading.Lock for all operations. Atomic JSON I/O via tmp-rename pattern.
In-memory dict index by id + tag→id inverted index for fast retrieval.
"""
import json
import logging
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Entry type prefixes for readable IDs
_TYPE_PREFIXES = {
    "memory": "mem",
    "routine": "rtn",
    "feedback": "fbk",
    "reminder": "rem",
    "summary": "sum",
}

_MAX_ROUTINES = 1000
_SUMMARY_ARCHIVE_DAYS = 30


class BrainStore:
    """Unified persistent knowledge store."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}  # id -> entry
        self._tag_index: dict[str, set[str]] = defaultdict(set)  # tag -> set of ids
        self._counters: dict[str, int] = {}  # type_prefix -> next number
        self._load()

    # ---- Core CRUD ----

    def add(self, entry_type: str, data: dict, tags: list[str] = None,
            status: str = "active", entry_id: str = None) -> str:
        """Add a new entry. Returns the assigned ID."""
        with self._lock:
            if entry_id is None:
                entry_id = self._next_id(entry_type)
            now = datetime.now().isoformat(timespec="seconds")
            tags = [t.lower() for t in (tags or [])]
            entry = {
                "id": entry_id,
                "type": entry_type,
                "created": now,
                "updated": now,
                "tags": tags,
                "status": status,
                "data": data,
            }
            self._entries[entry_id] = entry
            for tag in tags:
                self._tag_index[tag].add(entry_id)
            self._save()
            return entry_id

    def update(self, entry_id: str, data: dict = None, tags: list[str] = None,
               status: str = None) -> bool:
        """Update an existing entry. Returns False if not found."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            if data is not None:
                entry["data"] = data
            if tags is not None:
                # Remove old tag index entries
                for old_tag in entry["tags"]:
                    self._tag_index[old_tag].discard(entry_id)
                entry["tags"] = [t.lower() for t in tags]
                for tag in entry["tags"]:
                    self._tag_index[tag].add(entry_id)
            if status is not None:
                entry["status"] = status
            entry["updated"] = datetime.now().isoformat(timespec="seconds")
            self._save()
            return True

    def get(self, entry_id: str) -> Optional[dict]:
        """Get a single entry by ID."""
        with self._lock:
            entry = self._entries.get(entry_id)
            return dict(entry) if entry else None

    def remove(self, entry_id: str) -> bool:
        """Remove an entry entirely. Returns False if not found."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            for tag in entry["tags"]:
                self._tag_index[tag].discard(entry_id)
            del self._entries[entry_id]
            self._save()
            return True

    def query(self, entry_type: str = None, status: str = None,
              tags: list[str] = None, limit: int = 0) -> list[dict]:
        """Query entries by type, status, and/or tags. Returns copies."""
        with self._lock:
            if tags:
                # Intersect tag index sets for AND matching
                candidate_ids = None
                for tag in tags:
                    tag_ids = self._tag_index.get(tag.lower(), set())
                    if candidate_ids is None:
                        candidate_ids = set(tag_ids)
                    else:
                        candidate_ids &= tag_ids
                if not candidate_ids:
                    return []
                candidates = [self._entries[eid] for eid in candidate_ids if eid in self._entries]
            else:
                candidates = list(self._entries.values())

            results = []
            for e in candidates:
                if entry_type and e["type"] != entry_type:
                    continue
                if status and e["status"] != status:
                    continue
                results.append(dict(e))

            # Sort by created descending (newest first)
            results.sort(key=lambda e: e.get("created", ""), reverse=True)
            if limit > 0:
                results = results[:limit]
            return results

    # ---- Memory-specific helpers ----

    def find_memory(self, category: str, key: str) -> Optional[dict]:
        """Find a memory entry by category+key (used by save/forget commands)."""
        with self._lock:
            for e in self._entries.values():
                if (e["type"] == "memory" and e["status"] == "active"
                        and e["data"].get("category") == category
                        and e["data"].get("key") == key):
                    return dict(e)
            return None

    def save_memory(self, category: str, key: str, value: str) -> tuple[str, bool]:
        """Save or update a memory entry. Returns (id, is_update).

        Atomic: holds lock across find+add/update to prevent duplicate entries
        from concurrent threads (e.g. session summarizer + user command).
        """
        tags = [category, key]
        tags.extend(self._extract_tags(value))
        data = {"category": category, "key": key, "value": value}
        with self._lock:
            # Find existing under same lock
            existing_id = None
            for e in self._entries.values():
                if (e["type"] == "memory" and e["status"] == "active"
                        and e["data"].get("category") == category
                        and e["data"].get("key") == key):
                    existing_id = e["id"]
                    break
            if existing_id:
                # Inline update (already holding lock)
                entry = self._entries[existing_id]
                entry["data"] = data
                for old_tag in entry["tags"]:
                    self._tag_index[old_tag].discard(existing_id)
                entry["tags"] = [t.lower() for t in tags]
                for tag in entry["tags"]:
                    self._tag_index[tag].add(existing_id)
                entry["updated"] = datetime.now().isoformat(timespec="seconds")
                self._save()
                return existing_id, True
            # Inline add (already holding lock)
            entry_id = self._next_id("memory")
            now = datetime.now().isoformat(timespec="seconds")
            entry = {
                "id": entry_id, "type": "memory", "created": now, "updated": now,
                "tags": [t.lower() for t in tags], "status": "active", "data": data,
            }
            self._entries[entry_id] = entry
            for tag in entry["tags"]:
                self._tag_index[tag].add(entry_id)
            self._save()
            return entry_id, False

    def forget_memory(self, category: str, key: str) -> bool:
        """Remove a memory entry by category+key."""
        existing = self.find_memory(category, key)
        if existing:
            return self.remove(existing["id"])
        return False

    def get_all_memories(self) -> dict:
        """Get all active memories in the old {category: {key: value}} format."""
        result = {}
        with self._lock:
            for e in self._entries.values():
                if e["type"] == "memory" and e["status"] == "active":
                    cat = e["data"].get("category", "other")
                    key = e["data"].get("key", "")
                    val = e["data"].get("value", "")
                    if cat and key:
                        if cat not in result:
                            result[cat] = {}
                        result[cat][key] = val
        return result

    # ---- Feedback-specific helpers ----

    def add_feedback(self, issue: str, suggestion: str = "", context: str = "") -> int:
        """Add a feedback entry. Returns the numeric feedback ID.

        Atomic: holds lock across ID generation + add to prevent ID collisions.
        """
        tags = ["feedback"]
        tags.extend(self._extract_tags(issue))
        with self._lock:
            max_num = 0
            for e in self._entries.values():
                if e["type"] == "feedback":
                    try:
                        num = int(e["id"].split("_")[1])
                        max_num = max(max_num, num)
                    except (IndexError, ValueError):
                        pass
            new_num = max_num + 1
            entry_id = f"fbk_{new_num:03d}"
            now = datetime.now()
            entry = {
                "id": entry_id, "type": "feedback",
                "created": now.isoformat(timespec="seconds"),
                "updated": now.isoformat(timespec="seconds"),
                "tags": [t.lower() for t in tags], "status": "open",
                "data": {
                    "id": new_num, "issue": issue, "suggestion": suggestion,
                    "context": context, "timestamp": now.strftime("%Y-%m-%d %H:%M"),
                },
            }
            self._entries[entry_id] = entry
            for tag in entry["tags"]:
                self._tag_index[tag].add(entry_id)
            self._save()
        return new_num

    def list_feedback(self, status: str = "open") -> list[dict]:
        """List feedback entries filtered by status."""
        if status == "all":
            return self.query(entry_type="feedback")
        return self.query(entry_type="feedback", status=status)

    def resolve_feedback(self, feedback_num: int) -> Optional[dict]:
        """Resolve a feedback entry by its numeric ID. Returns the entry or None."""
        entry_id = f"fbk_{feedback_num:03d}"
        entry = self.get(entry_id)
        if not entry:
            # Try without zero-padding for legacy entries
            with self._lock:
                for e in self._entries.values():
                    if e["type"] == "feedback" and e["data"].get("id") == feedback_num:
                        entry = dict(e)
                        entry_id = e["id"]
                        break
        if not entry:
            return None
        self.update(entry_id, status="resolved")
        return entry

    # ---- Routine-specific helpers ----

    def log_routine(self, command: str, hour: int, day: int, params: dict = None):
        """Log a command execution as a routine entry."""
        tags = [command]
        if 6 <= hour < 12:
            tags.append("morning")
        elif 12 <= hour < 17:
            tags.append("afternoon")
        elif 17 <= hour < 22:
            tags.append("evening")
        else:
            tags.append("night")
        data = {"command": command, "hour": hour, "day": day}
        if params:
            data["params"] = params
        self.add("routine", data, tags=tags)

    def get_routine_entries(self) -> list[dict]:
        """Get all routine entries for pattern analysis."""
        return self.query(entry_type="routine")

    # ---- Reminder-specific helpers ----

    def add_reminder(self, name: str, fire_at: float, duration_seconds: float,
                     room_id: str = "default") -> str:
        """Add a persistent reminder. Returns the entry ID."""
        tags = ["reminder", name.lower()]
        return self.add("reminder", {
            "name": name,
            "fire_at": fire_at,
            "duration_seconds": duration_seconds,
            "room_id": room_id,
        }, tags=tags, status="pending")

    def fire_reminder(self, entry_id: str):
        """Mark a reminder as fired."""
        self.update(entry_id, status="fired")

    def cancel_reminder(self, name: str) -> bool:
        """Cancel a pending reminder by name. Atomic: find+update under single lock."""
        with self._lock:
            for e in self._entries.values():
                if (e["type"] == "reminder" and e["status"] == "pending"
                        and e["data"].get("name") == name):
                    e["status"] = "cancelled"
                    e["updated"] = datetime.now().isoformat(timespec="seconds")
                    self._save()
                    return True
            return False

    def get_pending_reminders(self) -> list[dict]:
        """Get all pending reminders."""
        return self.query(entry_type="reminder", status="pending")

    # ---- Summary-specific helpers ----

    def add_summary(self, text: str, topic_tags: list[str] = None):
        """Add a conversation summary entry."""
        tags = ["conversation"]
        if topic_tags:
            tags.extend(topic_tags)
        self.add("summary", {"text": text}, tags=tags)

    def get_recent_summaries(self, limit: int = 3) -> list[dict]:
        """Get the most recent active summaries."""
        return self.query(entry_type="summary", status="active", limit=limit)

    # ---- Retrieval (Phase 2) ----

    def retrieve_relevant(self, query: str, limit: int = 10) -> list[dict]:
        """Retrieve entries most relevant to a query string by tag overlap.

        Always includes behavior category entries. Scores by number of
        matching tags. Returns top N entries.
        """
        query_tokens = self._extract_tags(query)
        if not query_tokens:
            # No useful tokens — return behavior entries only
            return self.query(entry_type="memory", tags=["behavior"])

        scored = []
        with self._lock:
            for e in self._entries.values():
                if e["status"] not in ("active", "open", "pending"):
                    continue
                # Behavior entries always included
                if (e["type"] == "memory"
                        and e["data"].get("category") == "behavior"):
                    scored.append((e, 100))  # High score ensures inclusion
                    continue
                # Score by tag overlap
                entry_tags = set(e.get("tags", []))
                overlap = len(entry_tags & query_tokens)
                if overlap > 0:
                    scored.append((e, overlap))

        scored.sort(key=lambda x: -x[1])
        return [dict(e) for e, _ in scored[:limit]]

    def get_behavior_rules(self) -> str:
        """Format behavior category memories for the base system prompt block."""
        behaviors = []
        with self._lock:
            for e in self._entries.values():
                if (e["type"] == "memory" and e["status"] == "active"
                        and e["data"].get("category") == "behavior"):
                    behaviors.append(e["data"])
        if not behaviors:
            return ""
        lines = ["[behavior guidelines]"]
        for i, b in enumerate(sorted(behaviors, key=lambda x: x.get("key", "")), 1):
            lines.append(f"  {i}. {b.get('value', '')}")
        return "\n".join(lines)

    def format_relevant(self, entries: list[dict]) -> str:
        """Format retrieved entries for injection into the dynamic prompt block."""
        if not entries:
            return "(empty)"

        sections = {"memories": [], "feedback": [], "summaries": []}
        for e in entries:
            if e["type"] == "memory":
                cat = e["data"].get("category", "other")
                key = e["data"].get("key", "")
                val = e["data"].get("value", "")
                if cat == "behavior":
                    continue  # Behavior is in base block
                sections["memories"].append(f"  [{cat}] {key}: {val}")
            elif e["type"] == "feedback" and e["status"] == "open":
                sections["feedback"].append(f"  #{e['data'].get('id', '?')}: {e['data'].get('issue', '')}")
            elif e["type"] == "summary":
                sections["summaries"].append(f"  - {e['data'].get('text', '')}")

        parts = []
        if sections["memories"]:
            parts.append("Relevant memories:\n" + "\n".join(sections["memories"]))
        if sections["summaries"]:
            parts.append("Recent sessions:\n" + "\n".join(sections["summaries"]))
        if sections["feedback"]:
            parts.append("Open feedback:\n" + "\n".join(sections["feedback"]))

        return "\n\n".join(parts) if parts else "(empty)"

    # ---- Contextual defaults (Phase 5) ----

    def query_defaults(self, command: str, hour: int, min_occurrences: int = 5) -> dict:
        """Find most common parameters for a command at a given time bucket.

        Returns empty dict if fewer than min_occurrences found.
        """
        bucket = (hour // 2) * 2
        matching_params = []
        with self._lock:
            for e in self._entries.values():
                if (e["type"] == "routine" and e["data"].get("command") == command
                        and (e["data"].get("hour", -1) // 2) * 2 == bucket
                        and e["data"].get("params")):
                    matching_params.append(e["data"]["params"])

        if len(matching_params) < min_occurrences:
            return {}

        # Find most common value for each param key
        from collections import Counter
        defaults = {}
        all_keys = set()
        for p in matching_params:
            all_keys.update(p.keys())
        for key in all_keys:
            values = [p[key] for p in matching_params if key in p]
            if values:
                most_common = Counter(str(v) for v in values).most_common(1)
                if most_common and most_common[0][1] >= min_occurrences:
                    defaults[key] = most_common[0][0]
        return defaults

    # ---- Compaction ----

    def compact(self):
        """Trim routines to max entries, archive old summaries."""
        with self._lock:
            # Trim routines
            routines = [(eid, e) for eid, e in self._entries.items() if e["type"] == "routine"]
            if len(routines) > _MAX_ROUTINES:
                routines.sort(key=lambda x: x[1].get("created", ""))
                to_remove = routines[:len(routines) - _MAX_ROUTINES]
                for eid, e in to_remove:
                    for tag in e.get("tags", []):
                        self._tag_index[tag].discard(eid)
                    del self._entries[eid]

            # Archive old summaries
            cutoff = (datetime.now() - timedelta(days=_SUMMARY_ARCHIVE_DAYS)).isoformat(timespec="seconds")
            for e in self._entries.values():
                if (e["type"] == "summary" and e["status"] == "active"
                        and e.get("created", "") < cutoff):
                    e["status"] = "archived"

            self._save()

    # ---- Migration ----

    def migrate_legacy_files(self, data_dir: Path):
        """Migrate memory.json, routines.json, feedback.json into brain.json.

        Idempotent: skips if brain.json already has entries.
        Renames originals to *.migrated (never deleted — rollback safe).
        """
        if self._entries:
            logger.info("Brain already has entries, skipping migration")
            return

        migrated_any = False

        # Migrate memory.json
        memory_file = data_dir / "memory.json"
        if memory_file.exists():
            try:
                memories = json.loads(memory_file.read_text())
                count = 0
                for category, items in memories.items():
                    if isinstance(items, dict):
                        for key, value in items.items():
                            tags = [category, key]
                            tags.extend(self._extract_tags(str(value)))
                            self.add("memory", {
                                "category": category,
                                "key": key,
                                "value": str(value),
                            }, tags=tags)
                            count += 1
                memory_file.rename(memory_file.with_suffix(".migrated"))
                logger.info(f"Migrated {count} memories from memory.json")
                migrated_any = True
            except Exception as e:
                logger.error(f"Failed to migrate memory.json: {e}")

        # Migrate routines.json
        routines_file = data_dir / "routines.json"
        if routines_file.exists():
            try:
                entries = json.loads(routines_file.read_text())
                count = 0
                for entry in entries:
                    if isinstance(entry, dict) and "command" in entry:
                        cmd = entry["command"]
                        hour = entry.get("hour", 0)
                        day = entry.get("day", 0)
                        tags = [cmd]
                        if 6 <= hour < 12:
                            tags.append("morning")
                        elif 12 <= hour < 17:
                            tags.append("afternoon")
                        elif 17 <= hour < 22:
                            tags.append("evening")
                        else:
                            tags.append("night")
                        self.add("routine", {
                            "command": cmd,
                            "hour": hour,
                            "day": day,
                        }, tags=tags)
                        count += 1
                routines_file.rename(routines_file.with_suffix(".migrated"))
                logger.info(f"Migrated {count} routine entries from routines.json")
                migrated_any = True
            except Exception as e:
                logger.error(f"Failed to migrate routines.json: {e}")

        # Migrate feedback.json
        feedback_file = data_dir / "feedback.json"
        if feedback_file.exists():
            try:
                items = json.loads(feedback_file.read_text())
                count = 0
                for item in items:
                    if isinstance(item, dict):
                        num = item.get("id", count + 1)
                        entry_id = f"fbk_{num:03d}"
                        tags = ["feedback"]
                        tags.extend(self._extract_tags(item.get("issue", "")))
                        status = item.get("status", "open")
                        self.add("feedback", {
                            "id": num,
                            "issue": item.get("issue", ""),
                            "suggestion": item.get("suggestion", ""),
                            "context": item.get("context", ""),
                            "timestamp": item.get("timestamp", ""),
                        }, tags=tags, status=status, entry_id=entry_id)
                        count += 1
                feedback_file.rename(feedback_file.with_suffix(".migrated"))
                logger.info(f"Migrated {count} feedback items from feedback.json")
                migrated_any = True
            except Exception as e:
                logger.error(f"Failed to migrate feedback.json: {e}")

        if migrated_any:
            logger.info(f"Migration complete — {len(self._entries)} total entries in brain.json")

    # ---- Internal ----

    def _next_id(self, entry_type: str) -> str:
        """Generate the next sequential ID for a given type."""
        prefix = _TYPE_PREFIXES.get(entry_type, entry_type[:3])
        counter = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = counter
        return f"{prefix}_{counter:03d}"

    def _load(self):
        """Load brain.json into memory and rebuild indexes."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            entries = raw.get("entries", [])
            for e in entries:
                eid = e.get("id", "")
                self._entries[eid] = e
                for tag in e.get("tags", []):
                    self._tag_index[tag].add(eid)
            # Rebuild counters from existing IDs
            for eid in self._entries:
                parts = eid.rsplit("_", 1)
                if len(parts) == 2:
                    prefix = parts[0]
                    try:
                        num = int(parts[1])
                        self._counters[prefix] = max(self._counters.get(prefix, 0), num)
                    except ValueError:
                        pass
            logger.info(f"Brain loaded: {len(self._entries)} entries from {self._path.name}")
        except json.JSONDecodeError:
            backup = self._path.with_name(f"brain.bak.{int(time.time())}.json")
            self._path.rename(backup)
            logger.warning(f"Brain file corrupted — backed up to {backup.name}")
        except Exception as e:
            logger.error(f"Failed to load brain: {e}")

    def _save(self):
        """Save brain.json atomically (caller must hold _lock)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            data = {
                "version": 1,
                "entries": list(self._entries.values()),
            }
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            logger.error(f"Failed to save brain.json: {e}")
            # In-memory state is still correct; next successful save will persist

    @staticmethod
    def _extract_tags(text: str) -> set[str]:
        """Extract meaningful keywords from text for tag indexing.

        Strips common stop words and short tokens. Returns lowercase set.
        """
        _STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about", "like",
            "through", "after", "over", "between", "out", "up", "down", "off",
            "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "no", "only", "own", "same", "than",
            "too", "very", "just", "because", "if", "when", "while", "how",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "it", "its", "they", "them", "their",
        }
        words = set(re.findall(r'[a-z0-9]+', text.lower()))
        return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


# ---- Singleton ----

_brain: Optional[BrainStore] = None


def get_brain() -> BrainStore:
    """Get the global BrainStore singleton. Must call init_brain() first."""
    global _brain
    if _brain is None:
        raise RuntimeError("BrainStore not initialized — call init_brain() at startup")
    return _brain


def init_brain(path: Path) -> BrainStore:
    """Initialize the global BrainStore singleton."""
    global _brain
    if _brain is not None:
        logger.warning("BrainStore already initialized")
        return _brain
    _brain = BrainStore(path)
    return _brain
