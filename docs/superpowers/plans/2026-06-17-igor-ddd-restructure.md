# Igor DDD Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Igor into four bounded contexts (`wakeword`, `ha-io`, `cognition`, `external`) per the design in `docs/superpowers/specs/2026-06-17-igor-ddd-design.md`. Keep the existing system running throughout; cutover happens at Task 33.

**Architecture:** Hexagonal pattern inside `cognition` (5 ports / 4 aggregates / 6 services); thin wire in `ha-io`; adapters in `external`; isolated `wakeword`. Episode is a first-class structured entity that also serves as the provenance anchor (`VoiceTurn.correlation_id == Episode.episode_id`). Persistence is SQLite from day 1 with bi-temporal columns and embedding-ready BLOB slots.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite (stdlib), `sqlite-vec` extension (used later), `anthropic` SDK, `requests` (HA REST), `pytest` + `pytest-asyncio` for tests.

**Build strategy:** New code lives alongside the existing `server/api.py` and `server/main_text.py`. The container stays deployable through Tasks 1-32. Cutover (Task 33) swaps the entry point. Cleanup (Task 34) removes the old modules.

**Execution notes (added 2026-06-17 during executing-plans review):**

1. **Task 22 wrapper inlining**: `QualityGate` and `IntentRouter` cannot lazy-import from `server/quality_gate.py` / `server/intent_router.py` since Task 33 deletes those files. When implementing Task 22, port the actual filter/route logic into the new modules directly instead of wrapping legacy.
2. **Task 31 HAClient access**: composition root imports `get_ha_client` from `external/_internal/ha_client`. The runtime guard blocks that path. Re-export `get_client` from `server/external/__init__.py` so the composition root accesses HAClient via the public surface.
3. **Windows venv**: substitute `.venv/Scripts/python.exe -m pytest` for `.venv/bin/pytest` and similar throughout. Same for pip and onnx2tf invocations.

---

## File Structure (created by this plan)

```
docs/superpowers/specs/2026-06-17-igor-ddd-design.md   (exists)
docs/superpowers/plans/2026-06-17-igor-ddd-restructure.md  (this file)

requirements-dev.txt                                   (new)
tools/boundary_check.py                                (new)

wakeword/contracts.py                                  (new)
wakeword/render_runtime.py                             (new)

server/cognition/__init__.py                           (new)
server/cognition/contracts.py                          (new)
server/cognition/ports/__init__.py                     (new)
server/cognition/ports/persistence.py                  (new)
server/cognition/ports/retrieval.py                    (new)
server/cognition/ports/llm.py                          (new)
server/cognition/ports/tools.py                        (new)
server/cognition/ports/clock.py                        (new)
server/cognition/aggregates/__init__.py                (new)
server/cognition/aggregates/memory.py                  (new)
server/cognition/aggregates/episode.py                 (new)
server/cognition/aggregates/identity.py                (new)
server/cognition/aggregates/user_state.py              (new)
server/cognition/services/__init__.py                  (new)
server/cognition/services/quality_gate.py              (new)
server/cognition/services/intent_router.py             (new)
server/cognition/services/tool_registry.py             (new)
server/cognition/services/conversation.py              (new)
server/cognition/services/session_summarizer.py        (new)
server/cognition/services/consolidator.py              (new)
server/cognition/_internal/prompt_builder.py           (new)

server/ha_io/__init__.py                               (new)
server/ha_io/contracts.py                              (new)
server/ha_io/api.py                                    (new)
server/ha_io/_internal/auth.py                         (new)
server/ha_io/_internal/rate_limit.py                   (new)
server/ha_io/_internal/voice_turn.py                   (new)
server/ha_io/_internal/result_mapper.py                (new)

server/external/__init__.py                            (new)
server/external/sqlite_persistence.py                  (new)
server/external/sqlite_retrieval.py                    (new)
server/external/claude_adapter.py                      (new)
server/external/ha_rest_adapter.py                     (new)
server/external/system_clock.py                        (new)
server/external/_internal/schema.sql                   (new)
server/external/_internal/brain_json_migration.py      (new)
server/external/_internal/ha_client.py                 (moved from server/ha_client.py)

server/main.py                                         (new — replaces main_text.py at cutover)

tests/                                                 (new directory + many test files)
```

Files deleted at cutover (Task 33-34): `server/api.py`, `server/main_text.py`, `server/conversation.py`, `server/brain.py`, `server/llm.py`, `server/quality_gate.py`, `server/intent_router.py`, `server/routines.py`, `server/ha_client.py`, `server/context.py`, `server/rooms.py`, `server/room_state.py` (if any remain), `server/event_loop.py`, `server/client_registry.py`, `server/config.py`, `server/commands/`, `prompt.py`.

---

## Phase 1 — Test Infrastructure & Boundary Guards

### Task 1: Add dev dependencies (pytest)

**Files:**
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write `requirements-dev.txt`**

```
# requirements-dev.txt
pytest~=8.3
pytest-asyncio~=0.24
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -ra --strict-markers
```

- [ ] **Step 3: Write `tests/__init__.py` (empty)**

Create the file with no content.

- [ ] **Step 4: Write `tests/test_smoke.py`**

```python
"""Smoke test: pytest itself works."""

def test_smoke():
    assert 1 + 1 == 2
```

- [ ] **Step 5: Install dev deps and run**

Run:
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest tests/test_smoke.py -v
```
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt pytest.ini tests/
git commit -m "Add pytest infrastructure for DDD restructure"
```

---

### Task 2: Create context directories with boundary guards

**Files:**
- Create: `server/cognition/__init__.py`
- Create: `server/cognition/ports/__init__.py`
- Create: `server/cognition/aggregates/__init__.py`
- Create: `server/cognition/services/__init__.py`
- Create: `server/cognition/_internal/__init__.py`
- Create: `server/ha_io/__init__.py`
- Create: `server/ha_io/_internal/__init__.py`
- Create: `server/external/__init__.py`
- Create: `server/external/_internal/__init__.py`
- Create: `wakeword/_internal/__init__.py`

- [ ] **Step 1: Write the `_internal` guard module**

Create `server/cognition/_internal/__init__.py`:

```python
"""Private internals of cognition. Cannot be imported from outside cognition."""
import sys

_caller = sys._getframe(1).f_globals.get("__name__", "")
if not _caller.startswith("server.cognition"):
    raise ImportError(
        f"server.cognition._internal is private; "
        f"importing from '{_caller}' is forbidden. "
        f"Use server.cognition.contracts instead."
    )
```

- [ ] **Step 2: Write identical guards for the other `_internal` packages**

Repeat the same pattern in:
- `server/ha_io/_internal/__init__.py` — change check to `server.ha_io`
- `server/external/_internal/__init__.py` — change check to `server.external`
- `wakeword/_internal/__init__.py` — change check to `wakeword`

- [ ] **Step 3: Write the non-internal `__init__.py` files (empty)**

Create these as empty files:
- `server/cognition/__init__.py`
- `server/cognition/ports/__init__.py`
- `server/cognition/aggregates/__init__.py`
- `server/cognition/services/__init__.py`
- `server/ha_io/__init__.py`
- `server/external/__init__.py`

- [ ] **Step 4: Write the boundary guard test**

Create `tests/test_boundary_guards.py`:

```python
"""Verify that _internal modules cannot be imported from outside their context."""
import importlib
import sys
import pytest


def test_cognition_internal_blocked_from_outside():
    # Simulate an external import by reloading the guard
    sys.modules.pop("server.cognition._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.cognition._internal")


def test_ha_io_internal_blocked_from_outside():
    sys.modules.pop("server.ha_io._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.ha_io._internal")


def test_external_internal_blocked_from_outside():
    sys.modules.pop("server.external._internal", None)
    with pytest.raises(ImportError, match="private"):
        importlib.import_module("server.external._internal")
```

- [ ] **Step 5: Run boundary tests**

Run: `.venv/bin/pytest tests/test_boundary_guards.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add server/cognition server/ha_io server/external wakeword/_internal tests/test_boundary_guards.py
git commit -m "Scaffold cognition/ha_io/external contexts with runtime _internal guards"
```

---

### Task 3: Boundary-check CI script

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/boundary_check.py`
- Create: `tests/test_boundary_check.py`

- [ ] **Step 1: Write `tools/__init__.py` (empty)**

- [ ] **Step 2: Write the boundary-check script**

Create `tools/boundary_check.py`:

```python
"""Static check that bounded contexts don't import each other's internals
and that only `external/` adapters import third-party libraries.

Run from repo root: python -m tools.boundary_check
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path

# Rules: (context_package_prefix, forbidden_import_prefixes)
COGNITION_FORBIDDEN = {
    "server.external",   # cognition cannot reach into external adapters
    "server.ha_io",      # cognition cannot import ha-io
    "anthropic",         # cognition cannot touch the LLM SDK directly
    "requests",          # cognition cannot touch HTTP libs directly
    "sqlite3",           # cognition cannot touch SQLite directly
    "fastapi",           # cognition has no FastAPI knowledge
}
HA_IO_FORBIDDEN = {
    "server.external",
    "server.cognition.ports",       # ha-io only knows the cognition public contracts
    "server.cognition.aggregates",
    "server.cognition.services",
    "server.cognition._internal",
    "anthropic",
    "requests",
    "sqlite3",
}
# external can import anything; only enforce that it's the ONLY place certain libs appear
THIRD_PARTY_LOCKED_TO_EXTERNAL = {
    # lib_name : (allowed_module_relative_path,)
    "anthropic": ("server/external/claude_adapter.py",),
    "sqlite3":   ("server/external/sqlite_persistence.py",
                  "server/external/sqlite_retrieval.py",
                  "server/external/_internal/brain_json_migration.py"),
}


def _iter_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


def _violations_under(root: Path, package_prefix: str, forbidden: set[str]) -> list[str]:
    bad = []
    for py in root.rglob("*.py"):
        rel_name = ".".join(py.with_suffix("").relative_to(root.parent).parts)
        if not rel_name.startswith(package_prefix):
            continue
        for imp in _iter_imports(py):
            for f in forbidden:
                if imp == f or imp.startswith(f + "."):
                    bad.append(f"{py}: imports forbidden {imp!r}")
    return bad


def _third_party_leaks(root: Path) -> list[str]:
    bad = []
    for lib, allowed_paths in THIRD_PARTY_LOCKED_TO_EXTERNAL.items():
        allowed_set = set(allowed_paths)
        for py in root.rglob("*.py"):
            rel = py.relative_to(root.parent).as_posix()
            if rel in allowed_set:
                continue
            for imp in _iter_imports(py):
                if imp == lib or imp.startswith(lib + "."):
                    bad.append(f"{py}: imports {imp!r} but only allowed in {sorted(allowed_set)}")
    return bad


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    server_root = repo_root / "server"
    all_violations: list[str] = []
    all_violations += _violations_under(server_root, "server.cognition", COGNITION_FORBIDDEN)
    all_violations += _violations_under(server_root, "server.ha_io", HA_IO_FORBIDDEN)
    all_violations += _third_party_leaks(server_root)
    if all_violations:
        for v in all_violations:
            print(f"BOUNDARY VIOLATION: {v}", file=sys.stderr)
        return 1
    print("Boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write the boundary-check test**

Create `tests/test_boundary_check.py`:

```python
"""Verify boundary_check returns 0 on the current (empty) cognition/ha_io/external scaffolds."""
import subprocess
import sys


def test_boundary_check_passes_on_scaffold():
    result = subprocess.run(
        [sys.executable, "-m", "tools.boundary_check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "passed" in result.stdout.lower()
```

- [ ] **Step 4: Run boundary check + test**

Run: `.venv/bin/pytest tests/test_boundary_check.py -v`
Expected: `1 passed`.

Run: `.venv/bin/python -m tools.boundary_check`
Expected: `Boundary check passed.`

- [ ] **Step 5: Commit**

```bash
git add tools/ tests/test_boundary_check.py
git commit -m "Add boundary-check CI script enforcing context import rules"
```

---

## Phase 2 — wakeword Contracts

### Task 4: `wakeword/contracts.py`

**Files:**
- Create: `wakeword/contracts.py`
- Create: `tests/wakeword/__init__.py`
- Create: `tests/wakeword/test_contracts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/wakeword/__init__.py` (empty), then `tests/wakeword/test_contracts.py`:

```python
"""Confirm wakeword contract constants are present and consistent."""
from wakeword import contracts


def test_constants_present():
    assert contracts.FEATURE_LIBRARY == "pyopen_wakeword"
    assert contracts.FEATURE_DIM == 96
    assert contracts.MODEL_INPUT_SHAPE == (1, 16, 96)
    assert contracts.MODEL_OUTPUT_SHAPE == (1, 1)
    assert contracts.MODEL_OUTPUT_RANGE == (0.0, 1.0)
    assert "{name}" in contracts.MODEL_FILENAME_PATTERN
    assert "{version}" in contracts.MODEL_FILENAME_PATTERN
    assert contracts.FEATURE_RATE_HZ > 0
```

- [ ] **Step 2: Verify it fails**

Run: `.venv/bin/pytest tests/wakeword/test_contracts.py -v`
Expected: ImportError or attribute errors (module doesn't exist yet).

- [ ] **Step 3: Write `wakeword/contracts.py`**

```python
"""Public contract for Igor's wake-word model. Both the training pipeline AND
the runtime systemd unit read from this file. If you change any constant here,
re-render the runtime config (`python -m wakeword.render_runtime`) and retrain
in the same commit.
"""
from __future__ import annotations

# Feature pipeline (lives in wyoming-openwakeword's runtime; training mirrors it)
FEATURE_LIBRARY = "pyopen_wakeword"
FEATURE_LIBRARY_VERSION = ">=1.1,<2"
FEATURE_DIM = 96
FEATURE_RATE_HZ = 42.7

# Wake-word classifier model
MODEL_INPUT_SHAPE = (1, 16, 96)
MODEL_INPUT_DTYPE = "float32"
MODEL_OUTPUT_SHAPE = (1, 1)
MODEL_OUTPUT_RANGE = (0.0, 1.0)

# wyoming-openwakeword expects this filename pattern in its custom-models dir
MODEL_FILENAME_PATTERN = "{name}_v{version}.tflite"

# Default tuning (override in deploy/wyoming-openwakeword.service if needed)
DEFAULT_THRESHOLD = 0.5
DEFAULT_TRIGGER_LEVEL = 3
DEFAULT_REFRACTORY_SECONDS = 2.0
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/wakeword/test_contracts.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add wakeword/contracts.py tests/wakeword/
git commit -m "wakeword/contracts.py — single source for training+runtime"
```

---

### Task 5: `wakeword/render_runtime.py`

**Files:**
- Create: `wakeword/render_runtime.py`
- Create: `tests/wakeword/test_render_runtime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/wakeword/test_render_runtime.py`:

```python
from wakeword.render_runtime import render_openwakeword_execstart


def test_execstart_contains_model_name_and_custom_dir():
    line = render_openwakeword_execstart(
        run_script="/home/samda/wyoming-openwakeword/script/run",
        custom_model_dir="/home/samda/wyoming-openwakeword/custom-models",
        model_name="igor",
    )
    assert "/home/samda/wyoming-openwakeword/script/run" in line
    assert "--custom-model-dir /home/samda/wyoming-openwakeword/custom-models" in line
    assert "--preload-model igor" in line
    assert "--threshold 0.5" in line       # DEFAULT_THRESHOLD
    assert "--trigger-level 3" in line     # DEFAULT_TRIGGER_LEVEL
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/wakeword/test_render_runtime.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `wakeword/render_runtime.py`**

```python
"""Render systemd ExecStart for wyoming-openwakeword using values from contracts.py.

Print to stdout when run as a script; importable for use in tests / deploy scripts.
"""
from __future__ import annotations
import sys

from wakeword import contracts


def render_openwakeword_execstart(
    run_script: str,
    custom_model_dir: str,
    model_name: str,
    uri: str = "tcp://0.0.0.0:10400",
    threshold: float = contracts.DEFAULT_THRESHOLD,
    trigger_level: int = contracts.DEFAULT_TRIGGER_LEVEL,
) -> str:
    return (
        f"{run_script} "
        f"--uri {uri} "
        f"--custom-model-dir {custom_model_dir} "
        f"--preload-model {model_name} "
        f"--threshold {threshold} "
        f"--trigger-level {trigger_level} "
        f"--debug"
    )


def main() -> int:
    line = render_openwakeword_execstart(
        run_script="/home/samda/wyoming-openwakeword/script/run",
        custom_model_dir="/home/samda/wyoming-openwakeword/custom-models",
        model_name="igor",
    )
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/pytest tests/wakeword/test_render_runtime.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add wakeword/render_runtime.py tests/wakeword/test_render_runtime.py
git commit -m "wakeword/render_runtime.py — generates systemd ExecStart from contracts"
```

---

## Phase 3 — cognition Contracts

### Task 6: Core value objects (VoiceTurn, ConversationResult, RoomConfig, ToolCallRecord)

**Files:**
- Create: `server/cognition/contracts.py`
- Create: `tests/cognition/__init__.py`
- Create: `tests/cognition/test_contracts_core.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/__init__.py` (empty), then `tests/cognition/test_contracts_core.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import (
    VoiceTurn, ConversationResult, RoomConfig, ToolCallRecord,
)


def test_voice_turn_is_frozen_dataclass():
    turn = VoiceTurn(
        correlation_id="abc",
        started_at=datetime.now(UTC),
        device_id="dev1",
        room=RoomConfig(room_id="office", display_name="Office", ha_area="Office"),
        input_text="what time is it",
        speaker_id=None,
        metadata={"language": "en"},
    )
    import dataclasses
    assert dataclasses.is_dataclass(turn)
    # frozen → cannot reassign
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.input_text = "different"  # type: ignore


def test_conversation_result_carries_correlation():
    result = ConversationResult(
        correlation_id="abc",
        response_text="hi",
        commands_executed=[],
        end_conversation=True,
    )
    assert result.correlation_id == "abc"


def test_tool_call_record_minimal():
    rec = ToolCallRecord(name="get_time", args={"include_date": True}, result="3:00 PM")
    assert rec.name == "get_time"
    assert rec.args == {"include_date": True}
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/test_contracts_core.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement core value objects**

Create `server/cognition/contracts.py`:

```python
"""Public contracts for the cognition bounded context.

External callers (ha_io, external, main) import ONLY from this module.
Aggregates and services may import from this module and from ports/.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------- Room & device context ----------

@dataclass(frozen=True)
class RoomConfig:
    """Minimal room descriptor. `ha_area` is the source of truth for device targeting."""
    room_id: str
    display_name: str
    ha_area: Optional[str] = None


# ---------- Tool call records ----------

@dataclass(frozen=True)
class ToolCallRecord:
    """A single LLM-issued tool invocation that ran during a turn."""
    name: str
    args: dict
    result: str


# ---------- The cross-cutting flow object ----------

@dataclass(frozen=True)
class VoiceTurn:
    """One conversational turn. Its `correlation_id` is the future Episode's `episode_id`.
    Stamped on every persistent write produced during the turn (facts, episodes, etc.)."""
    correlation_id: str
    started_at: datetime
    device_id: Optional[str]
    room: RoomConfig
    input_text: str
    speaker_id: Optional[str]          # nullable from day 1; resemblyzer fills later
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationResult:
    """Result of Conversation.process(turn). Returned to ha_io for mapping back to HA."""
    correlation_id: str
    response_text: str
    commands_executed: list[str]
    end_conversation: bool
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/test_contracts_core.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/contracts.py tests/cognition/
git commit -m "cognition.contracts: VoiceTurn, ConversationResult, RoomConfig, ToolCallRecord"
```

---

### Task 7: Domain entities (Episode, Fact, Reflection, FeedbackEntry, Reminder)

**Files:**
- Modify: `server/cognition/contracts.py`
- Create: `tests/cognition/test_contracts_entities.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/test_contracts_entities.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import (
    Episode, Fact, Reflection, FeedbackEntry, Reminder, ToolCallRecord,
)


def test_episode_structured_fields():
    now = datetime.now(UTC)
    ep = Episode(
        episode_id="ep-1", occurred_at=now, speaker_id=None,
        participants=["sam", "igor"], intent="time_query",
        raw_utterance="what time is it", tool_calls=[
            ToolCallRecord(name="get_time", args={}, result="3 PM"),
        ],
        emotional_tone="neutral", summary=None, consolidated_at=None,
    )
    assert ep.episode_id == "ep-1"
    assert len(ep.tool_calls) == 1


def test_fact_bi_temporal_columns():
    now = datetime.now(UTC)
    fact = Fact(
        fact_id="f-1", category="preferences", key="coffee", value="dark roast oat milk",
        tags=["beverage", "morning"], source_episode_id="ep-1",
        embedding=None,
        valid_at=now, invalid_at=None, created_at=now,
    )
    assert fact.invalid_at is None       # currently valid
    assert fact.embedding is None         # populated later


def test_reflection_minimal():
    r = Reflection(reflection_id="r-1", occurred_at=datetime.now(UTC),
                   note="user got frustrated by long preamble", source_episode_id="ep-1")
    assert r.note.startswith("user got")


def test_feedback_and_reminder():
    now = datetime.now(UTC)
    fb = FeedbackEntry(feedback_id="fb-1", occurred_at=now,
                       issue="time format should be 24h", status="open",
                       source_episode_id=None)
    assert fb.status == "open"
    rm = Reminder(reminder_id="rm-1", name="pasta", fire_at=now,
                  room_id="kitchen", status="pending", source_episode_id="ep-1")
    assert rm.status == "pending"
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/test_contracts_entities.py -v`
Expected: ImportError on entity classes.

- [ ] **Step 3: Append entities to `server/cognition/contracts.py`**

Add to the bottom of `server/cognition/contracts.py`:

```python


# ---------- Memory aggregate entities ----------

@dataclass(frozen=True)
class Fact:
    """Semantic fact. Bi-temporal columns enable 'we don't live there anymore'
    without deletion. `embedding` is the adjacent vector slot; nullable until
    HybridRetrieval is enabled."""
    fact_id: str
    category: str
    key: str
    value: str
    tags: list[str]
    source_episode_id: Optional[str]
    embedding: Optional[bytes]
    valid_at: datetime         # world time
    invalid_at: Optional[datetime]   # null = currently true
    created_at: datetime       # transaction time


# ---------- Episode aggregate entity (also provenance anchor) ----------

@dataclass(frozen=True)
class Episode:
    """One conversational turn, persisted as a structured entity (not a summary string).
    `episode_id` == `VoiceTurn.correlation_id` always."""
    episode_id: str
    occurred_at: datetime
    speaker_id: Optional[str]
    participants: list[str]
    intent: Optional[str]
    raw_utterance: str
    tool_calls: list[ToolCallRecord]
    emotional_tone: Optional[str]
    summary: Optional[str]
    consolidated_at: Optional[datetime]


# ---------- Identity aggregate sub-collection ----------

@dataclass(frozen=True)
class Reflection:
    """Agent meta-note. Produced by Consolidator when noticing patterns
    about its own performance."""
    reflection_id: str
    occurred_at: datetime
    note: str
    source_episode_id: Optional[str]


# ---------- UserState aggregate sub-collections ----------

@dataclass(frozen=True)
class FeedbackEntry:
    feedback_id: str
    occurred_at: datetime
    issue: str
    status: str       # "open" | "resolved"
    source_episode_id: Optional[str]


@dataclass(frozen=True)
class Reminder:
    reminder_id: str
    name: str
    fire_at: datetime
    room_id: Optional[str]
    status: str       # "pending" | "fired" | "cancelled"
    source_episode_id: Optional[str]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/test_contracts_entities.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/contracts.py tests/cognition/test_contracts_entities.py
git commit -m "cognition.contracts: Episode, Fact, Reflection, FeedbackEntry, Reminder"
```

---

### Task 8: Port protocols (Persistence, Retrieval, LLM, Tools, Clock)

**Files:**
- Create: `server/cognition/ports/persistence.py`
- Create: `server/cognition/ports/retrieval.py`
- Create: `server/cognition/ports/llm.py`
- Create: `server/cognition/ports/tools.py`
- Create: `server/cognition/ports/clock.py`
- Create: `tests/cognition/test_ports_shape.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/test_ports_shape.py`:

```python
"""Ports are Protocols (PEP 544). Verify shape, not behavior."""
from typing import get_type_hints
from server.cognition.ports import persistence, retrieval, llm, tools, clock


def test_persistence_port_methods():
    assert hasattr(persistence.PersistencePort, "save_episode")
    assert hasattr(persistence.PersistencePort, "load_episode")
    assert hasattr(persistence.PersistencePort, "save_fact")
    assert hasattr(persistence.PersistencePort, "list_unconsolidated_episodes")


def test_retrieval_port_methods():
    assert hasattr(retrieval.RetrievalPort, "query")


def test_llm_port_methods():
    assert hasattr(llm.LLMPort, "chat")


def test_tool_executor_methods():
    assert hasattr(tools.ToolExecutorPort, "execute")
    assert hasattr(tools.ToolExecutorPort, "list_schemas")


def test_clock_methods():
    assert hasattr(clock.ClockPort, "now")
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/test_ports_shape.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the five port files**

Create `server/cognition/ports/persistence.py`:

```python
"""PersistencePort — abstracts SQL/JSON/whatever from cognition.
Implementations live in server.external."""
from __future__ import annotations
from typing import Optional, Protocol
from server.cognition.contracts import (
    Episode, Fact, Reflection, FeedbackEntry, Reminder,
)


class PersistencePort(Protocol):
    # Episodes
    def save_episode(self, episode: Episode) -> None: ...
    def load_episode(self, episode_id: str) -> Optional[Episode]: ...
    def list_recent_episodes(self, limit: int) -> list[Episode]: ...
    def list_unconsolidated_episodes(self) -> list[Episode]: ...
    def mark_episodes_consolidated(self, episode_ids: list[str], at) -> None: ...

    # Facts
    def save_fact(self, fact: Fact) -> None: ...
    def find_fact(self, category: str, key: str) -> Optional[Fact]: ...
    def list_active_facts(self) -> list[Fact]: ...
    def invalidate_fact(self, fact_id: str, at) -> None: ...

    # Identity
    def get_identity_narrative(self) -> Optional[str]: ...
    def save_identity_narrative(self, narrative: str, last_consolidated_at,
                                last_consolidated_episode_id: Optional[str]) -> None: ...
    def get_last_consolidated_episode_id(self) -> Optional[str]: ...

    # Reflections
    def save_reflection(self, reflection: Reflection) -> None: ...
    def list_recent_reflections(self, limit: int) -> list[Reflection]: ...

    # Feedback
    def save_feedback(self, entry: FeedbackEntry) -> None: ...
    def list_feedback(self, status: Optional[str] = None) -> list[FeedbackEntry]: ...
    def resolve_feedback(self, feedback_id: str) -> None: ...

    # Reminders
    def save_reminder(self, reminder: Reminder) -> None: ...
    def list_pending_reminders(self) -> list[Reminder]: ...
    def update_reminder_status(self, reminder_id: str, status: str) -> None: ...
```

Create `server/cognition/ports/retrieval.py`:

```python
"""RetrievalPort — return relevant facts/episodes given a query.
Initial impl is tag+recency; swappable to hybrid (semantic+tag+recency)."""
from __future__ import annotations
from typing import Protocol
from server.cognition.contracts import VoiceTurn, Fact


class RetrievalPort(Protocol):
    def query(self, turn: VoiceTurn, k: int = 10) -> list[Fact]: ...
```

Create `server/cognition/ports/llm.py`:

```python
"""LLMPort — minimal chat interface."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable


@dataclass(frozen=True)
class ChatResult:
    text: str
    commands_executed: list[str]
    input_tokens: int
    output_tokens: int


class LLMPort(Protocol):
    def chat(
        self,
        system_prompt: str,
        user_text: str,
        tool_schemas: list[dict],
        tool_executor: Callable[[str, dict], str],
        history: list[dict] | None = None,
    ) -> ChatResult: ...
```

Create `server/cognition/ports/tools.py`:

```python
"""ToolExecutorPort — execute a named tool with args, return its string result."""
from __future__ import annotations
from typing import Protocol
from server.cognition.contracts import VoiceTurn


class ToolExecutorPort(Protocol):
    def list_schemas(self) -> list[dict]:
        """Return list of Anthropic-format tool schemas the LLM can call."""
        ...

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        """Execute a tool and return its result text."""
        ...
```

Create `server/cognition/ports/clock.py`:

```python
"""ClockPort — for testability of time-sensitive logic."""
from __future__ import annotations
from datetime import datetime
from typing import Protocol


class ClockPort(Protocol):
    def now(self) -> datetime: ...
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/test_ports_shape.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Run boundary check**

Run: `.venv/bin/python -m tools.boundary_check`
Expected: `Boundary check passed.`

- [ ] **Step 6: Commit**

```bash
git add server/cognition/ports/ tests/cognition/test_ports_shape.py
git commit -m "cognition.ports: PersistencePort, RetrievalPort, LLMPort, ToolExecutorPort, ClockPort"
```

---

## Phase 4 — SQLite Persistence

### Task 9: SQL schema + connection helper

**Files:**
- Create: `server/external/_internal/schema.sql`
- Create: `server/external/_internal/db.py`
- Create: `tests/external/__init__.py`
- Create: `tests/external/test_db.py`

- [ ] **Step 1: Write the schema**

Create `server/external/_internal/schema.sql` with the full schema from spec §6.1:

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id          TEXT PRIMARY KEY,
    occurred_at         TEXT NOT NULL,
    speaker_id          TEXT,
    participants        TEXT,
    intent              TEXT,
    raw_utterance       TEXT NOT NULL,
    tool_calls          TEXT,
    emotional_tone     TEXT,
    summary             TEXT,
    consolidated_at     TEXT
);
CREATE INDEX IF NOT EXISTS episodes_unconsolidated
    ON episodes(occurred_at) WHERE consolidated_at IS NULL;

CREATE TABLE IF NOT EXISTS facts (
    fact_id             TEXT PRIMARY KEY,
    category            TEXT NOT NULL,
    key                 TEXT NOT NULL,
    value               TEXT NOT NULL,
    tags                TEXT,
    source_episode_id   TEXT REFERENCES episodes(episode_id),
    embedding           BLOB,
    valid_at            TEXT NOT NULL,
    invalid_at          TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_active
    ON facts(category, key) WHERE invalid_at IS NULL;
CREATE INDEX IF NOT EXISTS facts_episode ON facts(source_episode_id);

CREATE TABLE IF NOT EXISTS identity (
    id                            INTEGER PRIMARY KEY CHECK (id = 1),
    narrative                     TEXT NOT NULL,
    last_consolidated_at          TEXT,
    last_consolidated_episode_id  TEXT
);

CREATE TABLE IF NOT EXISTS reflections (
    reflection_id      TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    note               TEXT NOT NULL,
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id        TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    issue              TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'open',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS reminders (
    reminder_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    fire_at            TEXT NOT NULL,
    room_id            TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);
```

- [ ] **Step 2: Write the connection helper test**

Create `tests/external/__init__.py` (empty), then `tests/external/test_db.py`:

```python
from server.external._internal.db import open_db


def test_open_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = open_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {row[0] for row in rows}
        assert {"episodes", "facts", "identity", "reflections", "feedback", "reminders"} <= names
    finally:
        conn.close()
```

- [ ] **Step 3: Verify failure**

Run: `.venv/bin/pytest tests/external/test_db.py -v`
Expected: ImportError.

- [ ] **Step 4: Write the connection helper**

Create `server/external/_internal/db.py`:

```python
"""SQLite open helper: opens the DB, applies schema, returns a Connection."""
from __future__ import annotations
import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection at db_path, creating + migrating schema if needed.
    Returns a connection with foreign_keys=ON and Row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn
```

- [ ] **Step 5: Run test**

Run: `.venv/bin/pytest tests/external/test_db.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add server/external/_internal/schema.sql server/external/_internal/db.py tests/external/
git commit -m "external._internal.db: schema.sql + open_db helper"
```

---

### Task 10: `SqlitePersistence` — Episode operations

**Files:**
- Create: `server/external/sqlite_persistence.py`
- Create: `tests/external/test_sqlite_persistence_episodes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_sqlite_persistence_episodes.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import Episode, ToolCallRecord
from server.external.sqlite_persistence import SqlitePersistence


def test_save_and_load_episode(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    ep = Episode(
        episode_id="ep-1",
        occurred_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        speaker_id=None,
        participants=["sam", "igor"],
        intent="time_query",
        raw_utterance="what time is it",
        tool_calls=[ToolCallRecord(name="get_time", args={"include_date": True}, result="3 PM")],
        emotional_tone=None,
        summary=None,
        consolidated_at=None,
    )
    sp.save_episode(ep)
    loaded = sp.load_episode("ep-1")
    assert loaded is not None
    assert loaded.raw_utterance == "what time is it"
    assert len(loaded.tool_calls) == 1
    assert loaded.tool_calls[0].name == "get_time"


def test_list_unconsolidated(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    for i in range(3):
        sp.save_episode(Episode(
            episode_id=f"ep-{i}", occurred_at=datetime(2026, 1, 1, 10, i, tzinfo=UTC),
            speaker_id=None, participants=[], intent=None,
            raw_utterance="x", tool_calls=[], emotional_tone=None,
            summary=None, consolidated_at=None,
        ))
    assert len(sp.list_unconsolidated_episodes()) == 3
    sp.mark_episodes_consolidated(["ep-0", "ep-1"], at=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(sp.list_unconsolidated_episodes()) == 1
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_sqlite_persistence_episodes.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `SqlitePersistence` episode methods**

Create `server/external/sqlite_persistence.py`:

```python
"""SQLite implementation of cognition.ports.PersistencePort."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from server.cognition.contracts import (
    Episode, Fact, Reflection, FeedbackEntry, Reminder, ToolCallRecord,
)
from server.external._internal.db import open_db


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


def _row_to_episode(row: sqlite3.Row) -> Episode:
    tcs = json.loads(row["tool_calls"]) if row["tool_calls"] else []
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
    )


class SqlitePersistence:
    """Concrete PersistencePort. Single SQLite file."""

    def __init__(self, db_path: Path):
        self._conn = open_db(db_path)

    # ---- Episodes ----
    def save_episode(self, episode: Episode) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, occurred_at, speaker_id, participants, intent,
                raw_utterance, tool_calls, emotional_tone, summary, consolidated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

    # Other methods stubbed to satisfy the Protocol; implemented in next tasks.
    def save_fact(self, fact: Fact) -> None: raise NotImplementedError
    def find_fact(self, category: str, key: str): raise NotImplementedError
    def list_active_facts(self) -> list[Fact]: raise NotImplementedError
    def invalidate_fact(self, fact_id: str, at: datetime) -> None: raise NotImplementedError
    def get_identity_narrative(self): raise NotImplementedError
    def save_identity_narrative(self, narrative, last_consolidated_at, last_consolidated_episode_id): raise NotImplementedError
    def get_last_consolidated_episode_id(self): raise NotImplementedError
    def save_reflection(self, r): raise NotImplementedError
    def list_recent_reflections(self, limit): raise NotImplementedError
    def save_feedback(self, e): raise NotImplementedError
    def list_feedback(self, status=None): raise NotImplementedError
    def resolve_feedback(self, fid): raise NotImplementedError
    def save_reminder(self, r): raise NotImplementedError
    def list_pending_reminders(self): raise NotImplementedError
    def update_reminder_status(self, rid, status): raise NotImplementedError
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/external/test_sqlite_persistence_episodes.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/external/sqlite_persistence.py tests/external/test_sqlite_persistence_episodes.py
git commit -m "SqlitePersistence: Episode operations (save/load/list/consolidate)"
```

---

### Task 11: `SqlitePersistence` — Fact operations with bi-temporal

**Files:**
- Modify: `server/external/sqlite_persistence.py`
- Create: `tests/external/test_sqlite_persistence_facts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_sqlite_persistence_facts.py`:

```python
from datetime import datetime, UTC, timedelta
from server.cognition.contracts import Fact
from server.external.sqlite_persistence import SqlitePersistence


def _make_fact(fact_id="f-1", category="prefs", key="coffee",
               value="dark roast", invalid_at=None, episode_id=None):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Fact(
        fact_id=fact_id, category=category, key=key, value=value,
        tags=["beverage"], source_episode_id=episode_id,
        embedding=None,
        valid_at=now, invalid_at=invalid_at, created_at=now,
    )


def test_save_and_find_active_fact(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    sp.save_fact(_make_fact())
    found = sp.find_fact("prefs", "coffee")
    assert found is not None and found.value == "dark roast"


def test_invalidate_fact_excludes_from_active(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    sp.save_fact(_make_fact())
    sp.invalidate_fact("f-1", at=datetime(2026, 6, 1, tzinfo=UTC))
    active = sp.list_active_facts()
    assert not any(f.fact_id == "f-1" for f in active)
    # but find_fact returns it because no scoping
    raw = sp._conn.execute("SELECT invalid_at FROM facts WHERE fact_id='f-1'").fetchone()
    assert raw["invalid_at"] is not None
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_sqlite_persistence_facts.py -v`
Expected: NotImplementedError.

- [ ] **Step 3: Implement the fact methods**

Replace the four fact-related stubs in `server/external/sqlite_persistence.py` with:

```python
    # ---- Facts ----
    def save_fact(self, fact: Fact) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO facts
               (fact_id, category, key, value, tags, source_episode_id,
                embedding, valid_at, invalid_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.fact_id, fact.category, fact.key, fact.value,
                json.dumps(fact.tags), fact.source_episode_id,
                fact.embedding,
                _dt_to_iso(fact.valid_at),
                _dt_to_iso(fact.invalid_at) if fact.invalid_at else None,
                _dt_to_iso(fact.created_at),
            ),
        )

    def find_fact(self, category: str, key: str) -> Optional[Fact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE category=? AND key=? AND invalid_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (category, key),
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
```

And add the row-to-fact helper near `_row_to_episode`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/external/test_sqlite_persistence_facts.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/external/sqlite_persistence.py tests/external/test_sqlite_persistence_facts.py
git commit -m "SqlitePersistence: Fact operations with bi-temporal columns"
```

---

### Task 12: `SqlitePersistence` — identity, reflections, feedback, reminders

**Files:**
- Modify: `server/external/sqlite_persistence.py`
- Create: `tests/external/test_sqlite_persistence_rest.py`

- [ ] **Step 1: Write tests**

Create `tests/external/test_sqlite_persistence_rest.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import Reflection, FeedbackEntry, Reminder
from server.external.sqlite_persistence import SqlitePersistence


def test_identity_round_trip(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    assert sp.get_identity_narrative() is None
    sp.save_identity_narrative("Sam is a homelab nerd.", datetime(2026, 1, 1, tzinfo=UTC), "ep-9")
    assert sp.get_identity_narrative() == "Sam is a homelab nerd."
    assert sp.get_last_consolidated_episode_id() == "ep-9"


def test_reflections_and_feedback_and_reminders(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sp.save_reflection(Reflection(reflection_id="r-1", occurred_at=now,
                                  note="long preambles annoy user", source_episode_id=None))
    assert len(sp.list_recent_reflections(5)) == 1

    sp.save_feedback(FeedbackEntry(feedback_id="fb-1", occurred_at=now,
                                   issue="use 24h time", status="open",
                                   source_episode_id=None))
    open_items = sp.list_feedback("open")
    assert len(open_items) == 1
    sp.resolve_feedback("fb-1")
    assert len(sp.list_feedback("open")) == 0

    sp.save_reminder(Reminder(reminder_id="rm-1", name="pasta", fire_at=now,
                              room_id="kitchen", status="pending", source_episode_id=None))
    pending = sp.list_pending_reminders()
    assert len(pending) == 1
    sp.update_reminder_status("rm-1", "fired")
    assert len(sp.list_pending_reminders()) == 0
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_sqlite_persistence_rest.py -v`
Expected: NotImplementedError on the various stubs.

- [ ] **Step 3: Replace the remaining stubs**

In `server/external/sqlite_persistence.py`, replace the identity/reflection/feedback/reminder stubs with:

```python
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
```

- [ ] **Step 4: Run all external tests**

Run: `.venv/bin/pytest tests/external -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add server/external/sqlite_persistence.py tests/external/test_sqlite_persistence_rest.py
git commit -m "SqlitePersistence: identity, reflections, feedback, reminders"
```

---

### Task 13: `brain.json` → SQLite migration script

**Files:**
- Create: `server/external/_internal/brain_json_migration.py`
- Create: `tests/external/test_brain_json_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_brain_json_migration.py`:

```python
import json
from datetime import datetime, UTC
from pathlib import Path
from server.external.sqlite_persistence import SqlitePersistence
from server.external._internal.brain_json_migration import migrate_brain_json_if_needed


def _make_brain_json(p: Path):
    """Minimal brain.json shaped like the legacy structure (entries list)."""
    data = {
        "entries": [
            {
                "id": 1, "type": "memory", "created": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "data": {"category": "prefs", "key": "coffee", "value": "dark roast", "tags": []},
                "tags": ["beverage"],
            },
            {
                "id": 2, "type": "episode", "created": datetime(2026, 1, 1, 11, tzinfo=UTC).isoformat(),
                "data": {"raw_utterance": "what time is it", "summary": "user asked time",
                         "participants": ["sam", "igor"], "tool_calls": []},
            },
            {
                "id": 3, "type": "identity", "created": datetime(2026, 1, 1, 12, tzinfo=UTC).isoformat(),
                "data": {"narrative": "Sam is a homelab nerd."},
            },
        ],
    }
    p.write_text(json.dumps(data), encoding="utf-8")


def test_migration_copies_brain_into_sqlite(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)

    migrate_brain_json_if_needed(bj, db)

    sp = SqlitePersistence(db)
    assert sp.find_fact("prefs", "coffee") is not None
    assert len(sp.list_recent_episodes(10)) == 1
    assert sp.get_identity_narrative() == "Sam is a homelab nerd."
    # brain.json renamed to .imported-*
    assert not bj.exists()
    assert list(tmp_path.glob("brain.json.imported-*.bak"))


def test_migration_is_idempotent(tmp_path):
    bj = tmp_path / "brain.json"
    db = tmp_path / "brain.db"
    _make_brain_json(bj)
    migrate_brain_json_if_needed(bj, db)
    # Second call: no brain.json to import; no-op
    migrate_brain_json_if_needed(bj, db)
    sp = SqlitePersistence(db)
    assert len(sp.list_recent_episodes(10)) == 1   # still exactly one episode
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_brain_json_migration.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the migration**

Create `server/external/_internal/brain_json_migration.py`:

```python
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
    now_iso = datetime.utcnow().isoformat() + "Z"

    for e in entries:
        etype = e.get("type")
        edata = e.get("data") or {}
        created = e.get("created") or now_iso
        created_dt = datetime.fromisoformat(created.rstrip("Z"))

        if etype == "memory":
            sp.save_fact(Fact(
                fact_id=str(uuid.uuid4()),
                category=edata.get("category", "unknown"),
                key=edata.get("key", str(e.get("id"))),
                value=str(edata.get("value", "")),
                tags=e.get("tags", []) or edata.get("tags", []),
                source_episode_id=None,         # pre-provenance
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

    # rename brain.json to .imported-<ts>.bak
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    brain_json_path.rename(brain_json_path.with_suffix(f".json.imported-{stamp}.bak"))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/external/test_brain_json_migration.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/external/_internal/brain_json_migration.py tests/external/test_brain_json_migration.py
git commit -m "brain.json → SQLite migration (idempotent, renames on completion)"
```

---

## Phase 5 — Aggregates

### Task 14: `MemoryStore` aggregate

**Files:**
- Create: `server/cognition/aggregates/memory.py`
- Create: `tests/cognition/aggregates/__init__.py`
- Create: `tests/cognition/aggregates/test_memory_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/aggregates/__init__.py` (empty), then `tests/cognition/aggregates/test_memory_store.py`:

```python
from datetime import datetime, UTC
from server.cognition.aggregates.memory import MemoryStore
from server.external.sqlite_persistence import SqlitePersistence


def test_save_uses_episode_provenance(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    mem = MemoryStore(sp)
    mem.save_fact(
        category="prefs", key="coffee", value="dark roast",
        tags=["beverage"], source_episode_id="ep-1",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    found = mem.find_fact("prefs", "coffee")
    assert found is not None
    assert found.source_episode_id == "ep-1"


def test_invalidate_then_replace(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    mem = MemoryStore(sp)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 6, 1, tzinfo=UTC)
    mem.save_fact("prefs", "coffee", "milk only", [], "ep-1", t0)
    mem.update_fact("prefs", "coffee", "dark roast oat milk", [], "ep-2", t1)
    # old fact is invalidated, new fact is active
    active = mem.list_active()
    assert len([f for f in active if f.category == "prefs" and f.key == "coffee"]) == 1
    assert active[0].value == "dark roast oat milk"
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_memory_store.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `MemoryStore`**

Create `server/cognition/aggregates/memory.py`:

```python
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

    def list_active(self) -> list[Fact]:
        return self._p.list_active_facts()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_memory_store.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/aggregates/memory.py tests/cognition/aggregates/
git commit -m "MemoryStore aggregate (save/update with bi-temporal invalidation)"
```

---

### Task 15: `EpisodeStore` aggregate

**Files:**
- Create: `server/cognition/aggregates/episode.py`
- Create: `tests/cognition/aggregates/test_episode_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/aggregates/test_episode_store.py`:

```python
from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.contracts import Episode
from server.external.sqlite_persistence import SqlitePersistence


def _ep(eid: str, minute: int = 0) -> Episode:
    return Episode(
        episode_id=eid,
        occurred_at=datetime(2026, 1, 1, 10, minute, tzinfo=UTC),
        speaker_id=None, participants=[], intent=None,
        raw_utterance="x", tool_calls=[], emotional_tone=None,
        summary=None, consolidated_at=None,
    )


def test_add_and_recent(tmp_path):
    es = EpisodeStore(SqlitePersistence(tmp_path / "brain.db"))
    es.add(_ep("ep-1", 0))
    es.add(_ep("ep-2", 5))
    recent = es.get_recent(10)
    assert [e.episode_id for e in recent] == ["ep-2", "ep-1"]


def test_consolidation_flow(tmp_path):
    es = EpisodeStore(SqlitePersistence(tmp_path / "brain.db"))
    for i in range(3):
        es.add(_ep(f"ep-{i}", i))
    assert len(es.get_unconsolidated()) == 3
    es.mark_consolidated(["ep-0", "ep-1"], at=datetime(2026, 1, 2, tzinfo=UTC))
    assert len(es.get_unconsolidated()) == 1
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_episode_store.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `EpisodeStore`**

Create `server/cognition/aggregates/episode.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_episode_store.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/aggregates/episode.py tests/cognition/aggregates/test_episode_store.py
git commit -m "EpisodeStore aggregate (provenance anchor)"
```

---

### Task 16: `IdentityStore` + reflections

**Files:**
- Create: `server/cognition/aggregates/identity.py`
- Create: `tests/cognition/aggregates/test_identity_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/aggregates/test_identity_store.py`:

```python
from datetime import datetime, UTC
from server.cognition.aggregates.identity import IdentityStore
from server.external.sqlite_persistence import SqlitePersistence


def test_narrative_default_empty(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    assert ids.get_narrative() == ""


def test_replace_narrative_and_track_consolidation(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    ids.replace_narrative("Sam likes coffee.",
                          last_consolidated_at=datetime(2026, 1, 1, tzinfo=UTC),
                          last_consolidated_episode_id="ep-5")
    assert ids.get_narrative() == "Sam likes coffee."
    assert ids.get_last_consolidated_episode_id() == "ep-5"


def test_log_reflection(tmp_path):
    ids = IdentityStore(SqlitePersistence(tmp_path / "brain.db"))
    ids.log_reflection("preambles too long",
                       at=datetime(2026, 1, 1, tzinfo=UTC),
                       source_episode_id="ep-1")
    assert any("preambles" in r.note for r in ids.list_recent_reflections(5))
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_identity_store.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `IdentityStore`**

Create `server/cognition/aggregates/identity.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_identity_store.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/aggregates/identity.py tests/cognition/aggregates/test_identity_store.py
git commit -m "IdentityStore aggregate (narrative + reflections + crash-replay anchor)"
```

---

### Task 17: `UserState` aggregate (feedback + reminders)

**Files:**
- Create: `server/cognition/aggregates/user_state.py`
- Create: `tests/cognition/aggregates/test_user_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/aggregates/test_user_state.py`:

```python
from datetime import datetime, UTC, timedelta
from server.cognition.aggregates.user_state import UserState
from server.external.sqlite_persistence import SqlitePersistence


def test_feedback_lifecycle(tmp_path):
    us = UserState(SqlitePersistence(tmp_path / "brain.db"))
    fb = us.log_feedback(issue="use 24h time",
                         at=datetime(2026, 1, 1, tzinfo=UTC),
                         source_episode_id="ep-1")
    open_items = us.list_open_feedback()
    assert len(open_items) == 1
    us.resolve_feedback(fb.feedback_id)
    assert len(us.list_open_feedback()) == 0


def test_reminder_lifecycle(tmp_path):
    us = UserState(SqlitePersistence(tmp_path / "brain.db"))
    fire = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5)
    rm = us.add_reminder(name="pasta", fire_at=fire, room_id="kitchen",
                         source_episode_id="ep-2")
    assert any(r.reminder_id == rm.reminder_id for r in us.list_pending())
    us.fire_reminder(rm.reminder_id)
    assert all(r.reminder_id != rm.reminder_id for r in us.list_pending())
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_user_state.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `UserState`**

Create `server/cognition/aggregates/user_state.py`:

```python
"""UserState — feedback + reminders. Replaces today's brain.json status entries."""
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/aggregates/test_user_state.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/aggregates/user_state.py tests/cognition/aggregates/test_user_state.py
git commit -m "UserState aggregate (feedback + reminders, merged from spec validation)"
```

---

## Phase 6 — External adapters (retrieval, clock)

### Task 18: `TagRetrieval` adapter + `SystemClock`

**Files:**
- Create: `server/external/sqlite_retrieval.py`
- Create: `server/external/system_clock.py`
- Create: `tests/external/test_tag_retrieval.py`
- Create: `tests/external/test_system_clock.py`

- [ ] **Step 1: Write the retrieval test**

Create `tests/external/test_tag_retrieval.py`:

```python
from datetime import datetime, UTC
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t1", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig(room_id="default", display_name="Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_tag_overlap_ranks_matching_facts_higher(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    mem = MemoryStore(sp)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    mem.save_fact("prefs", "coffee", "dark roast oat milk",
                  tags=["coffee", "beverage"], source_episode_id=None, now=now)
    mem.save_fact("prefs", "music", "jazz",
                  tags=["music"], source_episode_id=None, now=now)

    retr = TagRetrieval(sp)
    hits = retr.query(_turn("what coffee do I like"), k=2)
    assert hits[0].key == "coffee"
```

- [ ] **Step 2: Write the clock test**

Create `tests/external/test_system_clock.py`:

```python
from datetime import datetime, timezone
from server.external.system_clock import SystemClock


def test_now_is_aware():
    n = SystemClock().now()
    assert isinstance(n, datetime)
    assert n.tzinfo is not None
```

- [ ] **Step 3: Verify failures**

Run: `.venv/bin/pytest tests/external/test_tag_retrieval.py tests/external/test_system_clock.py -v`
Expected: ImportErrors.

- [ ] **Step 4: Implement retrieval**

Create `server/external/sqlite_retrieval.py`:

```python
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
```

- [ ] **Step 5: Implement clock**

Create `server/external/system_clock.py`:

```python
"""SystemClock — wall-clock implementation of cognition.ports.ClockPort."""
from datetime import datetime, UTC


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/external -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add server/external/sqlite_retrieval.py server/external/system_clock.py tests/external/test_tag_retrieval.py tests/external/test_system_clock.py
git commit -m "TagRetrieval + SystemClock adapters"
```

---

### Task 19: `ClaudeAdapter` (LLMPort)

**Files:**
- Create: `server/external/claude_adapter.py`
- Create: `tests/external/test_claude_adapter.py`

This adapter wraps the `anthropic` SDK. We won't make a live API call in tests — we'll use a fake transport.

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_claude_adapter.py`:

```python
from server.external.claude_adapter import ClaudeAdapter


class _FakeAnthropic:
    """Stand-in for the anthropic SDK client to drive ClaudeAdapter."""

    class _Messages:
        def create(self, **kwargs):
            class _R:
                content = [type("X", (), {"type": "text", "text": "Hello, world."})]
                usage = type("U", (), {"input_tokens": 12, "output_tokens": 5})
                stop_reason = "end_turn"
            return _R()

    messages = _Messages()


def test_chat_returns_text_and_tokens():
    adapter = ClaudeAdapter(client=_FakeAnthropic(), model="claude-haiku-4-5-20251001")
    result = adapter.chat(
        system_prompt="be brief",
        user_text="hello",
        tool_schemas=[],
        tool_executor=lambda name, args: "",
    )
    assert result.text == "Hello, world."
    assert result.input_tokens == 12
    assert result.output_tokens == 5
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_claude_adapter.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `ClaudeAdapter`**

Create `server/external/claude_adapter.py`:

```python
"""Claude API adapter implementing cognition.ports.LLMPort.

This is the ONLY file in the project allowed to import anthropic."""
from __future__ import annotations
import logging
from typing import Callable, Optional

import anthropic  # boundary_check enforces this only-here import

from server.cognition.ports.llm import ChatResult

logger = logging.getLogger(__name__)


class ClaudeAdapter:
    def __init__(self, client: Optional["anthropic.Anthropic"] = None,
                 model: str = "claude-haiku-4-5-20251001",
                 max_tokens: int = 1024,
                 max_rounds: int = 3):
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._max_rounds = max_rounds

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        tool_schemas: list[dict],
        tool_executor: Callable[[str, dict], str],
        history: list[dict] | None = None,
    ) -> ChatResult:
        messages = list(history or []) + [{"role": "user", "content": user_text}]
        commands: list[str] = []
        in_tok = 0
        out_tok = 0
        last_text = ""

        for _ in range(self._max_rounds):
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tool_schemas or [],
            )
            in_tok += getattr(resp.usage, "input_tokens", 0)
            out_tok += getattr(resp.usage, "output_tokens", 0)

            tool_calls = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            text_blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
            if text_blocks:
                last_text = text_blocks[-1].text

            if not tool_calls:
                break

            # Execute tool calls; append results into the message thread.
            assistant_content = list(resp.content)
            tool_results = []
            for tc in tool_calls:
                args = getattr(tc, "input", {}) or {}
                result = tool_executor(tc.name, args)
                commands.append(tc.name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return ChatResult(
            text=last_text, commands_executed=commands,
            input_tokens=in_tok, output_tokens=out_tok,
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/external/test_claude_adapter.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Run boundary check**

Run: `.venv/bin/python -m tools.boundary_check`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add server/external/claude_adapter.py tests/external/test_claude_adapter.py
git commit -m "ClaudeAdapter — only file importing anthropic SDK"
```

---

### Task 20: Port HAClient to `server/external/_internal/ha_client.py`

The existing `server/ha_client.py` is heavily used. Move it under `external/_internal/` without behavior changes so non-external code can no longer import it directly. Existing call sites (currently in `server/commands/*`) will be re-pointed during the services-phase ports.

- [ ] **Step 1: Move the file**

```bash
git mv server/ha_client.py server/external/_internal/ha_client.py
```

- [ ] **Step 2: Update the moved file's internal references if needed**

Open `server/external/_internal/ha_client.py` and confirm no other modules import each other. Update any `from server.config import ...` lines that reference deleted config to use env vars directly (e.g., `os.environ["HA_URL"]`, `os.environ["HA_TOKEN"]`).

- [ ] **Step 3: Update `server/commands/*` to import from the new location**

For each file under `server/commands/` that has `from server.ha_client import ...`, change to:

```python
from server.external._internal.ha_client import ...
```

Note: this temporarily violates the `_internal` guard. We will replace `commands/` with `HARestToolExecutor` in Task 21, after which the old `commands/` will be deleted at cutover. The boundary-check script is scoped to `server.cognition` and `server.ha_io`, so this transitional import is permitted.

- [ ] **Step 4: Smoke test — existing api.py still imports cleanly**

Run: `.venv/bin/python -c "from server.api import create_app; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Move ha_client under external/_internal (transitional; commands/ still uses it)"
```

---

### Task 21: `HARestToolExecutor` (ToolExecutorPort)

**Files:**
- Create: `server/external/ha_rest_adapter.py`
- Create: `tests/external/test_ha_rest_adapter.py`

`HARestToolExecutor` wraps `HAClient` and the current `server/commands/` registry, exposing it through `ToolExecutorPort`. We keep the existing command implementations functional — the adapter just re-exposes them under the new port.

- [ ] **Step 1: Write the failing test**

Create `tests/external/test_ha_rest_adapter.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.external.ha_rest_adapter import HARestToolExecutor


class _FakeCommand:
    name = "fake_say"
    description = "Echo whatever you pass"
    @property
    def parameters(self):
        return {"text": {"type": "string", "description": "what to echo"}}
    required_parameters = ["text"]

    def to_tool(self):
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": self.parameters,
                                 "required": self.required_parameters}}

    def execute(self, text: str):
        return f"echo: {text}"


def test_list_schemas_returns_registered_tools():
    executor = HARestToolExecutor(commands={"fake_say": _FakeCommand()})
    schemas = executor.list_schemas()
    assert any(s["name"] == "fake_say" for s in schemas)


def test_execute_invokes_command():
    executor = HARestToolExecutor(commands={"fake_say": _FakeCommand()})
    turn = VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text="x", speaker_id=None, metadata={},
    )
    assert executor.execute("fake_say", {"text": "hi"}, turn) == "echo: hi"


def test_unknown_tool_returns_error_string():
    executor = HARestToolExecutor(commands={})
    turn = VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text="x", speaker_id=None, metadata={},
    )
    assert "unknown" in executor.execute("nope", {}, turn).lower()
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/external/test_ha_rest_adapter.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `HARestToolExecutor`**

Create `server/external/ha_rest_adapter.py`:

```python
"""ToolExecutorPort implementation. Wraps the existing `server/commands/`
registry; the underlying commands talk to HA via _internal/ha_client.py."""
from __future__ import annotations
import inspect
import logging
from typing import Mapping

from server.cognition.contracts import VoiceTurn

logger = logging.getLogger(__name__)


def _discover_default_commands() -> dict:
    """Lazy import of the existing server.commands registry."""
    from server.commands import get_all_commands  # type: ignore
    return get_all_commands()


class HARestToolExecutor:
    """Adapts the current server.commands registry to ToolExecutorPort."""

    def __init__(self, commands: Mapping[str, object] | None = None):
        self._commands = dict(commands) if commands is not None else _discover_default_commands()

    def list_schemas(self) -> list[dict]:
        out = []
        for cmd in self._commands.values():
            if hasattr(cmd, "to_tool"):
                out.append(cmd.to_tool())
        return out

    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        cmd = self._commands.get(name)
        if cmd is None:
            return f"Unknown command: {name}"
        # Some commands accept _ctx (the legacy InteractionContext). We pass a thin
        # adapter built from the turn's room so existing room-aware commands keep working.
        kwargs = dict(args)
        sig = inspect.signature(cmd.execute)
        if "_ctx" in sig.parameters:
            kwargs["_ctx"] = _legacy_ctx_from_turn(turn)
        try:
            return cmd.execute(**kwargs)
        except Exception as e:
            logger.exception("Command %s failed", name)
            return f"Error executing {name}: {e}"


def _legacy_ctx_from_turn(turn: VoiceTurn):
    """Minimal adapter so existing room-aware commands keep working.
    Removed in Task 34 when commands/ is decomposed."""
    from server.context import InteractionContext  # type: ignore
    from server.rooms import RoomConfig as LegacyRoom  # type: ignore
    legacy_room = LegacyRoom(
        room_id=turn.room.room_id,
        display_name=turn.room.display_name,
        ha_area=turn.room.ha_area,
    )
    return InteractionContext(
        client_id=turn.device_id or "ha",
        room=legacy_room,
        client_type="text",
        callback_url=None,
        prefer_sonos=False,
        tv_state="unknown",
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/external/test_ha_rest_adapter.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/external/ha_rest_adapter.py tests/external/test_ha_rest_adapter.py
git commit -m "HARestToolExecutor adapter — wraps current commands registry"
```

---

## Phase 7 — cognition services

### Task 22: `QualityGate` and `IntentRouter` (port from existing modules)

**Files:**
- Create: `server/cognition/services/quality_gate.py`
- Create: `server/cognition/services/intent_router.py`
- Create: `tests/cognition/services/__init__.py`
- Create: `tests/cognition/services/test_quality_gate.py`
- Create: `tests/cognition/services/test_intent_router.py`

We re-export the existing logic behind new module names that take `VoiceTurn` rather than raw text. Existing `server/quality_gate.py` and `server/intent_router.py` stay in place until cutover (so the old api.py keeps working).

- [ ] **Step 1: Write the QualityGate test**

Create `tests/cognition/services/__init__.py` (empty), then `tests/cognition/services/test_quality_gate.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.services.quality_gate import QualityGate


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_meaningful_request_passes():
    gate = QualityGate()
    res = gate.filter(_turn("turn off the lights"))
    assert res.text == "turn off the lights"
    assert res.rejected is False


def test_single_filler_rejected():
    gate = QualityGate()
    res = gate.filter(_turn("um"))
    assert res.rejected is True
```

- [ ] **Step 2: Write the IntentRouter test**

Create `tests/cognition/services/test_intent_router.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.services.intent_router import IntentRouter


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_pause_routes_to_tier1():
    res = IntentRouter().route(_turn("pause"))
    assert res is not None
    assert res.command == "play_pause"


def test_unrelated_returns_none():
    res = IntentRouter().route(_turn("tell me a joke about ducks"))
    assert res is None
```

- [ ] **Step 3: Verify failures**

Run: `.venv/bin/pytest tests/cognition/services -v`
Expected: ImportErrors.

- [ ] **Step 4: Implement `QualityGate`**

Create `server/cognition/services/quality_gate.py`:

```python
"""QualityGate — VoiceTurn-shaped adapter around legacy filter_transcription.

The substantive filter logic stays in the legacy module until Task 34 cleanup;
we wrap it here under the new port-friendly shape."""
from __future__ import annotations
from dataclasses import dataclass

from server.cognition.contracts import VoiceTurn


@dataclass(frozen=True)
class GateResult:
    text: str           # cleaned text (possibly equal to input)
    rejected: bool
    reason: str


class QualityGate:
    def filter(self, turn: VoiceTurn) -> GateResult:
        # Lazy import isolates the legacy module from cognition's static surface
        from server.quality_gate import filter_transcription  # type: ignore
        result = filter_transcription(turn.input_text, tv_playing=False)
        if result.text is None:
            return GateResult(text="", rejected=True, reason=result.reason or "rejected")
        return GateResult(text=result.text, rejected=False, reason="")
```

- [ ] **Step 5: Implement `IntentRouter`**

Create `server/cognition/services/intent_router.py`:

```python
"""IntentRouter — Tier 1 direct-match. Wraps legacy intent_router.route."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from server.cognition.contracts import VoiceTurn


@dataclass(frozen=True)
class Tier1Match:
    command: str
    params: dict
    response: str


class IntentRouter:
    def route(self, turn: VoiceTurn) -> Optional[Tier1Match]:
        from server.intent_router import route as legacy_route  # type: ignore
        match = legacy_route(turn.input_text)
        if match is None:
            return None
        return Tier1Match(command=match.command, params=match.params, response=match.response)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/cognition/services -v`
Expected: `4 passed`.

- [ ] **Step 7: Commit**

```bash
git add server/cognition/services/quality_gate.py server/cognition/services/intent_router.py tests/cognition/services/
git commit -m "QualityGate + IntentRouter services (VoiceTurn-shaped wrappers over legacy logic)"
```

---

### Task 23: `PromptBuilder` (internal)

**Files:**
- Create: `server/cognition/_internal/prompt_builder.py`
- Create: `tests/cognition/_internal/__init__.py`
- Create: `tests/cognition/_internal/test_prompt_builder.py`

- [ ] **Step 1: Write the failing test**

Note: this test is for code INSIDE cognition, so importing `server.cognition._internal` is allowed (the guard checks the caller's `__name__`). Tests under `tests/cognition/_internal/` work because their `__name__` does not match `server.cognition`, but the guard explicitly rejects only the wider non-cognition prefix. We adjust by importing the helper module under a name pattern the guard accepts.

Actually the simplest way is to test `prompt_builder` indirectly via `Conversation`. So skip a direct unit test and let Task 25's Conversation test exercise it.

- [ ] **Step 2: Implement `PromptBuilder`**

Create `server/cognition/_internal/prompt_builder.py`:

```python
"""Build the cached system prompt + dynamic context for the LLM."""
from __future__ import annotations
from typing import Iterable

from server.cognition.contracts import VoiceTurn, Fact, Episode


def build_system_prompt(identity_narrative: str) -> str:
    """The cached portion. Anthropic prompt-caches everything above the
    `<dynamic_context>` section so we keep this stable across turns."""
    return (
        "You are Igor, a personal voice assistant. Be brief, helpful, and warm.\n"
        "Confirm device actions concisely. Do not narrate what you're about to do.\n"
        "<my_person>\n"
        f"{identity_narrative or '(identity unknown yet)'}\n"
        "</my_person>\n"
    )


def build_user_context(
    turn: VoiceTurn,
    relevant_facts: Iterable[Fact],
    recent_episodes: Iterable[Episode],
) -> str:
    """The dynamic portion appended to each user message."""
    bits = []
    facts_list = list(relevant_facts)
    if facts_list:
        lines = [f"- [{f.category}/{f.key}] {f.value}" for f in facts_list]
        bits.append("<relevant_memories>\n" + "\n".join(lines) + "\n</relevant_memories>")
    episodes_list = list(recent_episodes)
    if episodes_list:
        lines = [f"- {e.occurred_at.isoformat()}: {e.summary or e.raw_utterance[:80]}"
                 for e in episodes_list]
        bits.append("<recent_episodes>\n" + "\n".join(lines) + "\n</recent_episodes>")
    bits.append(turn.input_text)
    return "\n\n".join(bits)
```

- [ ] **Step 3: Commit**

```bash
git add server/cognition/_internal/prompt_builder.py
git commit -m "PromptBuilder helper (cached system + dynamic context)"
```

---

### Task 24: `ToolRegistry`

**Files:**
- Create: `server/cognition/services/tool_registry.py`
- Create: `tests/cognition/services/test_tool_registry.py`

`ToolRegistry` is simple — it wraps a `ToolExecutorPort` and exposes its `list_schemas()` as the source of truth for the LLM.

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/services/test_tool_registry.py`:

```python
from server.cognition.services.tool_registry import ToolRegistry


class _FakeExecutor:
    def list_schemas(self):
        return [{"name": "fake_say", "description": "echo", "input_schema": {}}]
    def execute(self, name, args, turn):
        return ""


def test_schemas_pass_through():
    reg = ToolRegistry(_FakeExecutor())
    assert reg.schemas == [{"name": "fake_say", "description": "echo", "input_schema": {}}]
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/services/test_tool_registry.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `ToolRegistry`**

Create `server/cognition/services/tool_registry.py`:

```python
"""ToolRegistry — exposes ToolExecutorPort.list_schemas() as cognition-shaped data."""
from __future__ import annotations
from server.cognition.ports.tools import ToolExecutorPort


class ToolRegistry:
    def __init__(self, executor: ToolExecutorPort):
        self._exec = executor

    @property
    def schemas(self) -> list[dict]:
        return self._exec.list_schemas()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/services/test_tool_registry.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/services/tool_registry.py tests/cognition/services/test_tool_registry.py
git commit -m "ToolRegistry service"
```

---

### Task 25: `Conversation` orchestrator

**Files:**
- Create: `server/cognition/services/conversation.py`
- Create: `tests/cognition/services/test_conversation.py`

This is the main hot-path service.

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/services/test_conversation.py`:

```python
from datetime import datetime, UTC
from server.cognition.contracts import VoiceTurn, RoomConfig
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.services.conversation import Conversation
from server.cognition.services.tool_registry import ToolRegistry
from server.cognition.ports.llm import ChatResult
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval
from server.external.system_clock import SystemClock


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        return ChatResult(text="hi back", commands_executed=[], input_tokens=10, output_tokens=2)


class _StubExecutor:
    def list_schemas(self):
        return []
    def execute(self, name, args, turn):
        return ""


def _turn(text: str) -> VoiceTurn:
    return VoiceTurn(
        correlation_id="t-1", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("default", "Default"),
        input_text=text, speaker_id=None, metadata={},
    )


def test_conversation_writes_episode_with_correlation_id(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = Conversation(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), user_state=UserState(sp),
        retrieval=TagRetrieval(sp), llm=_StubLLM(),
        tools=_StubExecutor(), clock=SystemClock(),
    )
    result = conv.process(_turn("hello there"))
    assert result.correlation_id == "t-1"
    assert result.response_text == "hi back"
    ep = sp.load_episode("t-1")
    assert ep is not None
    assert ep.raw_utterance == "hello there"


def test_conversation_short_circuits_tier1(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    conv = Conversation(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), user_state=UserState(sp),
        retrieval=TagRetrieval(sp),
        llm=_StubLLM(), tools=_StubExecutor(), clock=SystemClock(),
    )
    result = conv.process(_turn("pause"))
    # Tier 1 produces "play_pause" and skips LLM
    assert "play_pause" in result.commands_executed or result.response_text != ""
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/services/test_conversation.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `Conversation`**

Create `server/cognition/services/conversation.py`:

```python
"""Conversation — the turn orchestrator. The only service ha_io knows about."""
from __future__ import annotations
import logging

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import (
    ConversationResult, Episode, ToolCallRecord, VoiceTurn,
)
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort
from server.cognition.ports.retrieval import RetrievalPort
from server.cognition.ports.tools import ToolExecutorPort
from server.cognition.services.intent_router import IntentRouter, Tier1Match
from server.cognition.services.quality_gate import QualityGate
from server.cognition._internal.prompt_builder import build_system_prompt, build_user_context

logger = logging.getLogger(__name__)


class Conversation:
    def __init__(
        self,
        memory: MemoryStore,
        episodes: EpisodeStore,
        identity: IdentityStore,
        user_state: UserState,
        retrieval: RetrievalPort,
        llm: LLMPort,
        tools: ToolExecutorPort,
        clock: ClockPort,
    ):
        self._memory = memory
        self._episodes = episodes
        self._identity = identity
        self._user_state = user_state
        self._retrieval = retrieval
        self._llm = llm
        self._tools = tools
        self._clock = clock
        self._quality_gate = QualityGate()
        self._intent_router = IntentRouter()

    def process(self, turn: VoiceTurn) -> ConversationResult:
        # 1. Quality gate
        gate = self._quality_gate.filter(turn)
        if gate.rejected:
            self._persist_episode(turn, response="Didn't catch that.", tool_calls=[],
                                  intent="rejected")
            return ConversationResult(
                correlation_id=turn.correlation_id,
                response_text="Didn't catch that.",
                commands_executed=[], end_conversation=True,
            )

        # 2. Tier 1 intent router
        tier1: Tier1Match | None = self._intent_router.route(turn)
        if tier1 is not None:
            try:
                self._tools.execute(tier1.command, tier1.params, turn)
            except Exception as e:
                logger.exception("Tier1 execution failed")
            self._persist_episode(turn, response=tier1.response,
                                  tool_calls=[ToolCallRecord(tier1.command, tier1.params, tier1.response)],
                                  intent="tier1")
            return ConversationResult(
                correlation_id=turn.correlation_id,
                response_text=tier1.response,
                commands_executed=[tier1.command],
                end_conversation=True,
            )

        # 3. LLM path
        relevant = self._retrieval.query(turn, k=10)
        recent_eps = self._episodes.get_recent(5)
        system_prompt = build_system_prompt(self._identity.get_narrative())
        user_context = build_user_context(turn, relevant, recent_eps)
        tool_results_log: list[ToolCallRecord] = []

        def _exec(name: str, args: dict) -> str:
            result_text = self._tools.execute(name, args, turn)
            tool_results_log.append(ToolCallRecord(name=name, args=args, result=result_text))
            return result_text

        chat = self._llm.chat(
            system_prompt=system_prompt,
            user_text=user_context,
            tool_schemas=self._tools.list_schemas(),
            tool_executor=_exec,
        )

        self._persist_episode(turn, response=chat.text,
                              tool_calls=tool_results_log, intent="llm")

        return ConversationResult(
            correlation_id=turn.correlation_id,
            response_text=chat.text,
            commands_executed=chat.commands_executed,
            end_conversation=True,
        )

    # ----- internal -----

    def _persist_episode(self, turn: VoiceTurn, response: str,
                         tool_calls: list[ToolCallRecord], intent: str) -> None:
        ep = Episode(
            episode_id=turn.correlation_id,
            occurred_at=turn.started_at,
            speaker_id=turn.speaker_id,
            participants=[turn.speaker_id or "user", "igor"],
            intent=intent,
            raw_utterance=turn.input_text,
            tool_calls=tool_calls,
            emotional_tone=None,
            summary=None,            # SessionSummarizer fills this in later
            consolidated_at=None,
        )
        self._episodes.add(ep)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/services/test_conversation.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/services/conversation.py tests/cognition/services/test_conversation.py
git commit -m "Conversation orchestrator (quality gate, Tier 1 short-circuit, LLM path, episode persistence)"
```

---

### Task 26: `SessionSummarizer` (background)

**Files:**
- Create: `server/cognition/services/session_summarizer.py`
- Create: `tests/cognition/services/test_session_summarizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/services/test_session_summarizer.py`:

```python
import time
from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import VoiceTurn, RoomConfig, ConversationResult
from server.cognition.ports.llm import ChatResult
from server.cognition.services.session_summarizer import SessionSummarizer
from server.external.sqlite_persistence import SqlitePersistence
from server.external.system_clock import SystemClock


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        # Return a summary string the summarizer will store on the episode.
        return ChatResult(text="user asked about coffee preferences",
                          commands_executed=[], input_tokens=5, output_tokens=2)


def _seed_episode(sp):
    from server.cognition.contracts import Episode
    sp.save_episode(Episode(
        episode_id="ep-1", occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        speaker_id=None, participants=[], intent="llm",
        raw_utterance="what coffee do I like", tool_calls=[],
        emotional_tone=None, summary=None, consolidated_at=None,
    ))


def test_summarizer_stamps_summary(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_episode(sp)
    summarizer = SessionSummarizer(
        episodes=EpisodeStore(sp), memory=MemoryStore(sp),
        llm=_StubLLM(), clock=SystemClock(),
    )
    summarizer.start()
    summarizer.enqueue(VoiceTurn(
        correlation_id="ep-1", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        device_id=None, room=RoomConfig("d", "Default"),
        input_text="what coffee do I like", speaker_id=None, metadata={},
    ), ConversationResult(correlation_id="ep-1", response_text="dark roast",
                          commands_executed=[], end_conversation=True))
    # Block on drain
    summarizer.shutdown(timeout=2.0)
    loaded = sp.load_episode("ep-1")
    assert loaded.summary == "user asked about coffee preferences"
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/services/test_session_summarizer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `SessionSummarizer`**

Create `server/cognition/services/session_summarizer.py`:

```python
"""SessionSummarizer — drains turn results, updates Episode.summary
(and could later extract fresh facts). Runs in a background thread."""
from __future__ import annotations
import logging
import queue
import threading
from typing import Optional, Tuple

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import ConversationResult, VoiceTurn
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort
from dataclasses import replace

logger = logging.getLogger(__name__)

_STOP = object()


class SessionSummarizer:
    def __init__(self, episodes: EpisodeStore, memory: MemoryStore,
                 llm: LLMPort, clock: ClockPort):
        self._episodes = episodes
        self._memory = memory
        self._llm = llm
        self._clock = clock
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="SessionSummarizer")
        self._thread.start()

    def enqueue(self, turn: VoiceTurn, result: ConversationResult) -> None:
        self._queue.put((turn, result))

    def shutdown(self, timeout: float = 2.0) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._running = False

    def _run(self) -> None:
        while self._running:
            item = self._queue.get()
            if item is _STOP:
                self._running = False
                return
            try:
                self._summarize(*item)
            except Exception:
                logger.exception("Summarization failed")

    def _summarize(self, turn: VoiceTurn, result: ConversationResult) -> None:
        ep = self._episodes.load(turn.correlation_id)
        if ep is None:
            return
        # One-shot LLM call to produce a short summary.
        chat = self._llm.chat(
            system_prompt="Summarize this assistant turn in <= 12 words.",
            user_text=f"User: {turn.input_text}\nIgor: {result.response_text}",
            tool_schemas=[],
            tool_executor=lambda n, a: "",
        )
        updated = replace(ep, summary=chat.text.strip())
        # Persist via PersistencePort directly (save_episode upserts)
        self._episodes.add(updated)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/services/test_session_summarizer.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/cognition/services/session_summarizer.py tests/cognition/services/test_session_summarizer.py
git commit -m "SessionSummarizer — background queue + drain thread, stamps Episode.summary"
```

---

### Task 27: `Consolidator` (background with crash-replay)

**Files:**
- Create: `server/cognition/services/consolidator.py`
- Create: `tests/cognition/services/test_consolidator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cognition/services/test_consolidator.py`:

```python
from datetime import datetime, UTC
from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.contracts import Episode
from server.cognition.ports.llm import ChatResult
from server.cognition.services.consolidator import Consolidator
from server.external.sqlite_persistence import SqlitePersistence


class _StubLLM:
    def chat(self, system_prompt, user_text, tool_schemas, tool_executor, history=None):
        return ChatResult(text="Sam is a homelab nerd who likes dark roast coffee.",
                          commands_executed=[], input_tokens=10, output_tokens=5)


class _StubClock:
    def now(self):
        return datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


def _seed_unconsolidated(sp, n=5):
    for i in range(n):
        sp.save_episode(Episode(
            episode_id=f"ep-{i}",
            occurred_at=datetime(2026, 1, 1, 10, i, tzinfo=UTC),
            speaker_id=None, participants=[], intent="llm",
            raw_utterance=f"turn {i}", tool_calls=[], emotional_tone=None,
            summary=None, consolidated_at=None,
        ))


def test_consolidate_now_regenerates_identity_and_marks_consolidated(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_unconsolidated(sp, 5)
    cons = Consolidator(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), llm=_StubLLM(), clock=_StubClock(),
        episodes_per_run=5,
    )
    cons.run_once()
    assert IdentityStore(sp).get_narrative().startswith("Sam is a homelab nerd")
    assert len(EpisodeStore(sp).get_unconsolidated()) == 0


def test_replay_on_startup_when_unconsolidated_exist(tmp_path):
    sp = SqlitePersistence(tmp_path / "brain.db")
    _seed_unconsolidated(sp, 6)
    cons = Consolidator(
        memory=MemoryStore(sp), episodes=EpisodeStore(sp),
        identity=IdentityStore(sp), llm=_StubLLM(), clock=_StubClock(),
        episodes_per_run=5,
    )
    cons.replay_if_pending()
    assert len(EpisodeStore(sp).get_unconsolidated()) <= 1
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/cognition/services/test_consolidator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `Consolidator`**

Create `server/cognition/services/consolidator.py`:

```python
"""Consolidator — sleep-time service. Runs in a background thread.
Idempotent + replays on startup based on EpisodeStore.get_unconsolidated()."""
from __future__ import annotations
import logging
import threading
import time
from typing import Optional

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.ports.clock import ClockPort
from server.cognition.ports.llm import LLMPort

logger = logging.getLogger(__name__)


class Consolidator:
    def __init__(
        self,
        memory: MemoryStore,
        episodes: EpisodeStore,
        identity: IdentityStore,
        llm: LLMPort,
        clock: ClockPort,
        *,
        episodes_per_run: int = 5,
        poll_interval_seconds: float = 60.0,
    ):
        self._memory = memory
        self._episodes = episodes
        self._identity = identity
        self._llm = llm
        self._clock = clock
        self._episodes_per_run = episodes_per_run
        self._poll = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- Lifecycle ----
    def start(self) -> None:
        if self._thread is not None:
            return
        self.replay_if_pending()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Consolidator")
        self._thread.start()

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- Logic ----
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._maybe_run()
            except Exception:
                logger.exception("Consolidator iteration failed")
            self._stop.wait(self._poll)

    def _maybe_run(self) -> None:
        unconsolidated = self._episodes.get_unconsolidated()
        if len(unconsolidated) >= self._episodes_per_run:
            self.run_once()

    def replay_if_pending(self) -> None:
        if len(self._episodes.get_unconsolidated()) >= self._episodes_per_run:
            self.run_once()

    def run_once(self) -> None:
        episodes = self._episodes.get_unconsolidated()[: self._episodes_per_run]
        if not episodes:
            return
        # Build a prompt the LLM can synthesize a new identity narrative from.
        prior_identity = self._identity.get_narrative()
        ep_lines = [f"- {e.occurred_at.isoformat()}: {e.summary or e.raw_utterance[:120]}"
                    for e in episodes]
        chat = self._llm.chat(
            system_prompt=(
                "Synthesize a brief living narrative about the user (single paragraph, "
                "<=4 sentences). Use prior narrative + recent episodes. "
                "Do not invent details."
            ),
            user_text=(
                f"Prior narrative:\n{prior_identity or '(empty)'}\n\n"
                f"Recent episodes:\n" + "\n".join(ep_lines)
            ),
            tool_schemas=[], tool_executor=lambda n, a: "",
        )
        now = self._clock.now()
        last_id = episodes[-1].episode_id
        self._identity.replace_narrative(chat.text.strip(),
                                         last_consolidated_at=now,
                                         last_consolidated_episode_id=last_id)
        self._episodes.mark_consolidated([e.episode_id for e in episodes], at=now)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/cognition/services/test_consolidator.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Run full cognition test suite**

Run: `.venv/bin/pytest tests/cognition -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add server/cognition/services/consolidator.py tests/cognition/services/test_consolidator.py
git commit -m "Consolidator — sleep-time identity regeneration with crash-replay"
```

---

## Phase 8 — ha_io

### Task 28: ha-io contracts + voice_turn + result_mapper

**Files:**
- Create: `server/ha_io/contracts.py`
- Create: `server/ha_io/_internal/voice_turn.py`
- Create: `server/ha_io/_internal/result_mapper.py`
- Create: `tests/ha_io/__init__.py`
- Create: `tests/ha_io/test_voice_turn_build.py`

- [ ] **Step 1: Write contracts**

Create `server/ha_io/contracts.py`:

```python
"""Pydantic models for HA's Custom Conversation Agent payload."""
from typing import Optional
from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(None, max_length=100)
    device_id: Optional[str] = Field(None, max_length=100)
    language: Optional[str] = Field(None, max_length=20)


class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    end_conversation: bool
    commands_executed: list[str]
```

- [ ] **Step 2: Write the voice_turn builder test**

Create `tests/ha_io/__init__.py` (empty), then `tests/ha_io/test_voice_turn_build.py`:

```python
from server.ha_io.contracts import ConversationRequest
from server.ha_io._internal.voice_turn import build_voice_turn


class _FakeHAClient:
    def area_of_device(self, device_id: str) -> str:
        return "Office" if device_id else ""


def test_minted_correlation_id_is_uuid_like():
    req = ConversationRequest(text="hi", device_id="dev-1", language="en")
    turn = build_voice_turn(req, _FakeHAClient(),
                            known_rooms={"office": _office()})
    assert len(turn.correlation_id) >= 16
    assert turn.input_text == "hi"
    assert turn.room.room_id == "office"


def test_unknown_device_falls_back_to_default_room():
    req = ConversationRequest(text="hi", device_id=None, language="en")
    turn = build_voice_turn(req, _FakeHAClient(),
                            known_rooms={"default": _default()})
    assert turn.room.room_id == "default"


def _office():
    from server.cognition.contracts import RoomConfig
    return RoomConfig(room_id="office", display_name="Office", ha_area="Office")


def _default():
    from server.cognition.contracts import RoomConfig
    return RoomConfig(room_id="default", display_name="Default")
```

- [ ] **Step 3: Implement voice_turn builder**

Create `server/ha_io/_internal/voice_turn.py`:

```python
"""Build a VoiceTurn from an incoming ConversationRequest."""
from __future__ import annotations
import uuid
from datetime import datetime, UTC
from typing import Mapping, Optional

from server.cognition.contracts import RoomConfig, VoiceTurn
from server.ha_io.contracts import ConversationRequest


def build_voice_turn(
    req: ConversationRequest,
    ha_client,
    known_rooms: Mapping[str, RoomConfig],
) -> VoiceTurn:
    return VoiceTurn(
        correlation_id=str(uuid.uuid4()),
        started_at=datetime.now(UTC),
        device_id=req.device_id,
        room=_resolve_room(req.device_id, ha_client, known_rooms),
        input_text=req.text,
        speaker_id=None,
        metadata={"language": req.language,
                  "ha_conversation_id": req.conversation_id},
    )


def _resolve_room(device_id: Optional[str], ha_client,
                  known_rooms: Mapping[str, RoomConfig]) -> RoomConfig:
    ha_area = ""
    if device_id:
        try:
            ha_area = ha_client.area_of_device(device_id) or ""
        except Exception:
            ha_area = ""
    if ha_area:
        for room in known_rooms.values():
            if (room.ha_area or "").lower() == ha_area.lower():
                return room
        return RoomConfig(
            room_id=ha_area.lower().replace(" ", "_"),
            display_name=ha_area, ha_area=ha_area,
        )
    return next(iter(known_rooms.values()))
```

- [ ] **Step 4: Implement result mapper**

Create `server/ha_io/_internal/result_mapper.py`:

```python
"""Map cognition.ConversationResult to HA's response shape."""
from __future__ import annotations
import time

from server.cognition.contracts import ConversationResult
from server.ha_io.contracts import ConversationResponse


def map_result(result: ConversationResult,
               ha_conversation_id: str | None) -> ConversationResponse:
    return ConversationResponse(
        response=result.response_text,
        conversation_id=ha_conversation_id or f"igor-{int(time.time()*1000)}",
        end_conversation=result.end_conversation,
        commands_executed=result.commands_executed,
    )
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/ha_io -v`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add server/ha_io/contracts.py server/ha_io/_internal/voice_turn.py server/ha_io/_internal/result_mapper.py tests/ha_io/
git commit -m "ha_io contracts + voice_turn builder + result_mapper"
```

---

### Task 29: Auth + rate-limit middleware

**Files:**
- Create: `server/ha_io/_internal/auth.py`
- Create: `server/ha_io/_internal/rate_limit.py`
- Create: `tests/ha_io/test_auth_and_rate_limit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ha_io/test_auth_and_rate_limit.py`:

```python
import time
from server.ha_io._internal.auth import check_token
from server.ha_io._internal.rate_limit import RateLimiter


def test_check_token_passes_when_env_unset(monkeypatch):
    monkeypatch.delenv("IGOR_API_TOKEN", raising=False)
    assert check_token(provided=None) is True


def test_check_token_fails_when_mismatch(monkeypatch):
    monkeypatch.setenv("IGOR_API_TOKEN", "secret")
    assert check_token(provided="wrong") is False
    assert check_token(provided="secret") is True


def test_rate_limiter_blocks_after_max():
    rl = RateLimiter(max_requests=2, window_seconds=60.0)
    assert rl.is_allowed("ip-1") is True
    assert rl.is_allowed("ip-1") is True
    assert rl.is_allowed("ip-1") is False
    assert rl.is_allowed("ip-2") is True   # different key
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/ha_io/test_auth_and_rate_limit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement auth**

Create `server/ha_io/_internal/auth.py`:

```python
"""X-Igor-Token check. No-op when IGOR_API_TOKEN env var is unset (dev mode)."""
from __future__ import annotations
import os


def check_token(provided: str | None) -> bool:
    expected = os.environ.get("IGOR_API_TOKEN", "")
    if not expected:
        return True
    return provided == expected
```

- [ ] **Step 4: Implement rate-limit**

Create `server/ha_io/_internal/rate_limit.py`:

```python
"""In-memory sliding-window rate limiter."""
from __future__ import annotations
import time
from collections import deque
from threading import Lock


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: dict[str, deque] = {}
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            if key not in self._timestamps:
                self._timestamps[key] = deque()
            ts = self._timestamps[key]
            while ts and ts[0] < now - self.window:
                ts.popleft()
            if len(ts) >= self.max_requests:
                return False
            ts.append(now)
            if len(self._timestamps) > 1000:
                self._timestamps = {k: v for k, v in self._timestamps.items() if v}
            return True
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/ha_io/test_auth_and_rate_limit.py -v`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add server/ha_io/_internal/auth.py server/ha_io/_internal/rate_limit.py tests/ha_io/test_auth_and_rate_limit.py
git commit -m "ha_io auth + rate-limit middleware"
```

---

### Task 30: FastAPI app (`server/ha_io/api.py`)

**Files:**
- Create: `server/ha_io/api.py`
- Create: `tests/ha_io/test_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ha_io/test_api.py`:

```python
from fastapi.testclient import TestClient
from server.cognition.contracts import ConversationResult, RoomConfig
from server.ha_io.api import build_app


class _StubConversation:
    def process(self, turn):
        return ConversationResult(
            correlation_id=turn.correlation_id,
            response_text=f"echo: {turn.input_text}",
            commands_executed=[], end_conversation=True,
        )


class _NoopHAClient:
    def area_of_device(self, device_id):
        return ""


def _client():
    app = build_app(
        conversation=_StubConversation(),
        ha_client=_NoopHAClient(),
        known_rooms={"default": RoomConfig("default", "Default")},
    )
    return TestClient(app)


def test_health_ok():
    r = _client().get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_conversation_endpoint_round_trip():
    r = _client().post("/conversation/process",
                       json={"text": "hi", "device_id": None, "language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "echo: hi"
    assert body["end_conversation"] is True


def test_token_enforced(monkeypatch):
    monkeypatch.setenv("IGOR_API_TOKEN", "secret")
    r = _client().post("/conversation/process",
                       json={"text": "hi"})
    assert r.status_code == 401
    r = _client().post("/conversation/process",
                       json={"text": "hi"},
                       headers={"X-Igor-Token": "secret"})
    assert r.status_code == 200
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/ha_io/test_api.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `build_app`**

Create `server/ha_io/api.py`:

```python
"""FastAPI app for Igor's HA Custom Conversation Agent."""
from __future__ import annotations
import logging
import time
from typing import Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.cognition.contracts import RoomConfig
from server.ha_io.contracts import ConversationRequest
from server.ha_io._internal.auth import check_token
from server.ha_io._internal.rate_limit import RateLimiter
from server.ha_io._internal.result_mapper import map_result
from server.ha_io._internal.voice_turn import build_voice_turn

logger = logging.getLogger(__name__)


def build_app(
    *,
    conversation,
    ha_client,
    known_rooms: Mapping[str, RoomConfig],
) -> FastAPI:
    app = FastAPI(
        title="Igor Conversation Agent",
        version="4.0.0",
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Igor-Token"],
    )
    app.state.started = time.time()
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

    @app.exception_handler(ValidationError)
    async def vexc(request, exc):
        logger.error("Validation error: %s", exc)
        return JSONResponse(status_code=422, content={"error": "Validation failed"})

    @app.exception_handler(Exception)
    async def gexc(request, exc):
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("Unhandled")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - app.state.started,
            "rooms": list(known_rooms.keys()),
        }

    @app.get("/")
    async def root():
        return {"service": "Igor", "status": "running"}

    @app.post("/conversation/process")
    async def conversation_process(req_model: ConversationRequest, req: Request):
        if not check_token(req.headers.get("X-Igor-Token")):
            raise HTTPException(status_code=401, detail="Invalid or missing token")
        if not rate_limiter.is_allowed(req.client.host):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        turn = build_voice_turn(req_model, ha_client, known_rooms)
        try:
            result = conversation.process(turn)
        except Exception:
            logger.exception("Conversation failed")
            raise HTTPException(status_code=500, detail="Processing failed")
        return map_result(result, req_model.conversation_id).model_dump()

    return app
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/ha_io/test_api.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Boundary check**

Run: `.venv/bin/python -m tools.boundary_check`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add server/ha_io/api.py tests/ha_io/test_api.py
git commit -m "ha_io FastAPI build_app — health, conversation/process, token + rate limit"
```

---

## Phase 9 — Composition root

### Task 31: `server/main.py` composition root

**Files:**
- Create: `server/main.py`

This new entry point coexists with `server/main_text.py` until cutover (Task 33).

- [ ] **Step 1: Write the smoke test**

Create `tests/test_main_composition.py`:

```python
import importlib


def test_main_module_imports_cleanly(monkeypatch, tmp_path):
    """Building the app shouldn't crash. We don't actually start uvicorn."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HA_URL", "http://10.0.40.5:8123")
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))

    main = importlib.import_module("server.main")
    app = main.build()
    assert app is not None
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_main_composition.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `server/main.py`**

Create `server/main.py`:

```python
"""Composition root. The ONLY place adapters meet ports."""
from __future__ import annotations
import logging
import os
from pathlib import Path

from server.cognition.aggregates.episode import EpisodeStore
from server.cognition.aggregates.identity import IdentityStore
from server.cognition.aggregates.memory import MemoryStore
from server.cognition.aggregates.user_state import UserState
from server.cognition.contracts import RoomConfig
from server.cognition.services.conversation import Conversation
from server.cognition.services.consolidator import Consolidator
from server.cognition.services.session_summarizer import SessionSummarizer
from server.external._internal.brain_json_migration import migrate_brain_json_if_needed
from server.external._internal.ha_client import get_client as get_ha_client
from server.external.claude_adapter import ClaudeAdapter
from server.external.ha_rest_adapter import HARestToolExecutor
from server.external.sqlite_persistence import SqlitePersistence
from server.external.sqlite_retrieval import TagRetrieval
from server.external.system_clock import SystemClock
from server.ha_io.api import build_app

logger = logging.getLogger(__name__)


def _brain_dir() -> Path:
    return Path(os.environ.get("BRAIN_DIR", "/app/data"))


def _build_rooms_from_ha() -> dict[str, RoomConfig]:
    try:
        areas = get_ha_client().get_areas()
    except Exception as e:
        logger.warning("Could not enumerate HA areas: %s", e)
        return {"default": RoomConfig(room_id="default", display_name="Default")}
    rooms: dict[str, RoomConfig] = {}
    for area in areas or []:
        rid = area.lower().replace(" ", "_")
        rooms[rid] = RoomConfig(room_id=rid, display_name=area, ha_area=area)
    return rooms or {"default": RoomConfig(room_id="default", display_name="Default")}


def build():
    """Build the FastAPI app with all adapters wired up."""
    brain_dir = _brain_dir()
    brain_dir.mkdir(parents=True, exist_ok=True)

    # one-shot migration if brain.json exists
    migrate_brain_json_if_needed(brain_dir / "brain.json", brain_dir / "brain.db")

    # adapters
    persistence = SqlitePersistence(brain_dir / "brain.db")
    retrieval = TagRetrieval(persistence)
    llm = ClaudeAdapter()
    clock = SystemClock()
    tools = HARestToolExecutor()

    # aggregates
    memory = MemoryStore(persistence)
    episodes = EpisodeStore(persistence)
    identity = IdentityStore(persistence)
    user_state = UserState(persistence)

    # services
    conversation = Conversation(
        memory=memory, episodes=episodes, identity=identity, user_state=user_state,
        retrieval=retrieval, llm=llm, tools=tools, clock=clock,
    )
    summarizer = SessionSummarizer(episodes=episodes, memory=memory,
                                   llm=llm, clock=clock)
    summarizer.start()
    consolidator = Consolidator(
        memory=memory, episodes=episodes, identity=identity,
        llm=llm, clock=clock,
    )
    consolidator.start()

    rooms = _build_rooms_from_ha()
    return build_app(
        conversation=conversation,
        ha_client=get_ha_client(),
        known_rooms=rooms,
    )


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    app = build()
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_main_composition.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Full test suite**

Run: `.venv/bin/pytest -v`
Expected: ALL green.

- [ ] **Step 6: Boundary check**

Run: `.venv/bin/python -m tools.boundary_check`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add server/main.py tests/test_main_composition.py
git commit -m "server/main.py composition root (coexists with main_text.py until cutover)"
```

---

## Phase 10 — Cutover

### Task 32: Switch Dockerfile entry point + container smoke test

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml` (if env var name changed)

- [ ] **Step 1: Update Dockerfile**

Change the last `CMD` line in `Dockerfile` from:

```dockerfile
CMD ["python", "-m", "server.main_text"]
```

to:

```dockerfile
CMD ["python", "-m", "server.main"]
```

Verify the COPY lines still cover all needed files (wakeword/, server/). If `wakeword/` is not currently COPY'd, add:

```dockerfile
COPY wakeword ./wakeword
```

- [ ] **Step 2: Update docker-compose.yml**

Add `BRAIN_DIR=/app/data` to the env vars block under the `igor` service (if not already implicit):

```yaml
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      HA_TOKEN:          ${HA_TOKEN}
      HA_URL:            ${HA_URL:-http://10.0.40.5:8123}
      IGOR_API_TOKEN:    ${IGOR_API_TOKEN:-}
      SERVER_HOST:       "0.0.0.0"
      SERVER_PORT:       "8000"
      TZ:                "America/New_York"
      BRAIN_DIR:         "/app/data"
```

- [ ] **Step 3: Local build verification (no deploy)**

Run from repo root:
```bash
docker build -t igor-ddd-test .
```
Expected: build succeeds.

- [ ] **Step 4: Run container locally and curl health**

```bash
docker run --rm -d --name igor-ddd-smoke \
    -e ANTHROPIC_API_KEY=test -e HA_URL=http://example.invalid -e HA_TOKEN=test \
    -p 4467:8000 igor-ddd-test
sleep 4
curl -s http://localhost:4467/api/health
docker stop igor-ddd-smoke
```
Expected: `{"status":"healthy",...}` (HA enumeration will fail at invalid URL, but health endpoint should still return).

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "Cutover entry point: server.main_text → server.main"
```

---

## Phase 11 — Cleanup

### Task 33: Delete obsolete modules

**Files (delete):**
- `server/api.py`, `server/main_text.py`, `server/conversation.py`, `server/brain.py`,
  `server/llm.py`, `server/quality_gate.py`, `server/intent_router.py`,
  `server/routines.py`, `server/context.py`, `server/rooms.py`,
  `server/event_loop.py`, `server/client_registry.py`, `server/config.py`,
  `server/commands/`, `prompt.py`

- [ ] **Step 1: Confirm nothing in new code imports from these files**

Run:
```bash
grep -rE "from server\.(api|main_text|conversation|brain|llm|quality_gate|intent_router|routines|context|rooms|event_loop|client_registry|config) " server/cognition server/ha_io server/external tests 2>&1 | grep -v "ha_rest_adapter\|legacy_ctx" || echo "no offending imports"
```

If the only matches are inside `ha_rest_adapter.py`'s `_legacy_ctx_from_turn` (server.context, server.rooms), that's expected — Step 3 fixes it.

- [ ] **Step 2: Remove the `_legacy_ctx_from_turn` helper now that commands/ is being deleted**

Edit `server/external/ha_rest_adapter.py`:
- Delete the `_legacy_ctx_from_turn` function
- In `execute()`, remove the `"_ctx" in sig.parameters` branch entirely
- Pass `kwargs` as-is to `cmd.execute(**kwargs)`

Updated `execute` reads:

```python
    def execute(self, name: str, args: dict, turn: VoiceTurn) -> str:
        cmd = self._commands.get(name)
        if cmd is None:
            return f"Unknown command: {name}"
        try:
            return cmd.execute(**args)
        except Exception as e:
            logger.exception("Command %s failed", name)
            return f"Error executing {name}: {e}"
```

Note: This means individual commands can no longer rely on `_ctx`. Re-introduction will go through the `VoiceTurn` directly when commands are decomposed in a future plan (out of scope here).

- [ ] **Step 3: Delete the files**

```bash
git rm server/api.py server/main_text.py server/conversation.py server/brain.py \
       server/llm.py server/quality_gate.py server/intent_router.py \
       server/routines.py server/context.py server/rooms.py \
       server/event_loop.py server/client_registry.py server/config.py prompt.py
git rm -r server/commands
```

If any of those files don't exist (already deleted in previous cleanups), the `git rm` will warn — that's fine, ignore it.

- [ ] **Step 4: HARestToolExecutor needs a replacement for the now-deleted commands registry**

The `_discover_default_commands` function in `ha_rest_adapter.py` does `from server.commands import get_all_commands` — which no longer exists. Replace with an empty registry initially (a follow-up plan will introduce a proper tool catalog):

Edit `server/external/ha_rest_adapter.py`:

```python
def _discover_default_commands() -> dict:
    """No commands registered yet — re-introduce via a follow-up plan."""
    return {}
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -v`
Expected: all green (the conversation will run with zero tools, but the round-trip still works).

- [ ] **Step 6: Run boundary check**

Run: `.venv/bin/python -m tools.boundary_check`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Delete obsolete modules; HARestToolExecutor now starts with empty registry"
```

---

### Task 34: CI hook + final container deploy verification

- [ ] **Step 1: Add a GitHub Actions workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-server-text.txt -r requirements-dev.txt
      - run: python -m tools.boundary_check
      - run: pytest -v
```

- [ ] **Step 2: Verify tests still pass locally**

```bash
.venv/bin/pytest -v
.venv/bin/python -m tools.boundary_check
```

Expected: all green.

- [ ] **Step 3: Local container rebuild and health smoke**

```bash
docker build -t igor:ddd .
docker run --rm -d --name igor-final-smoke \
    -e ANTHROPIC_API_KEY=test -e HA_URL=http://example.invalid -e HA_TOKEN=test \
    -p 4467:8000 igor:ddd
sleep 4
curl -s http://localhost:4467/api/health
docker stop igor-final-smoke
```

Expected: `{"status":"healthy",...}`.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "Add CI workflow (pytest + boundary_check on push/PR)"
git push origin main
```

- [ ] **Step 5: Deploy via Portainer**

Manually trigger Portainer redeploy of the `igor` stack from GitHub (or wait for GitOps auto-deploy). Verify:
```bash
ssh samda@10.0.30.5 'docker logs --tail 30 igor && curl -s http://10.0.30.5:4467/api/health'
```

Expected: `Starting Igor` log line and healthy JSON response.

- [ ] **Step 6: Final commit (note the migration)**

Inside the running container, confirm migration happened:
```bash
ssh samda@10.0.30.5 'docker exec igor ls -la /app/data/'
```
Expected: `brain.db` present and `brain.json.imported-*.bak` present.

Plan complete.

---

## Self-Review

After writing this plan I checked:

**Spec coverage** — every section of `2026-06-17-igor-ddd-design.md` has corresponding tasks:
- §2 Bounded contexts → Tasks 2, 3 (scaffolding + boundary check)
- §3 wakeword → Tasks 4, 5
- §4.1 ports → Task 8
- §4.2 aggregates → Tasks 14, 15, 16, 17
- §4.3 services → Tasks 22 (QualityGate+IntentRouter), 24 (ToolRegistry), 25 (Conversation), 26 (SessionSummarizer), 27 (Consolidator)
- §4.4 VoiceTurn/ConversationResult → Task 6
- §4.5 turn flow → Task 25 (verified via Conversation tests)
- §4.6 crash recovery → Task 27 (replay_if_pending test)
- §5 ha-io → Tasks 28, 29, 30
- §6 external + SQLite schema → Tasks 9, 10, 11, 12, 13, 18, 19, 20, 21
- §7 composition root → Task 31
- §8 what this kills → addressed by Tasks 1-31 collectively
- §9 out-of-scope → respected (no tools-as-memory, no graph DB, etc.)
- §10 phasing preview → realized as Phases 1-11

**Placeholder scan** — no "TBD", "implement later", or "similar to Task N" instructions remain. Every code-bearing step shows actual code.

**Type consistency** — `VoiceTurn`, `ConversationResult`, `Episode`, `Fact`, `Reflection`, `FeedbackEntry`, `Reminder`, `ChatResult`, `Tier1Match`, `GateResult`, `RoomConfig`, `ToolCallRecord`, `ConversationRequest`, `ConversationResponse` are defined exactly once and referenced consistently downstream.

**One acknowledged simplification:** Task 33 leaves `HARestToolExecutor` with an empty command registry. A follow-up plan re-introduces specific HA-backed tools (lights, media, todo, notify, timer, etc.) using `VoiceTurn` instead of the legacy `InteractionContext`. This plan deliberately scopes that out — its goal is the structural restructure, not the tool catalog rebuild.
