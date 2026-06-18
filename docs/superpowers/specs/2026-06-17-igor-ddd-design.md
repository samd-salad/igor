# Igor — Domain-Driven Design Restructure

**Date:** 2026-06-17
**Status:** Approved design, awaiting implementation plan
**Predecessors:** `MEMORY_ROADMAP.md` (Tier 1 upgrade priorities), recurring debugging sessions on wake-word training/runtime mismatch and `brain.json` coupling issues

---

## 1. Motivation

Igor's bugs over the last two months have clustered around four root causes that the current monolithic structure makes systemic rather than incidental:

1. **Implicit contracts between subsystems.** Wake-word training pipeline assumed `openwakeword.utils.AudioFeatures`; runtime silently used `pyopen_wakeword.OpenWakeWordFeatures`. Same function names, different feature distributions. The "contract" was a function signature buried in a private import.
2. **Hidden coupling through shared state.** `brain.json` is a single blob holding semantic memories, episode summaries, routines, identity narrative, feedback, and reminders. Touching one accidentally affects another (consolidation rewrites memory based on episodes, episodes reference fact ids, etc.).
3. **Cross-cutting flows that nobody owns.** "User says wake word" is a journey across Pi5 → HA → Igor → HA → Pi5. When it breaks, no single component is responsible. Debug requires correlating logs across five processes.
4. **Dead-code rot.** Old `client/`, `shared/`, audio modules survived multiple cleanup passes because nothing made the "this is the new shape" boundary explicit.

A DDD restructure addresses all four by making contracts explicit, decoupling shared state into proper aggregates, naming cross-cutting flows as first-class values, and enforcing context boundaries through both convention and runtime guards.

**North Star:** The memory subsystem (the part that makes Igor "Igor" over time) is the key. The architecture must let memory storage, retrieval, embedding, and consolidation evolve independently of conversation orchestration, the LLM call, and the HA integration.

---

## 2. Bounded Contexts

Four contexts. Each is a Python package with a `contracts.py` (public DTOs + protocols) and `_internal/` (everything else). Cross-context calls route through `contracts` only.

**Prose names** use single-word or hyphenated form for readability: `wakeword`, `ha-io`, `cognition`, `external`. Python module names use underscores where hyphens would break imports (`ha_io/` on disk; `ha-io` in writing).

| Context | What it owns | Why a boundary |
|---|---|---|
| **wakeword** | Wake-word model training pipeline, model artifact, feature-pipeline contract | Genuinely offline; produces a `.tflite` artifact consumed elsewhere. The library + feature spec is the contract that both training and runtime must respect. |
| **ha-io** | HA Custom Conversation Agent wire protocol — receive HA POST, return text response | Thin wire. Adapts HA's payload shape to/from cognition's `VoiceTurn`. Changes when HA's contract changes; otherwise stable. |
| **cognition** | Memory, identity, episodes, conversation orchestration, sleep-time consolidation, command schemas | The "brain." Heaviest context. Hexagonal pattern inside — five ports, four aggregates, six services. |
| **external** | Concrete adapters that satisfy cognition's ports: Claude, HA REST, SQLite persistence, SQLite retrieval | The only files in the project that import third-party libraries (anthropic, requests, sqlite3, sqlite-vec). |

### Repository shape

```
igor/
├── wakeword/                          # Wake-Word context (offline; produces .tflite)
│   ├── contracts.py                   # FEATURE_LIBRARY, MODEL_INPUT_SHAPE, naming pattern
│   ├── train.py                       # entry point
│   ├── render_runtime.py              # generates wyoming systemd args from contract
│   ├── samples/                       # positive/, negative/ (gitignored)
│   ├── models/                        # outputs (.tflite, .onnx — gitignored)
│   └── _internal/
│       ├── features.py                # thin wrapper over pyopen_wakeword
│       ├── pipeline.py                # windowing, labeling, augmentation
│       ├── classifier.py              # PyTorch model + ONNX/TFLite export
│       └── eval.py                    # offline scoring
│
└── server/
    ├── cognition/                     # Cognition context — heaviest
    │   ├── contracts.py               # VoiceTurn, Episode, Fact, ConversationResult, port protocols
    │   ├── ports/
    │   │   ├── persistence.py         # PersistencePort (storage + embedding slot)
    │   │   ├── retrieval.py           # RetrievalPort
    │   │   ├── llm.py                 # LLMPort
    │   │   ├── tools.py               # ToolExecutorPort
    │   │   └── clock.py               # ClockPort
    │   ├── aggregates/
    │   │   ├── memory.py              # MemoryStore (Facts)
    │   │   ├── episode.py             # EpisodeStore (Episodes — load-bearing entity)
    │   │   ├── identity.py            # IdentityStore (narrative + reflections)
    │   │   └── user_state.py          # UserState (Feedback + Reminders, typed sub-collections)
    │   ├── services/
    │   │   ├── quality_gate.py
    │   │   ├── intent_router.py
    │   │   ├── tool_registry.py       # auto-discovers tool schemas
    │   │   ├── conversation.py        # Conversation — turn orchestrator
    │   │   ├── session_summarizer.py  # post-turn extraction, off hot path
    │   │   └── consolidator.py        # sleep-time process, background thread
    │   └── _internal/
    │       └── prompt_builder.py      # system prompt construction
    │
    ├── ha_io/                         # HA Interaction context — thin wire
    │   ├── contracts.py               # ConversationRequest/Response (mirrors HA's payload)
    │   ├── api.py                     # FastAPI endpoints
    │   └── _internal/
    │       ├── auth.py                # X-Igor-Token check
    │       ├── rate_limit.py          # 10/min sliding window per IP
    │       ├── voice_turn.py          # ConversationRequest → VoiceTurn (mints correlation_id)
    │       └── result_mapper.py       # ConversationResult → ConversationResponse
    │
    ├── external/                      # Adapters satisfying cognition ports
    │   ├── claude_adapter.py          # implements LLMPort (only file importing anthropic)
    │   ├── ha_rest_adapter.py         # implements ToolExecutorPort (only file importing requests for HA)
    │   ├── sqlite_persistence.py      # implements PersistencePort
    │   ├── sqlite_retrieval.py        # implements RetrievalPort (tag + recency; hybrid later)
    │   └── _internal/
    │       ├── claude_prompt_cache.py
    │       ├── ha_areas_cache.py
    │       └── brain_json_migration.py # one-shot brain.json → SQLite
    │
    └── main.py                        # Composition root — wires adapters → ports → cognition → ha_io
```

### Boundary invariants

Enforced via lightweight import guards in each context's `__init__.py`:

- `cognition/` may import only from itself and standard library — no concrete third-party libraries.
- `external/` adapters may import third-party libraries freely; they implement cognition's port protocols.
- `ha_io/` may import only `cognition.contracts` (specifically `VoiceTurn`, `ConversationResult`, and the `Conversation` protocol) — not cognition internals, not external.
- `wakeword/` is fully isolated. The runtime systemd config (`deploy/`) is generated from `wakeword/contracts.py` via `render_runtime.py`.
- Importing a context's `_internal` from outside the context raises `ImportError` at import time (small `__init__.py` runtime guard).

These rules can be enforced statically via a single CI script — `python -m tools.boundary_check` — that fails the build if any forbidden import path appears in a context's source.

---

## 3. wakeword Context

Wake-word is structurally isolated. Training produces a `.tflite` artifact and a paired runtime config. Both training and runtime read from `wakeword/contracts.py`.

### Contract surface

```python
# wakeword/contracts.py
FEATURE_LIBRARY = "pyopen_wakeword"
FEATURE_LIBRARY_VERSION = ">=1.1,<2"
FEATURE_DIM = 96
FEATURE_RATE_HZ = 42.7
MODEL_INPUT_SHAPE = (1, 16, 96)
MODEL_INPUT_DTYPE = "float32"
MODEL_OUTPUT_SHAPE = (1, 1)
MODEL_OUTPUT_RANGE = (0.0, 1.0)
MODEL_FILENAME_PATTERN = "{name}_v{version}.tflite"   # wyoming-openwakeword convention
```

### Runtime config generation

`wakeword/render_runtime.py` reads `contracts.py` and outputs the systemd `ExecStart` lines for `wyoming-openwakeword`. When the feature library or model shape changes, one commit updates `contracts.py`; the runtime is regenerated; training picks it up. The class of bug where training and runtime drift apart silently becomes impossible.

### What stays in `_internal/`

The PyTorch classifier, the windowing logic, the cosine-similarity template matching, the ONNX → TFLite conversion via onnx2tf, the offline eval. None of it crosses the context boundary.

---

## 4. cognition Context

Heaviest context. Ports + adapters inside.

### 4.1 Ports (`cognition/ports/`)

Five abstract interfaces. Each evolves on its own schedule. The asymmetric rigor justification: this is where memory work concentrates.

| Port | Initial adapter | Future swaps |
|---|---|---|
| `PersistencePort` | `SqlitePersistence` | Postgres / DuckDB at >10⁵ rows |
| `RetrievalPort` | `TagRetrieval` (tag overlap + recency) | `HybridRetrieval` (tag + sqlite-vec + recency) after ~150 conversations |
| `LLMPort` | `ClaudeAdapter` | swap model versions or providers without touching cognition |
| `ToolExecutorPort` | `HARestToolExecutor` | local-mode adapter for tests |
| `ClockPort` | `SystemClock` | `FrozenClock` in tests for time-sensitive logic |

Embedding is *not* a separate port. Production systems (Mem0, Letta, Graphiti) couple embeddings to storage. Splitting them creates a synchronization headache (whose IDs win when one swaps). The `facts.embedding` BLOB column lives inside `PersistencePort`'s domain.

### 4.2 Aggregates (`cognition/aggregates/`)

Four aggregates decompose the `brain.json` blob. Each owns its data and its consistency rules. Each persists through `PersistencePort`.

**`MemoryStore`** — semantic facts.
```python
class Fact:
    fact_id: str
    category: str
    key: str
    value: str
    tags: list[str]
    source_episode_id: str | None     # provenance
    embedding: bytes | None           # adjacent to fact; populated when index turns on
    valid_at: datetime                # world time
    invalid_at: datetime | None       # null = currently true
    created_at: datetime              # transaction time
```
Operations: `save(fact)`, `forget(category, key)`, `invalidate(fact_id, at)`, `query(...)`.

**`EpisodeStore`** — load-bearing entity AND provenance anchor.
```python
class Episode:
    episode_id: str                   # == VoiceTurn.correlation_id
    occurred_at: datetime
    speaker_id: str | None            # nullable; resemblyzer fills later
    participants: list[str]           # users + "igor"
    intent: str | None                # from intent_router or LLM
    raw_utterance: str
    tool_calls: list[ToolCallRecord]
    emotional_tone: str | None
    summary: str | None               # derived, secondary
    consolidated_at: datetime | None
```
Operations: `add(episode)`, `get_recent(n)`, `get_unconsolidated()`, `mark_consolidated(ids, at)`.

The episode is a structured entity, not a summary string. Free-text `summary` is derived. This matches mid-2026 consensus (arXiv 2502.06975, 2605.06716, 2511.17208) that episodic memory should be first-class.

**`IdentityStore`** — single-row narrative plus an explicit reflections sub-collection.
```python
class Identity:
    narrative: str
    last_consolidated_at: datetime | None         # crash-recovery anchor
    last_consolidated_episode_id: str | None

class Reflection:
    reflection_id: str
    occurred_at: datetime
    note: str
    source_episode_id: str | None
```
Reflections are explicit agent meta-notes (the Consolidator produces them when noticing patterns about its own performance). Separate from the narrative; cheaper than regenerating the whole identity for every observation.

**`UserState`** — feedback and reminders, typed sub-collections.
```python
class FeedbackEntry:
    feedback_id: str
    occurred_at: datetime
    issue: str
    status: Literal["open", "resolved"]
    source_episode_id: str | None

class Reminder:
    reminder_id: str
    name: str
    fire_at: datetime
    room_id: str | None
    status: Literal["pending", "fired", "cancelled"]
    source_episode_id: str | None
```
Merging Feedback + Reminders into one aggregate (vs separate) matches production practice — no surveyed system splits them.

### 4.3 Services (`cognition/services/`)

- **`QualityGate`** — pure-ish filter for junk transcriptions.
- **`IntentRouter`** — Tier 1 direct-match (zero LLM latency for "pause", "lights off").
- **`ToolRegistry`** — auto-discovers tool schemas, exposes them to the LLM.
- **`Conversation`** — *the* turn orchestrator. The only service `ha_io` knows about from inside cognition.
- **`SessionSummarizer`** — post-turn, off hot path via a queue. Extracts facts from the turn's transcript and writes them with `source_episode_id` stamped automatically.
- **`Consolidator`** — sleep-time process. Background thread. Triggered by N unconsolidated episodes OR by HA's `chat_session.async_on_cleanup` (the 5-minute idle signal). Does: regenerate identity narrative, merge duplicate facts, detect contradictions (same category+key, different values → mark older as `invalidate_at = now`), demote stale facts (low recent retrieval), promote high-recall facts, generate reflections.

### 4.4 `VoiceTurn` and `ConversationResult` value objects — provenance for free

```python
@dataclass(frozen=True)
class VoiceTurn:
    correlation_id: str            # uuid4; also == future Episode.episode_id
    started_at: datetime
    device_id: str | None
    room: RoomConfig
    input_text: str
    speaker_id: str | None         # nullable from day 1
    metadata: dict                 # language, ha conversation_id, etc.

@dataclass(frozen=True)
class ConversationResult:
    correlation_id: str            # matches the originating VoiceTurn
    response_text: str
    commands_executed: list[str]
    end_conversation: bool         # tells the satellite whether to keep listening
```

`ConversationResult` is what `Conversation.process(turn)` returns and what `ha_io` maps back to HA's expected response shape.

Created in `ha_io/_internal/voice_turn.py` at the moment HA's POST arrives. Threaded through every service. **Stamped on every persistent write** — every Fact, every Episode, every Reflection, every Feedback, every Reminder carries `source_episode_id == turn.correlation_id`.

`correlation_id == episode_id` is an explicit invariant, not a coincidence. The Episode row IS the provenance anchor. A correlation_id with no Episode is a bug.

### 4.5 Turn flow (sync hot path, <2s)

```
HA POST /conversation/process
   └─→ ha_io.api builds VoiceTurn(correlation_id=uuid4(), input_text, ...)
        └─→ Conversation.process(turn) → ConversationResult
              1. QualityGate.filter(turn)                      # bail early if junk
              2. IntentRouter.route(turn)                      # Tier 1 short-circuit if matched
              3. RetrievalPort.query(turn, k=10)               # relevant facts
              4. context := PromptBuilder.build(
                     identity, recent_episodes, retrieved_facts, turn
                 )
              5. LLMPort.chat(context, tools=registry.schemas)
                     └─ for each tool_call:
                          ToolExecutorPort.execute(name, args, turn)
              6. EpisodeStore.add(Episode(
                     episode_id=turn.correlation_id, ...,
                     tool_calls=[...]
                 ))
              7. return ConversationResult
        └─→ enqueue SessionSummarizer.run(turn, result)        # off hot path
        └─→ Consolidator.maybe_trigger()                       # decoupled, async
```

`SessionSummarizer` and `Consolidator` are background — they own their own threads, communicate via thread-safe queues. The hot path returns text in <2s without blocking on writes. This is the Letta/Mem0 "primary agent + sleep-time agent" split made structural.

### 4.6 Crash recovery

A `threading.Thread` daemon doesn't survive `docker compose restart`. If consolidation is mid-flight when the container restarts, the work is lost. Mitigation: Consolidator is *idempotent* and *replayable*. On startup it reads `IdentityStore.last_consolidated_episode_id`; any episodes with `id > last_consolidated_episode_id AND consolidated_at IS NULL` get re-processed. No durable queue infrastructure required at this scale.

---

## 5. ha-io Context

Thin wire. Three endpoints: `POST /conversation/process`, `GET /api/health`, `GET /`.

### 5.1 Where the episode_id is born

`ha_io/_internal/voice_turn.py` mints the `correlation_id` (uuid4) at the moment HA's request lands:

```python
def build(req: ConversationRequest, ha_client) -> VoiceTurn:
    return VoiceTurn(
        correlation_id=str(uuid.uuid4()),    # = future Episode.episode_id
        started_at=datetime.now(UTC),
        device_id=req.device_id,
        room=resolve_room_from_device(req.device_id, ha_client),
        input_text=req.text,
        speaker_id=None,                      # nullable; resemblyzer fills later
        metadata={
            "language": req.language,
            "ha_conversation_id": req.conversation_id,
        },
    )
```

That id IS the Episode's primary key downstream. Request logs in ha-io grep-correlate against later memory writes.

### 5.2 What ha-io owns

- Auth (`X-Igor-Token` header check when `IGOR_API_TOKEN` env set)
- Rate limiting (10 req/min sliding window per IP)
- Payload validation (Pydantic models in `contracts.py`)
- HA-shape ↔ Cognition-shape mapping in both directions
- Health endpoint (delegates to `Conversation.health()` returning a status dict)

### 5.3 What stays out

No LLM imports, no aggregate imports, no command imports, no prompt construction. ha-io's only inward dependency is the `Conversation` service protocol from `cognition.contracts`.

### 5.4 Evolution

When HA's API contract changes (HA 2027 adds context fields, payload shape evolves), only `ha_io/contracts.py` and `_internal/voice_turn.py` change. Cognition untouched.

---

## 6. external Context

Four adapters. Each is the only place in the project allowed to import its concrete third-party library.

| Adapter | Implements | Third-party libs allowed |
|---|---|---|
| `claude_adapter.py` | `cognition.ports.LLMPort` | `anthropic` |
| `ha_rest_adapter.py` | `cognition.ports.ToolExecutorPort` + internal HAClient | `requests` (for HA REST) |
| `sqlite_persistence.py` | `cognition.ports.PersistencePort` | `sqlite3`, `sqlite-vec` |
| `sqlite_retrieval.py` | `cognition.ports.RetrievalPort` | `sqlite3`, `sqlite-vec` |

A boundary-check script (CI) fails the build if any `import anthropic` appears outside `claude_adapter.py`, etc.

### 6.1 SQLite schema (`data/brain.db`)

```sql
-- Load-bearing entity AND provenance anchor.
CREATE TABLE episodes (
    episode_id          TEXT PRIMARY KEY,        -- == VoiceTurn.correlation_id
    occurred_at         TEXT NOT NULL,
    speaker_id          TEXT,                    -- nullable until speaker-ID lands
    participants        TEXT,                    -- JSON array
    intent              TEXT,
    raw_utterance       TEXT NOT NULL,
    tool_calls          TEXT,                    -- JSON array of ToolCallRecord
    emotional_tone      TEXT,
    summary             TEXT,                    -- derived, secondary
    consolidated_at     TEXT
);
CREATE INDEX episodes_unconsolidated
    ON episodes(occurred_at) WHERE consolidated_at IS NULL;

-- Bi-temporal facts with adjacent embedding slot.
CREATE TABLE facts (
    fact_id             TEXT PRIMARY KEY,
    category            TEXT NOT NULL,
    key                 TEXT NOT NULL,
    value               TEXT NOT NULL,
    tags                TEXT,                    -- JSON array
    source_episode_id   TEXT REFERENCES episodes(episode_id),
    embedding           BLOB,                    -- sqlite-vec col; nullable until index enabled
    valid_at            TEXT NOT NULL,           -- world time
    invalid_at          TEXT,                    -- null = currently true
    created_at          TEXT NOT NULL            -- transaction time
);
CREATE INDEX facts_active
    ON facts(category, key) WHERE invalid_at IS NULL;
CREATE INDEX facts_episode ON facts(source_episode_id);

-- Single-row identity + reflections sub-collection.
CREATE TABLE identity (
    id                            INTEGER PRIMARY KEY CHECK (id = 1),
    narrative                     TEXT NOT NULL,
    last_consolidated_at          TEXT,          -- crash-recovery anchor
    last_consolidated_episode_id  TEXT
);
CREATE TABLE reflections (
    reflection_id      TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    note               TEXT NOT NULL,
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);

-- UserState aggregate.
CREATE TABLE feedback (
    feedback_id        TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    issue              TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'open',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);
CREATE TABLE reminders (
    reminder_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    fire_at            TEXT NOT NULL,
    room_id            TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);
```

Every fact and every user-state row points back to the episode that produced it via `source_episode_id`. Provenance is a foreign key, not a future task.

### 6.2 Retrieval evolution

`sqlite_retrieval.py` starts as tag overlap + recency decay (matches today's behavior; ConvoMem says native context is sufficient until ~150 conversations). When we cross that threshold:

```python
class HybridRetrieval(RetrievalPort):
    """tag overlap + sqlite-vec semantic + recency"""
```

`facts.embedding` is already there as a nullable column. We backfill via a background `BgeSmallEmbedding` job, enable the sqlite-vec virtual index, swap one line in `main.py`. No Cognition changes.

### 6.3 One-shot migration

`brain_json_migration.py` is idempotent:
- Skip if `brain.db` already populated.
- Read `data/brain.json` once; split entries by type → write into the right tables.
- Episodes without an existing UUID get one generated.
- Pre-existing facts get `source_episode_id = NULL` (visible-in-query as "imported pre-provenance").
- Original `brain.json` renamed `brain.json.imported-<timestamp>.bak`.

Runs on container startup via `main.py` before the FastAPI app starts.

---

## 7. Composition Root (`server/main.py`)

```python
def build() -> FastAPI:
    # one-shot migration if needed (idempotent)
    migrate_brain_json_if_needed(BRAIN_DIR / "brain.json", BRAIN_DIR / "brain.db")

    # adapters
    persistence = SqlitePersistence(BRAIN_DIR / "brain.db")
    retrieval   = TagRetrieval(persistence)            # later: HybridRetrieval
    llm         = ClaudeAdapter()
    tools       = HARestToolExecutor(HAClient())
    clock       = SystemClock()

    # aggregates
    memory      = MemoryStore(persistence)
    episodes    = EpisodeStore(persistence)
    identity    = IdentityStore(persistence)
    user_state  = UserState(persistence)

    # services
    conversation = Conversation(
        memory, episodes, identity, user_state,
        retrieval, llm, tools, clock,
    )
    summarizer   = SessionSummarizer(llm, memory, episodes, clock)
    consolidator = Consolidator(memory, episodes, identity, llm, clock)
    consolidator.start()   # background thread; replays on crash via last_consolidated_at

    return ha_io.build_app(conversation)
```

The composition root is the only place adapters meet ports. Every other module receives its dependencies. Testing becomes: build a comp root with fake adapters.

---

## 8. What This Concretely Kills

| Current pain | Resolution |
|---|---|
| `brain.json` is one fragile blob | Four aggregate stores, separate tables, separate consistency boundaries |
| Provenance is "future work" | `source_episode_id` foreign key on every derived row; `correlation_id == episode_id` enforced |
| Sleep-time consolidation interleaves with conversation | `Consolidator` is a background thread reading aggregate contracts only |
| Embedding adoption requires touching everything | Swap one `main.py` line: `TagRetrieval` → `HybridRetrieval`. Backfill happens in background. |
| Adding bi-temporal timestamps requires refactor across 20 files | Already columns from day 1 |
| Test setup currently requires the whole server | Build a tiny comp root with in-memory fakes; test services in isolation |
| Training-runtime feature pipeline drift | `wakeword/contracts.py` is the single source; runtime config is generated from it |
| HA payload shape changes break cognition | Only `ha_io/contracts.py` + `_internal/voice_turn.py` care |
| Concrete library swap (anthropic → other) cascades | `external/claude_adapter.py` is the only file affected |
| Dead code rot | Boundary-check CI; `_internal/` is private; unused public surface is grep-visible |
| Multi-user home support is a future migration | `speaker_id` nullable from day 1 on `VoiceTurn` and `Episode` |

---

## 9. Out of Scope

Explicitly deferred from this restructure:

- **Tools-as-memory** (ToolCaching, MemTool patterns from arXiv 2601.15335, 2507.21428). Research is real but premature at Igor's scale. Today's `server/routines.py` (tool-call frequency logging by hour/day) is the precursor that will inform when this is worth promoting. Revisit when usage data justifies it.
- **Graph-database storage** (Neo4j, FalkorDB). Overkill at <1k entities, <100 distinct people. Bi-temporal columns on SQLite cover the same query needs at this scale.
- **MCP memory server** (Anthropic's memory tool spec). HA's MCP client doesn't yet support `resources` as of mid-2026; not actionable.
- **Multi-modal memory** (vision / image embeddings). Needs hardware (Pi AI Camera or equivalent) before software design is meaningful.
- **Speaker ID** itself. `speaker_id` is stamped nullable; populating it is a separate workstream involving resemblyzer or HA's eventual speaker-recognition primitive.

---

## 10. Implementation Phasing (preview — full plan in writing-plans output)

The implementation plan will sequence this as roughly:

1. **Scaffolding** — create the four context directories with `__init__.py` boundary guards. Boundary-check CI script. No behavior change.
2. **Contracts and ports** — write `cognition/contracts.py` and all five port protocols. Write `wakeword/contracts.py` and the runtime-render script. No adapters yet.
3. **SQLite persistence adapter + migration** — `SqlitePersistence`, schema creation, `brain_json_migration.py`. Run migration in dev; verify equivalent reads.
4. **Aggregates** — `MemoryStore`, `EpisodeStore`, `IdentityStore`, `UserState`. Unit-tested with `InMemoryPersistence` fake.
5. **External adapters** — `ClaudeAdapter`, `HARestToolExecutor`, `TagRetrieval`. Cognition services start to compose.
6. **Services** — `QualityGate`, `IntentRouter`, `ToolRegistry`, `Conversation`. Port the existing logic, change call sites to use aggregates + ports.
7. **ha-io** — `ConversationRequest/Response`, `voice_turn.py`, FastAPI endpoints. Replace current `server/api.py`.
8. **Background services** — `SessionSummarizer` (queue + drain thread), `Consolidator` (background thread + crash-replay).
9. **Composition root** — `server/main.py` wires everything; replaces current `server/main_text.py`.
10. **Cutover** — Docker rebuild, deploy, verify health, verify voice flow.
11. **Cleanup** — delete old monolithic modules, run boundary-check, commit.

---

## 11. References

- `MEMORY_ROADMAP.md` — Tier 1/2/3 prioritization that this design operationalizes
- Validation research (this session, 2026-06-17) — verified design against Letta, Mem0, Graphiti, Cognee, LangMem, plus 2026 arXiv papers 2502.06975, 2605.06716, 2511.17208, 2511.10523, 2606.09900 (Engram), 2601.15335 (ToolCaching), 2507.21428 (MemTool)
- Letta archival memory docs — `docs.letta.com/guides/core-concepts/memory`
- Letta "Our Next Phase" (March 16 2026) — `letta.com/blog/our-next-phase`
- Mem0 State of AI Agent Memory 2026 — `mem0.ai/blog/state-of-ai-agent-memory-2026`
- Graphiti / Zep paper — arXiv 2501.13956
- Engram (Jun 5 2026) — arXiv 2606.09900
- ConvoMem benchmark — arXiv 2511.10523
