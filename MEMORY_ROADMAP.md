# Igor Memory System — 2026 Research Synthesis & Roadmap

Captured 2026-04-19 from a four-agent research session: (1) full audit of the current implementation, (2) 2026 state-of-the-art in LLM agent memory, (3) JARVIS-tier home-assistant capability gap analysis, (4) Home Assistant ecosystem context.

This is a decision/roadmap document, not a design spec. Pick it up when we return to memory work — Wyoming voice pipeline takes priority first.

---

## TL;DR

Igor's three-tier living-memory architecture (cached identity narrative + recent episodes + semantic facts + consolidation + knowledge gaps) **matches what the 2025/2026 literature converged on** — and is already ahead of every shipping Home Assistant LLM integration. The gap is not at the architecture level; it's at the **data model and retrieval layers**, which are stuck in 2023-era patterns (flat keyword-tag matching, no timestamps, no embeddings, no provenance, regeneration without reorganization).

Five low-effort additions take Igor from "decent" to frontier-class: **bi-temporal timestamps, semantic embeddings, provenance links, writes-off-hot-path, and sleep-time reorganization.** JARVIS-feel (anticipation, relationship awareness, temporal reasoning) layers on top cheaply afterwards.

---

## 1. Where Igor Stands Today

### Architecture (good — keep it)

```
Store:      data/brain.json (single JSON, tmp-rename atomic writes, thread-lock)
Tiers:      identity narrative    → always injected into cached system prompt
            recent episodes (5)   → injected into dynamic prompt per turn
            semantic facts        → category/key/value, tag-indexed
            routines              → command frequency by hour/day
            feedback              → change requests
            reminders             → persistent timers
Write:      LLM tool_choice=auto → save_memory (synchronous during turn)
            Background session_summarizer → analyze_conversation() extracts facts + episode
Retrieve:   <=50 memories total → inject all (small-scale safety)
            >50 → tag-overlap scoring, top 10
Consolidate: every 5 unconsolidated episodes, daemon thread regenerates identity narrative
            brain.compact() trims routines >1000, archives summaries >30d, caps episodes at 200
Schema:     _PROFILE_SCHEMA — 14 slots (name, birthday, job, partner, household, pets,
            music, food, hobbies, wake_time, sleep_time, work_schedule, home_layout,
            evening_routine). get_knowledge_gaps() returns unfilled slot questions,
            embedded into identity narrative so LLM naturally asks.
Current:    14 memories · 38 summaries · 56 routines · 5 reminders · 2 feedback · 1 identity
```

### What's good

- **Identity narrative in the cached system prompt** is exactly what ChatGPT Memory and Claude Memory do in 2026. This was right years before the field caught up.
- **Session summarizer + consolidation as background daemon threads** — non-blocking by design, matches Letta's sleep-time-compute pattern.
- **Knowledge gap schema** is unusual and valuable — it lets Igor know what he doesn't know and ask intelligently.
- **Atomic JSON persistence** with tmp-rename is cleanly implemented; single-writer assumption is safe for the current deployment.
- **Typed entry model** (memory, episode, identity, routine, feedback, reminder, summary) gives semantic structure without a schema registry.

### Gaps & rough edges

**Scale:**
- Magic threshold `_ALWAYS_ALL_THRESHOLD=50` switches retrieval strategies abruptly. Below 50 all memories inject (wastes tokens on irrelevant facts); above 50 jumps to keyword-scored (loses facts that don't share tokens with the query).
- Routine cap at 1000 deletes oldest chronologically — no recency-weighted pruning.
- Linear scans everywhere; ~100ms startup index rebuild at 1000+ entries.

**Retrieval:**
- Tag-based only. "I'm cold" won't retrieve "temperature" or "thermostat" unless exact keyword appears in tags. Compound words (`coffee_preference` vs `coffee`) cause misses. Typos fatal.
- No temporal reasoning. Episodes have timestamps but aren't used in scoring — old memories treated same as recent.
- No TF-IDF / weighting / normalization. Score is raw tag-overlap count.
- "Always-include" heuristic scores behavior + personal categories at 100, which prevents forgetting names but wastes tokens when the query is about weather.

**Cognitive capabilities missing:**
- No contradiction detection. "I wake at 8" and "I wake at 9" both stored. Dedup is only by exact (category, key) match — silent update clobbers the old value with no history.
- No relationship graph. "Ellie" is a people.girlfriend fact, but "where is Ellie" / "does Ellie like coffee" can't join against other Ellie-tagged facts.
- No temporal context on episodes beyond `emotional_tone`. No causality threading.
- No provenance. Facts don't link to the episode they came from — impossible to audit where a claim originated.
- Hardcoded profile schema. Can't personalize to different users' lives.

**Write path:**
- `save_memory` fires synchronously during the LLM turn (via tool_choice=auto). Adds latency and fights the primary/sleep-time-agent separation the field settled on.
- LLM decides when to save with almost no guardrails — category whitelist and length sanitization, nothing else. If prompt-injected, garbage lands in the store.

**Consolidation:**
- Regenerates identity narrative but doesn't *reorganize* the store. No merge-duplicates pass, no contradiction detection, no staleness demotion.
- No throttle. Could fire every 5 episodes with no floor on time-since-last-update.

**Process safety:**
- Single-process lock only. Multi-instance writes to `brain.json` would race. OK for current Pi5-container-only deployment.
- No versioning or rollback. brain.json corruption would lose memory; we rename `.bak.<timestamp>.json` on load-fail but no proactive rotation.
- `remove()` hard-deletes — no soft-delete, no audit log. Accidental tool call = gone.

---

## 2. The 2026 Field — What Serious Memory Systems Look Like

Surveyed Letta (formerly MemGPT), Mem0, Graphiti/Zep, Cognee, LangMem, ChatGPT Memory (reverse-engineered), Claude Memory + Auto-Dream, A-MEM (NeurIPS 2025), LightMem, ConvoMem, and the LongMemEval/LOCOMO/DMR benchmark ecosystem.

### Eight patterns showed up everywhere

1. **Primary agent + sleep-time agent split.** User-facing agent can't edit memory; a separate background agent does. Letta made this a first-class design pattern; Mem0 went async-by-default in v1.0.0; Claude Code calls it Auto-Dream; Cognee calls it Memify; LightMem calls it "offline consolidation." **This is the single most-agreed-upon architectural pattern in the field.**

2. **Add-only writes with temporal supersession at read time.** Mem0's 2025 reversal: drop the diff-against-existing write-time algorithm (slow and lossy). Just stamp timestamps and let read-time reasoning figure out current truth. Cut their write latency in half.

3. **Bi-temporal timestamps.** Graphiti's killer move. Every fact carries **four timestamps** — `t_valid`/`t_invalid` (when true in the world) plus `t_created`/`t_expired` (when we learned about it). Unlocks "we don't live there anymore" without deletion, and "what did I think was true last month" history queries. Paper: arXiv 2501.13956.

4. **Hybrid retrieval.** Semantic (embeddings) + keyword (BM25 / tag) + recency + optional graph hops. Pure semantic misses exact-match; pure keyword misses paraphrases. The winner is cheap hybrid.

5. **Provenance.** Every derived fact links to the episode it came from. Graphiti's episode subgraph is the canonical implementation. Critical for audit and — more importantly — critical against **false memory implantation** (see Risks).

6. **Sleep-time reorganization, not just regeneration.** Consolidation doesn't just produce a new summary — it merges duplicates, promotes high-recall facts, demotes stale ones, detects contradictions, indexes by new-discovered relationships. Claude Code's Auto-Dream is the most visible production instance.

7. **Identity paragraph always in the cached system prompt.** ChatGPT Memory does this, Claude Memory does this, Letta's core memory does this, Igor does this. **Validated by the field.**

8. **Don't over-engineer.** Letta's own 2025 benchmark found that agents using a **filesystem** beat Mem0's graph on LOCOMO (74.0% vs 68.5%) — because Claude has seen filesystems in training and uses them well, but doesn't know bespoke memory APIs. ConvoMem (arXiv 2511.10523) found **native context is sufficient below ~150 distinct conversations.** You probably don't need infrastructure.

### Frameworks at a glance

| System | Storage model | Write | Read | Consolidation | Headline claim |
|---|---|---|---|---|---|
| **Letta** | Core (in-context) / recall (history DB) / archival (vector) | LLM tool-calls (`memory_replace`, `memory_insert`, `archival_memory_insert`); **sleep-time agent only** | Core always visible, archival via `archival_memory_search` | Recursive summarization + sleep-time edits | Primary/sleep split; shareable memory blocks |
| **Mem0** | Flat facts + embeddings (+ optional graph variant Mem0^g) | Single-call extract, add-only, async-by-default | Vector + metadata filter | Implicit at read time | 26% accuracy gain + 91% latency reduction vs full context |
| **Graphiti/Zep** | Bi-temporal knowledge graph (Neo4j default) | Per-episode: NER → entity resolution → fact extraction → edge invalidation | Hybrid (cosine + BM25 + n-hop graph) with RRF/MMR rerank | Edge invalidation preserves history | 94.8% on DMR, 18.5% gain on LongMemEval, 115k→1.6k tokens |
| **Cognee** | Graph DB + vector DB + metastore | Cognify pipeline → entity/rel extraction → dual-indexed | Dual: vector + graph, LLM-merged | Memify post-processing pipeline | Ontology grounding, full provenance, multimodal |
| **LangMem** | Library primitives (profile + collection + procedural) | Memory Managers (extract/update/remove) + Prompt Optimizers | Semantic search on namespaced BaseStore | Application-specific | Framework-level, not a service |
| **Claude Memory** | Filesystem (`/memories/*`) — client-side | `view`/`create`/`str_replace`/`insert`/`delete` tools | Agent reads files on demand ("ALWAYS VIEW YOUR MEMORY DIRECTORY FIRST") | Auto-Dream: background topic-file reorganization | Simplest interface; agents use filesystems better |
| **ChatGPT Memory** | ~1,200-word User Memory + ~15 recent-chat summaries | Explicit "remember X" or heuristic extraction | Priority: current session → recent summaries → permanent facts | Undocumented | Long-term personalization outranks short-term context |

### Key research findings worth knowing

- **ConvoMem (arXiv 2511.10523)**: Native context works for personal assistants under ~150 distinct conversations. Igor is deeply under this. **Don't build vector infrastructure prematurely.**
- **Letta benchmarking blog (2025)**: Filesystem memory beats bespoke memory APIs on LOCOMO because Claude is pre-trained on filesystem tools. **Don't invent schemas Claude has never seen.**
- **A-MEM (NeurIPS 2025)**: Zettelkasten-inspired; each memory has 7 attributes, auto-generates links to neighbors on insert. ~2× gain over MemGPT on multi-hop, ~85% fewer tokens.
- **LightMem (arXiv 2510.18866)**: Atkinson-Shiffrin 3-stage (sensory pre-compression → short-term → long-term) with heavy consolidation offline. 32–117× token reduction, 1.67–12.45× speedup, 2.7–9.65% accuracy gain.
- **Survey "Memory for Autonomous LLM Agents" (arXiv 2603.07670)**: Clean 5-family taxonomy — context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, policy-learned management. Igor touches families 1, 2, 3. State of the art combines multiple.

### Where the hype is vs where the substance is

- **Substance**: primary/sleep-time split, bi-temporal modeling, add-only writes, reflective consolidation, provenance, identity-paragraph-in-prompt.
- **Hype**: graph-memory marketing vs Mem0's token-accounting rebuttal (7k vs 600k+ per store — contested). Benchmarks are **not comparable across papers** — LongMemEval and LOCOMO both allow different retrieval budgets; Mem0's own LOCOMO number moved 67%→92% between papers without methodology disclosure. Treat headline numbers as directional.
- **The unsolved problems** (Mem0's own 2026 report names these honestly): application-specific evaluation, privacy/consent architecture, cross-session identity resolution, and **memory staleness** (detecting when a high-relevance memory is confidently wrong). None of the frameworks solve these. **This is Igor's headroom.**

---

## 3. JARVIS-Tier Capabilities — What Would Actually Feel Alive

Strip the MCU iconography. What fans consistently cite about JARVIS is four properties: **distinct personality, continuous learning, anticipation, apparent theory of mind.** Igor already has personality. The other three are capability gaps.

### Cognitive primitives that matter for a home assistant

- **Prospective memory** — remembering to do a future thing without an external timer. "Remind me to talk to my sister before her birthday." Best implemented as **policies** (`WHEN condition THEN surface Y`) evaluated against HA state every minute, not as timers.
- **Source memory** — "who told me this, and when?" Prevents Igor from confidently attributing Sam's words to his spouse. Graphiti keeps an episode layer every derived fact traces back to. Same idea applies here.
- **Temporal reasoning** — "last Tuesday," "a while back," "before the kid's birthday." Flat vector stores can't do this. Needs bi-temporal facts + routine detection.
- **Theory of mind** — PNAS 2024 shows GPT-4-class models are at or above human on indirect requests, false beliefs, misdirection. Cheap implementation: have the system prompt carry a **rolling hypothesis about what the user wants right now**, updated each turn, and let Igor check it when unsure ("I think you're asking because…").
- **Emotional tagging** — empirical autobiographical-memory literature: emotional salience is what makes events retrievable. Igor already tags tone on episodes — **use it in retrieval scoring.**

### Relationship/social graph

Single biggest pragmatic gap in all commercial assistants. Alexa+ tracks the kid's soccer schedule but can't answer "when did I last talk to my sister." Gemini for Home identifies speakers but doesn't model relationships between household members.

A household-grade relationship model needs four things:

1. **Person nodes as first-class entities** (name, relation-to-user, voiceprint, birthday, last-contact, key facts).
2. **Typed edges** (lives-with, parent-of, works-with, pet-of). Resolves "the dog's vet appointment" to *which dog*.
3. **Per-speaker memory segregation** with privacy gradients — kid voiceprint by default can't read adult memories; guest voices open ephemeral sessions.
4. **Pets as objects, not people** — same schema, different type.

### Proactive patterns that make assistants feel alive

- **Passive routine detection.** You already log tool calls by hour/day. Layer a frequency table + weekly Haiku pass summarizing "things Sam seems to do regularly" → stored as usage observations. Resolves "the usual" / "like last Tuesday."
- **Context-bound triggers.** "When I get home after 7, remind me about laundry." Persistent rules evaluated against HA state (zones, devices, time-of-day).
- **mmWave presence.** ~$15/room (Aqara FP2). Biggest hardware-cheap unlock for "follow-me" audio/lighting and anticipatory lights. Zero LLM cost.
- **Observation loops.** Detect anomalies in routine (kid not home by 9:30, fridge not opened in 10h, kitchen light on all night).
- **Proactive surfacing engine.** A 20–30 min scheduler thread that runs one cheap Haiku call with context: *"anything worth surfacing right now?"* Hard caps: max 2/hour, none during TV, none to non-primary voices. This is where real anticipation lives.

---

## 4. Critical Risks

### False memory implantation

**arXiv 2408.04681 (2024)**: Conversational chatbots induced **3× more false memories than controls, 1.7× more than standard surveys, 36.8% persistence at one week.** Mechanisms: sycophancy, confirmatory feedback loops, repeat-with-embellishment.

If Sam says "remember I told you X?" and Igor confidently responds "yes, last Tuesday," Sam may now *believe* he told Igor X on Tuesday. This is the single most important risk to engineer against.

**Mitigations (non-negotiable):**
- Never confabulate "yes you told me X" without a retrieved memory with matching provenance.
- Surface confidence when recall is fuzzy: "I don't have that in my notes — is this new?"
- Expose a `what do you know about me?` tool that returns inspectable facts with episode provenance.
- One-command forget for anything surfaced.

### "Creepy" zone

**CHI 2023 creepy-assistant four-factor scale**: control, privacy, behavior, value. Continuous monitoring hits visceral creep harder than online tracking. The uncanny "nearly-human voice with subtle flaws" effect is more unsettling than clearly synthetic voices.

**Mitigations:** explicit reveal commands, one-command forget, visible listening indicators, no unsolicited personal-fact references early in relationships.

### Memory drift / context rot

Stuffing more context is **actively bad** per Hagoel's 2025 field guide and the "Can an LLM Induce a Graph" paper (arXiv 2510.03611): even o1-class models start losing relational coherence well before context limits.

**Mitigation**: token budget per-section in the dynamic prompt. Cap `<relevant_memories>` at N tokens. Better to retrieve fewer but more relevant facts than dump everything.

### Catastrophic forgetting

Often cited but **doesn't apply** to Igor. Catastrophic forgetting is a weights-update problem. Igor's memory is entirely external (in-prompt + in-store). Don't fine-tune Claude; keep memory external. Filed here only to dismiss it.

---

## 5. Home Assistant Ecosystem Context

### HA core will not solve long-term memory in 2026

Paulus explicitly positioned memory as an integration concern in architecture discussion #1068. Voice Chapters 10 (June 2025) and 11 (October 2025) and the Sept 2025 "AI in HA" blog mention memory zero times as a pillar. HA's `chat_session` helper has a hardcoded `CONVERSATION_TIMEOUT = 5 minutes` and no disk persistence.

**This is a moat for Igor, not a liability.** Every official HA LLM integration (OpenAI, Anthropic, Google, Ollama, OpenRouter) has **no cross-session memory**. Igor is already lapping them.

### What HA gives for free — consume, don't duplicate

- **Area/Floor registry** — spatial memory.
- **Device registry** — what's where, by ID.
- **Person domain + device_tracker** — who's home (not who's speaking).
- **History / Logbook API** — 10-day state history; queryable as "the house's memory."
- **Todo entities** — legitimate episodic-reminder surface; user can browse.
- **Calendar entities** — LLM Vision integration already uses these as memory-of-events.
- **`.storage/<domain>.memory.<entry_id>`** — convention for integration-owned JSON (Google ADK uses this).

### What HA doesn't give you — bring your own

- **Speaker identity.** `ConversationInput` carries `device_id` but not `user_id` (usually) and no voice-ID. Igor already has resemblyzer; keep it as the authoritative user-mapping layer until HA ships speaker ID natively.
- **Memory persistence across conversations.** 5-min `chat_session`, no disk. Own this completely.
- **Per-user privacy gradients.** No first-class model. Build on top of voice-ID.

### Pattern to converge on

- Hook Igor's consolidation to `chat_session.async_on_cleanup(cb)` — fires on the 5-min idle expiry. Closest thing HA gives to a "conversation ended" signal.
- Migrate `brain.json` into `.storage/igor.memory.<entry_id>` per ADK pattern.
- Expose HA-native surfaces as tools — `get_history`, `add_to_todo`, `get_calendar_events` — rather than shadowing state in Igor's own store.
- MCP memory server: **not yet** — HA's MCP client doesn't support `resources` as of 2026.4, only `tools`.

---

## 6. Prioritized Roadmap

Ordered by **user-value-delta / hours-of-work**. Each item is independently verifiable.

### Tier 1 — Foundation (biggest wins, do first)

1. **Per-fact bi-temporal timestamps.** Add `valid_at`, `invalid_at`, `created_at` (`created` already exists) to every memory entry. At read time, inject active facts normally; inject superseded facts with a `[was: until 2026-03-12]` marker so the LLM can reason about history. **Unlocks:** "we don't live there anymore" without deletion, "what did I think was true last month" queries, true contradiction handling. **Effort:** ~1 day.

2. **Semantic embeddings alongside tags.** Ship a tiny local model (`bge-small-en-v1.5` or `all-MiniLM-L6-v2`, ~100MB CPU-inference). Embed each memory value on save; embed queries at retrieval. **NumPy cosine over a list — no vector DB needed at current scale** (ConvoMem validates this well below 150 conversations). Hybrid score: semantic similarity + tag overlap + recency decay. **Unlocks:** "I'm cold" → thermostat facts, paraphrase robustness. **Effort:** ~1 day.

3. **Provenance links.** Every fact gets an `episode_id` reference (or `null` if entered manually). When Igor surfaces a fact, he can quote "you said this on 2026-03-12 after dinner." **Unlocks:** inspectable memory, false-memory defense, audit. **Effort:** ~half-day.

4. **Move all memory writes off the hot path.** `save_memory` currently fires synchronously during the LLM turn via tool_choice=auto. Queue writes into the same daemon thread the session summarizer already uses. Matches Letta's primary/sleep-time-agent pattern. **Unlocks:** reduced turn latency, cleaner separation of concerns. **Effort:** ~1 day.

5. **Sleep-time reorganization during consolidation.** Today `_run_consolidation` only regenerates the identity narrative. Add a pass that: merges exact-duplicate facts, detects contradictions (same category+key, different values → mark older as superseded), demotes stale facts (last-recalled >90 days ago + contradicted), promotes high-recall facts (recalled within last 7 days → boost retrieval score). One Haiku call. **Unlocks:** staleness handling, the core "sleep-time-compute" win. **Effort:** ~2 days.

**Tier 1 total: ~5–7 days.** After this, Igor is firmly in "frontier" territory per the survey taxonomy.

### Tier 2 — JARVIS-adjacent (modest build)

6. **First-class `person` entity type with relation edges.** Add `person` as an entry type; add `relation` as a separate type linking person-IDs with relation labels (spouse, parent-of, pet-of). No graph DB — typed records + lookup by name resolve "the sister who cried last month." **Effort:** ~2 days.

7. **Context-bound policy engine.** Let LLM emit persistent rules of the form `{condition: "area=home AND time>=18:00 AND weekday", action: "surface laundry reminder", expires: null}`. Tiny scheduler thread evaluates rules against HA state every minute. **Unlocks:** "when I get home, remind me to…" done right. **Effort:** ~3 days.

8. **Passive routine detection.** Daily Haiku pass over the existing `routine` entries: "summarize patterns in the last 30 days → what does Sam consistently do and when?" Store as usage observations. **Unlocks:** "the usual" / "like last Tuesday." **Effort:** ~1 day.

9. **Emotion-weighted retrieval scoring.** Episodes already have `emotional_tone`. Add it as a scoring term — boost high-salience episodes. Recall tags also used for hit-counting. **Effort:** ~half-day.

10. **Memory review tool.** `what_do_you_know_about(person_or_topic)` returns inspectable list with provenance links. `forget_memory` already exists. **Unlocks:** false-memory mitigation, creepy-zone reduction, user trust. **Effort:** ~half-day.

11. **Proactive surfacing engine.** Scheduler thread every 20–30 min, one Haiku call with context (time, presence from HA `person` domain, recent episodes, open reminders, calendar events). Hard caps: max 2/hour, none during TV playback, none to non-primary voices. **Unlocks:** anticipation — the thing that makes it feel alive. **Effort:** ~3 days.

**Tier 2 total: ~10 days.**

### Tier 3 — HA alignment (ongoing, as migration settles)

12. Move brain.json to `.storage/igor.memory.<entry_id>` — HA-native persistence pattern.
13. Hook consolidation to `chat_session.async_on_cleanup` (5-min idle callback) in addition to the every-5-episodes rule.
14. Expose HA-native memory surfaces via tools: `get_history`, `add_to_todo`, `get_calendar_events`.
15. Keep resemblyzer-based voice-ID as user-mapping layer until HA ships speaker ID natively.

### Explicitly NOT to build (yet)

- **Full Neo4j / Graphiti graph DB.** Overkill at household scale (~10 person nodes, under 1000 facts). Revisit at 500+ facts / 20+ distinct people.
- **MCP memory server.** HA's MCP client doesn't support `resources` as of 2026.4.
- **Fine-tuning / LoRA adapters for memory.** Keep memory external; Claude is good enough.
- **Multi-modal / vision memory.** Needs Pi AI Camera hardware first.
- **Federated cross-device memory.** Current single-brain works for the whole house; room-aware InteractionContext covers the 90%.

### Benchmark to build alongside

Before Tier 1 lands, write a tiny LongMemEval-style evaluation set specific to Igor: ~50 questions split across single-hop, multi-hop, temporal, knowledge-update, and abstention categories. Example: *"When did I last mention my sister?"* / *"What's my schedule like on Fridays?"* / *"Do I still work at [old job]?"* / *"What don't you know about me?"*

Without a measuring stick every change is vibes. Target: aim for >85% single-hop, >70% multi-hop, >90% abstention accuracy.

---

## 7. Tier 1 Implementation Sketches

Pre-positioning the shape of the code for when we return.

### 7.1 Bi-temporal timestamps — schema change in `brain.py`

```python
# BrainEntry.data for type=memory gains:
{
    "category": "preferences",
    "key": "coffee",
    "value": "dark_roast_oat_milk",
    "valid_at": "2026-03-01T00:00:00Z",    # when true in the world
    "invalid_at": null,                      # null = currently valid
    # existing: created, updated (transaction time already present)
}

# retrieve_relevant() changes:
#   - Return (active, superseded) tuple
#   - Format superseded as "[was true until 2026-03-12] <fact>"
#   - LLM sees both; can answer "what did I used to drink" AND "what do I drink"

# save_memory() changes:
#   - On update: set old entry's invalid_at = now, create new entry
#   - Don't clobber — preserve history
```

### 7.2 Embeddings — new file `server/embeddings.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at startup — ~80MB, ~50ms per embed on Pi5 CPU
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(text: str) -> np.ndarray:
    return _model.encode(text, normalize_embeddings=True)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # pre-normalized → cosine = dot product
```

```python
# brain.py — add embedding column to memory entries
entry["data"]["embedding"] = embed(value).tolist()

# retrieve_relevant() — hybrid score:
query_emb = embed(query)
for entry in memory_entries:
    sem_score = cosine(query_emb, np.array(entry["data"]["embedding"]))
    tag_score = len(query_tokens & set(entry["tags"])) / max(len(query_tokens), 1)
    recency = exp(-age_days / 30)    # half-life 30 days
    total = 0.5 * sem_score + 0.3 * tag_score + 0.2 * recency
```

### 7.3 Provenance — `episode_id` on every derived fact

```python
# orchestrator.py _run_session_summarizer:
#   1. Call llm.analyze_conversation() → gets facts + episode
#   2. brain.add_episode(...) → returns episode_id
#   3. For each extracted fact: brain.save_memory(cat, key, val, source_episode_id=episode_id)
#      (new kwarg; stored in entry.data.source_episode_id)
```

### 7.4 Writes off hot path — queue + background drain

```python
# server/memory_queue.py
_queue: queue.Queue = queue.Queue()

def enqueue_save(category, key, value, source_episode_id=None):
    _queue.put(("save", category, key, value, source_episode_id))

def _drain_loop():
    while True:
        op, *args = _queue.get()
        if op == "save":
            brain.save_memory(*args)
        # ... other ops (forget, etc.)

threading.Thread(target=_drain_loop, daemon=True).start()

# server/commands/memory_cmd.py SaveMemoryCommand.execute:
#   replace brain.save_memory() call with enqueue_save() → returns immediately
```

### 7.5 Sleep-time reorganization — new method on BrainStore

```python
def reorganize(self, llm: LLM) -> None:
    """Run during consolidation. Sleep-time reorganization."""
    # 1. Merge exact duplicates (same category+key+value → keep newest)
    # 2. Detect contradictions (same category+key, different values)
    #    → ask LLM to reconcile; mark older as invalid_at=now
    # 3. Demote stale (last_recalled > 90d, low recent-hit count)
    # 4. Promote recalled (last_recalled < 7d → boost retrieval_boost)
    # Single Haiku call for the contradiction-reconciliation step.
```

---

## 8. Key Sources

**Frameworks:**
- Letta — https://www.letta.com/blog/agent-memory · https://www.letta.com/blog/sleep-time-compute · https://www.letta.com/blog/benchmarking-ai-agent-memory
- Mem0 — https://arxiv.org/abs/2504.19413 · https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm · https://mem0.ai/blog/state-of-ai-agent-memory-2026
- Graphiti / Zep — https://arxiv.org/abs/2501.13956 · https://github.com/getzep/graphiti
- Cognee — https://github.com/topoteretes/cognee · https://www.cognee.ai/blog/cognee-news/product-update-memify
- LangMem — https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
- Claude Memory / Auto-Dream — https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool · https://claudefa.st/blog/guide/mechanics/auto-dream

**Research:**
- A-MEM (NeurIPS 2025) — https://arxiv.org/abs/2502.12110
- LightMem — https://arxiv.org/html/2510.18866v1
- ConvoMem — https://arxiv.org/pdf/2511.10523
- Survey: Memory for Autonomous LLM Agents — https://arxiv.org/html/2603.07670v1
- False Memories in Witness Interviews — https://arxiv.org/html/2408.04681v1
- Can an LLM Induce a Graph? (drift) — https://arxiv.org/html/2510.03611v1
- LOCOMO Benchmark — https://snap-research.github.io/locomo/

**Home Assistant:**
- Conversation entity API — https://developers.home-assistant.io/docs/core/entity/conversation/
- HA LLM API — https://developers.home-assistant.io/docs/core/llm/
- Voice Chapter 10 — https://www.home-assistant.io/blog/2025/06/25/voice-chapter-10/
- Voice Chapter 11 — https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11/
- Architecture discussion #1068 (memory as integration concern) — https://github.com/home-assistant/architecture/discussions/1068
- Google ADK (`.storage` pattern) — https://github.com/allenporter/home-assistant-google-adk
- hass-agent-llm (ChromaDB + extraction) — https://github.com/aradlein/hass-agent-llm

**Ecosystem:**
- ChatGPT Memory reverse-engineered — https://llmrefs.com/blog/reverse-engineering-chatgpt-memory
- Simon Willison on ChatGPT's 2025 memory shift — https://simonwillison.net/2025/May/21/chatgpt-new-memory/
- Supermemory — https://supermemory.ai/research/
- CHI 2023 Creepy Assistant scale — https://dl.acm.org/doi/full/10.1145/3544548.3581346
- PNAS ToM in LLMs — https://www.pnas.org/doi/10.1073/pnas.2405460121
