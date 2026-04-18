# Igor → HA Voice Pipeline + Pi5 Docker Container

Started 2026-04-18. Delete this file when migration is complete and the new state is reflected in CLAUDE.md.

> **This is a rewrite.** The earlier "LXC + HA action delegation" plan was scaffolding — superseded once we decided Igor should also drop STT/TTS and become a pure Custom Conversation Agent. The Proxmox LXC at `10.0.30.20` is no longer the destination; tear it down at the end.

## Why

1. **Igor stays the brain, drops everything else.** HA owns the entire voice pipeline (wake word, STT, TTS, audio routing, device control). Igor becomes a stateless-ish text-in/text-out service: receive transcribed user text from HA, return response text + tool calls.
2. **Drastic simplification.** Roughly 40-50% of the codebase deletes. No more `client/`, no Whisper, no Kokoro, no wake word training infra, no ALSA/PyAudio, no Pi-specific code, no PC client, no beep generation, no audio playback callbacks.
3. **Igor as a Pi5 Docker container.** Per ADR-002, this is exactly the workload tier for "always-on, low-power, stateless services managed by Portainer." After paring down, Igor's working set is ~256MB / 0.5 core idle. ARM-compatible (pure Python: anthropic SDK + FastAPI).
4. **Cheap voice satellites.** Existing Pi runs `wyoming-satellite` (with our trained `igor.onnx` for wake word — that work is preserved). HA Voice PE drops in later for new rooms; zero Igor code changes when it does.

## Final Architecture

```
[Voice satellite: existing Pi running wyoming-satellite + igor.onnx wakeword]
        ↓ Wyoming protocol
[Home Assistant @ 10.0.40.5]
   ├─ Wyoming-faster-whisper add-on (STT)
   ├─ Wyoming-piper add-on (TTS)
   ├─ Conversation agent → Igor (HTTP POST /conversation/process)
   └─ Audio output → media_player.living_room (Sonos) or satellite speaker
                ↑
[Pi5 Docker: igor container — text in, text out, mounted brain.json volume]
```

## Architectural Decisions (locked in this session)

- Igor is a **HA Custom Conversation Agent** — endpoint accepts `{text, conversation_id, device_id, language}`, returns `{response, end_conversation}`.
- **No audio anywhere in Igor.** STT/TTS are HA's job (Wyoming-faster-whisper + Wyoming-piper add-ons). We accept losing Kokoro `am_onyx`; user is fine with Piper voice.
- **Wake word stays ours**: existing `igor.onnx` (OpenWakeWord) loads into `wyoming-openwakeword` on the satellite. The wake word training pipeline (`record_samples.py`, `onnx_models/wakeword_creation/`) stays in the repo for future retraining.
- **Igor runs as a Docker container on Pi5**, not the LXC. ARM64 image, alongside Portainer / Uptime Kuma. `brain.json` lives in a bind-mounted volume on the Pi5 host so it survives container rebuilds.
- **HA Areas are the source of truth for room → entity mapping.** `rooms.yaml` shrinks (or disappears entirely — TBD if it has any value once HA owns rooms).
- **Per-room context**: HA passes `device_id` in the conversation request → Igor maps to `ha_area` → uses for memory/context (e.g., "the office" reference) and for any room-scoped commands.
- **Quality gate stays.** Even with HA's STT, hallucination/garbage filtering is still useful pre-LLM.
- **Intent router stays.** Tier-1 short-phrase commands save tokens and latency.
- **All device action commands**: thin HA REST calls. `light_cmd`, `media_cmd`, `tv_cmd`, `todo_cmd` (NEW), `notify_cmd` (NEW). One `ha_client.py` underneath.

## What Gets Deleted

| Path | Reason |
|---|---|
| `client/` (entire dir — main, wakeword, audio, vad_recorder, hardware, pi_server, suppress, pc_main, pc_audio, pc_server, pc_config) | HA + wyoming-satellite owns the audio side now |
| `server/transcription.py` | HA does STT |
| `server/synthesis.py` | HA does TTS |
| `kokoro/` (model files) | No more local TTS |
| `server/pi_callback.py` | No Pi to call back to |
| `server/event_loop.py` (audio delivery parts; keep timer scheduling logic if reused) | HA handles timer alert audio via notify/media_player |
| `server/beeps.py`, `server/commands/beep` logic | HA pipeline handles audio cues |
| `server/commands/lifx_cmd.py`, `sonos_cmd.py`, `tv_cmd.py`, `adb_cmd.py` (replaced by HA-backed versions) | HA integrations cover all of this |
| `server/commands/system_cmd.py` (volume RPC bits) | HA media_player owns volume |
| `server/pair_google_tv.py`, `enroll_speaker.py` | TV pairing → HA. Speaker ID → reconsider after migration. |
| `record_samples.py` (move to `tools/`?) | Still useful for wake word retraining; relocate but don't delete |
| `setup_client.sh`, `setup_server.sh` | Replaced by Dockerfile + docker-compose |
| `mcp_server.py` audio test tools | Test the conversation endpoint directly instead |
| `data/rooms.yaml` (likely; possibly slim version remains) | HA Areas replaces it |
| Deps: `lifxlan, soco, androidtvremote2, adb-shell, faster-whisper, kokoro-onnx, openwakeword, pyaudio, sounddevice` | Server-side deletes; satellite Pi keeps openwakeword via wyoming-openwakeword (separate package) |

## Sequenced Plan

Order matters: each step independently verifiable. A bug at step N shouldn't make us doubt N-1.

### Step 1 — Refactor Igor to Conversation-Agent shape

Goal: same Igor brain, but with a single `POST /conversation/process` endpoint replacing the audio pipeline. Run locally or on the LXC for fast iteration.

- [ ] Add `POST /conversation/process` to `server/api.py` accepting HA's payload shape (`{text, conversation_id, device_id, language}`) and returning `{response, end_conversation, conversation_id}`
- [ ] Refactor `Orchestrator` so its core method is `process_text(text, ctx) -> ChatResult` — the existing `process_audio` becomes thin wrapper (will be deleted)
- [ ] Build `device_id → ha_area → InteractionContext` resolver. HA device_ids are stable UUIDs; cache the mapping at startup (refresh on demand).
- [ ] Replace device commands with HA-backed versions:
  - [ ] `server/ha_client.py` — auth, GET `/api/states`, POST `/api/services/{domain}/{service}`, area templating
  - [ ] `light_cmd.py` (replaces `lifx_cmd`)
  - [ ] `media_cmd.py` (replaces `sonos_cmd` + Sonos parts of `system_cmd`)
  - [ ] `tv_cmd.py` (replaces both `tv_cmd` and `adb_cmd` — uses `remote.send_command` + `media_player.*`)
  - [ ] `todo_cmd.py` — NEW. `add_todo`, `list_todo`, `complete_todo`. Uses HA `todo` domain.
  - [ ] `notify_cmd.py` — NEW. `notify(message, target, title)`. Uses HA `notify.*`.
- [ ] Adapt timer alerts (`event_loop.py`) to deliver via HA — `notify` for ambient, `media_player.play_media` for room audio
- [ ] Keep: `quality_gate`, `intent_router`, `llm`, `brain`, `routines`, `feedback_cmd`, `memory_cmd`, `time_cmd`, `weather_cmd`, `math_cmd`, `network_cmd`, `delayed_command`
- [ ] Delete: everything in **What Gets Deleted** above
- [ ] Update `prompt.py` — the persona stays, but examples about audio/Sonos/etc. should reference how HA exposes things now
- [ ] Tests: unit-test the conversation endpoint with a synthetic HA payload + mocked HA client

### Step 2 — Containerize and deploy to Pi5

- [ ] Write `Dockerfile` (multi-arch base: `python:3.12-slim`, `--platform=linux/arm64`)
- [ ] Write `docker-compose.yml`:
  - bind mount `./data:/app/data` (brain.json persists outside container)
  - env vars: `ANTHROPIC_API_KEY`, `HA_URL=http://10.0.40.5:8123`, `HA_TOKEN` from EnvironmentFile or Docker secret
  - port mapping: `8000:8000`
  - `restart: unless-stopped`
- [ ] Build image — either on Pi5 (`docker build`) or push from LXC/dev box via `docker buildx build --platform linux/arm64`
- [ ] Copy current `data/brain.json` from PC to Pi5 host data volume — preserves identity narrative, episodes, feedback, routines
- [ ] Deploy via Portainer (Pi5)
- [ ] DNS entry on Pi-hole: `igor.kingdahm.com → <Pi5 IP>`
- [ ] Add to Uptime Kuma — health check `GET /api/health`
- [ ] Generate fresh HA long-lived token from `http://10.0.40.5:8123/profile/security`, store in Pi5's secret mechanism

### Step 3 — HA voice pipeline add-ons

- [ ] HA → Settings → Add-ons → install **Whisper** (Wyoming-faster-whisper); pick `small-int8` to match what we had
- [ ] Install **Piper** (Wyoming-piper); choose voice (suggest `en_US-ryan-high` or browse — listen at piper-voices preview)
- [ ] HA → Settings → Voice assistants → Create pipeline:
  - STT: Whisper add-on
  - TTS: Piper add-on
  - Conversation agent: **Igor** (added in step 4)

### Step 4 — Register Igor as HA Custom Conversation Agent

- [ ] In Igor: implement HA's conversation agent contract (response shape, intent matching for HA's expected format)
- [ ] In HA: add Igor's URL as a custom conversation agent. Two paths:
  - Option A: Igor exposes a HACS-installable HA integration (cleanest, but more work)
  - Option B: Use HA's built-in OpenAI Conversation integration pointed at Igor (Igor implements the OpenAI-compatible chat completions endpoint — fastest path, just one endpoint to mock)
- [ ] Set the new pipeline's conversation agent to Igor
- [ ] Test from HA UI: Settings → Voice assistants → Pipeline → "Try it" — type a sentence, confirm Igor responds

### Step 5 — Install wyoming-satellite on existing Pi

- [ ] On the Pi: install `wyoming-satellite` + `wyoming-openwakeword`
- [ ] Copy `oww_models/igor.onnx` to satellite's wakeword model dir
- [ ] systemd unit for `wyoming-openwakeword` (loads our model)
- [ ] systemd unit for `wyoming-satellite` (mic + audio out config, points at HA)
- [ ] HA → Settings → Devices & Services → Wyoming → discovers satellite, assign to Area
- [ ] Verify: say "Igor", speak a command, response plays through Sonos (or Pi speaker, depending on output config)

### Step 6 — Tear down old infra

- [ ] Stop Igor on user's PC (the original deployment)
- [ ] Tear down the LXC at `10.0.30.20` on Snap (was scaffolding only)
- [ ] Decommission existing Pi as Igor client (now repurposed as wyoming-satellite — same hardware, new role)
- [ ] Update `wirenest` services-registry: HA marked Running, Igor (Pi5 Docker) added, LXC removed
- [ ] Update `wirenest` adr-002 if needed

### Step 7 — Voice PE (when it arrives)

- [ ] Plug in PE, complete onboarding wizard via HA
- [ ] Assign to Area
- [ ] Either retire the wyoming-satellite Pi or keep both for multi-room
- [ ] No Igor code changes needed

## Open Questions / TODOs

- **OpenAI-compatible vs HACS integration** for the conversation agent — Option B (OpenAI shim) is faster to ship; Option A (HACS) is cleaner long-term. Recommend B for MVP.
- **`rooms.yaml` fate**: probably deleted entirely once HA Areas drive everything; only survives if we need per-area output preferences (Sonos vs satellite speaker) that HA doesn't model directly.
- **Speaker identification (`server/speaker_id.py`)**: keep or delete? It was always optional. HA's `person` entity covers "who's home" but not "who just spoke." Defer.
- **Behavior memory / consolidation**: still runs as a background daemon thread inside the container. Confirm it works fine in a slim Docker setup (no display, no audio).
- **Timer audio**: HA `notify` for phone push, `media_player.play_media` for room speakers. Replace `event_loop.py` Pi callback mechanism.
- **Wake word retraining workflow**: `record_samples.py` no longer runs on the original Pi (it's wyoming-satellite now). Keep training script, document new sample collection path (probably from satellite's recordings + HA's logged audio).

## Where We Left Off (2026-04-18)

Done this session:
- Trained new wakeword model with template-matched windowing — committed `6c1631d`
- Stood up Proxmox LXC at `10.0.30.20` (now scaffolding — will tear down at Step 6)
- Cloned repo into LXC, ssh keys set up, GitHub deploy key added
- Verified HA live at `10.0.40.5`, enumerated entities (4 lights, 2 TVs, Sonos, switches, person, device_tracker)
- Architecture pivoted: HA owns voice pipeline; Igor pares down to conversation agent on Pi5 Docker

Not started:
- Step 1 — code refactor. Best to start here next session, can work from any dev env.
