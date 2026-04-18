# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Dev Session Workflow

At the start of every session, check `data/brain.json` for open feedback entries and proactively offer to fix them. Example: "You have 2 open items: response verbosity after light commands, and Sonos reconnect on stale cache. Want me to work through them?"

Read feedback with:
```python
from server.brain import init_brain; from server.config import BRAIN_FILE
brain = init_brain(BRAIN_FILE)
for e in brain.list_feedback("open"): print(f"#{e['data']['id']}: {e['data']['issue']}")
```

After fixing an item, call `brain.resolve_feedback(N)`, or remind the user to say "resolve feedback #N" to Igor.

## Project Overview

Igor is a local voice assistant. A **Raspberry Pi** handles audio I/O; a **PC** handles compute (STT, LLM, TTS).

- **STT**: Faster Whisper (`small` model, CPU)
- **LLM**: Claude API (`claude-haiku-4-5-20251001`)
- **TTS**: Kokoro ONNX (`am_onyx` voice, 24 kHz) — model files in `kokoro/`
- **Wake Word**: OpenWakeWord custom binary classifier (trained on recorded samples, no account)
- **Server**: FastAPI · **Client callbacks**: Flask

## Architecture

```
Pi (client/)                         PC (server/)
─────────────────────────────────    ─────────────────────────────
OpenWakeWord wake word detection →   /api/process_interaction
PyAudio VAD recording                  Whisper STT
Flask callback server           ←      Quality Gate (reject garbage)
  /api/play_audio                       Intent Router (Tier 1 direct)
  /api/hardware_control          →      Claude LLM (Tier 2+, tool_choice=auto)
  /api/play_beep                        Kokoro TTS
  /api/suppress_wakeword         →    /api/play_audio (timer alerts)
                                      /api/hardware_control (volume RPC)
Phone/text client           →        /api/text_interaction (no STT/TTS)
Any client at startup       →        /api/register (dynamic routing)

Multi-client routing:
  data/rooms.yaml            →  RoomConfig (devices per room)
  InteractionContext         →  flows through entire pipeline per request
  ClientRegistry             →  dynamic IP allowlist + callback routing
  RoomStateManager           →  per-room TV state, Sonos cache, TTS buffer

Living memory (background):
  Identity narrative         →  always in cached base prompt (<my_person>)
  Episodic memory            →  recent interactions in dynamic prompt
  Consolidation engine       →  regenerates identity every 5 episodes
  Knowledge gaps             →  14-slot profile schema, gaps in narrative
```

## Directory Structure

```
smart_assistant/
├── client/          # Raspberry Pi
│   ├── main.py      # Entry point + main loop
│   ├── wakeword.py  # OpenWakeWord detector
│   ├── audio.py     # PyAudio + beeps (sox)
│   ├── vad_recorder.py
│   ├── hardware.py  # ALSA volume
│   ├── pi_server.py # Flask callback server
│   ├── suppress.py  # Wake word suppression state (thread-safe)
│   ├── config.py
│   ├── pc_main.py   # PC client entry point (office endpoint)
│   ├── pc_audio.py  # Windows PyAudio wrapper
│   ├── pc_server.py # PC Flask callback server (:8081)
│   └── pc_config.py # PC-specific config (ports, devices, room)
├── server/          # PC
│   ├── main.py
│   ├── api.py       # FastAPI endpoints + rate limiter
│   ├── orchestrator.py  # STT→Gate→Router→LLM→TTS pipeline
│   ├── rooms.py     # RoomConfig dataclass, load_rooms(), make_default_room()
│   ├── context.py   # InteractionContext — per-request pipeline context
│   ├── client_registry.py  # Dynamic client registration + IP allowlist
│   ├── room_state.py  # Per-room mutable state (TV, Sonos, TTS buffer)
│   ├── quality_gate.py  # Post-STT filter (hallucinations, TV dialogue, garbage)
│   ├── intent_router.py # Tier 1 direct command matching (pause, lights off, etc.)
│   ├── transcription.py
│   ├── llm.py       # Claude API client (tool_choice=auto, ChatResult, 3-round max)
│   ├── synthesis.py # Kokoro TTS
│   ├── event_loop.py  # Timer thread (room-aware delivery)
│   ├── pi_callback.py # HTTP client → Pi
│   ├── speaker_id.py  # Resemblyzer speaker identification (optional)
│   ├── enroll_speaker.py  # CLI tool for enrolling speaker voice profiles
│   ├── pair_google_tv.py  # CLI tool for TV pairing (androidtvremote2)
│   ├── beeps.py       # Generate beep WAV files for Sonos output
│   ├── brain.py       # Unified Brain Store (memory, routines, feedback, reminders, episodes, identity narrative)
│   ├── routines.py    # Command usage pattern logging
│   ├── config.py
│   └── commands/    # Auto-discovered LLM tools
│       ├── base.py
│       ├── timer_cmd.py
│       ├── memory_cmd.py
│       ├── math_cmd.py
│       ├── time_cmd.py
│       ├── weather_cmd.py
│       ├── network_cmd.py
│       ├── system_cmd.py  # set_volume / adjust_volume / get_volume (RPC → Pi)
│       ├── lifx_cmd.py   # LIFX bulb control (local LAN, lifxlan)
│       ├── sonos_cmd.py  # Sonos volume control (local LAN, soco)
│       ├── tv_cmd.py     # Google TV power/nav (androidtvremote2)
│       ├── adb_cmd.py    # Google TV app launch/playback/search (adb-shell)
│       ├── feedback_cmd.py  # Change-request logging: log_feedback, list_feedback, resolve_feedback
│       └── _utils.py     # Shared: parse_amount, parse_direction_updown, parse_volume_word
├── shared/
│   ├── models.py    # Pydantic request/response models
│   ├── protocol.py  # Endpoint path constants
│   └── utils.py
├── oww_models/        # Trained .onnx wake word models (Pi + PC train step)
├── wakeword_samples/  # Positive samples for training (record on Pi)
│   └── positive/      # WAV files recorded via record_samples.py
├── onnx_models/wakeword_creation/train_wakeword.py  # PC training script
├── record_samples.py  # Pi recording script (run on Pi before training)
├── kokoro/            # Kokoro ONNX model files (PC only): kokoro-v1.0.onnx, voices-v1.0.bin
├── data/              # Persistent data: brain.json, benchmark.csv, rooms.yaml
│   └── rooms.yaml.example  # Template for multi-room configuration
├── mcp_server.py      # MCP server for Claude Code (commands + pipeline testing)
├── .mcp.json          # MCP server config (auto-loaded by Claude Code)
├── setup_client.sh    # Pi setup script (deps + OWW base model download)
├── setup_server.sh    # PC setup script (deps + voice download)
└── prompt.py          # LLM system prompt (Igor persona)
```

## MCP Testing Tools

`mcp_server.py` exposes Igor commands and pipeline testing tools to Claude Code.
Heavy models (Whisper, Kokoro) are lazy-loaded on first use.

| MCP Tool | What it does |
|----------|-------------|
| `list_commands` | List all auto-discovered voice commands |
| `run_command` | Execute any command by name + JSON args |
| `get_command_schema` | Return a command's parameter schema |
| `test_intent_router` | Probe Tier 1 routing — returns match/fallthrough |
| `test_quality_gate` | Probe quality gate — returns accept/reject + reason |
| `test_tts` | Synthesize text, report timing/cache hit/duration |
| `test_transcription` | Run Whisper on a WAV file, return per-segment confidence |
| `test_pipeline` | Full gate→router→LLM→TTS with timings per stage |
| `run_benchmark` | Batch test suite with CSV logging to `data/benchmark.csv` |
| `tail_logs` | Read recent server log lines with level filtering |

## Command System

Commands are auto-discovered from `server/commands/*_cmd.py`. Each is a subclass of `Command` (`base.py`) and becomes an LLM tool automatically.

```python
from server.commands.base import Command

class MyCommand(Command):
    name = "my_command"
    description = "What the LLM should know about when to call this"

    @property
    def parameters(self) -> dict:
        return {
            "thing": {"type": "string", "description": "The thing"}
        }

    def execute(self, thing: str) -> str:
        return f"did it: {thing}"
```

- `parameters` returns the **properties dict** (not the full schema — `to_tool()` wraps it)
- All parameters are required by default; override `required_parameters` property to return a subset
- No-parameter commands must use `execute(self, **_)` to avoid TypeError from command dispatch
- Hardware commands (volume) use `self.pi_client` injected at startup via `commands.inject_dependencies()`
- Room-aware commands accept `_ctx=None` parameter — injected automatically by `commands.execute()` via `inspect.signature`

| Command | Trigger example |
|---------|----------------|
| `get_time` | "What time is it?" |
| `calculate` | "15% tip on $47" |
| `set_volume` / `adjust_volume` / `get_volume` | "Your volume to 75" / "Turn yourself up" (RPC → Pi) |
| `set_sonos_volume` / `adjust_sonos_volume` / `sonos_mute` | "TV volume to 50" / "Turn the music up a bit" (Sonos) |
| `set_light` / `set_brightness` / `set_color` / `set_color_temp` | "Turn off the lights" / "Make the lights blue" (LIFX) |
| `adjust_brightness` / `adjust_color_temp` / `shift_hue` | "Slightly brighter" / "A lot warmer" (LIFX relative) |
| `list_lights` / `list_scenes` / `set_scene` | "What lights are on?" / "Set the movie scene" (LIFX) |
| `list_sonos` | "What Sonos speakers are there?" |
| `tv_power` | "Turn the TV on/off" (androidtvremote2) |
| `tv_key` | "Go home" / "Mute the TV" (androidtvremote2 nav keys) |
| `tv_launch` / `tv_playback` / `tv_skip` / `tv_search_youtube` | "Open YouTube" / "Pause" / "Skip 30 seconds" (ADB) |
| `tv_adb_connect` | "Test the TV connection" (ADB initial setup) |
| `save_memory` / `forget_memory` | "Remember I prefer dark roast" |
| `log_feedback` / `list_feedback` / `resolve_feedback` | "Log that" / "Show my change requests" |
| `set_timer` / `cancel_timer` / `list_timers` | "5 minute timer" |
| `get_weather` | "What's the weather?" |
| `scan_network` / `list_known_devices` / `pending_network_alerts` | "Scan for new devices" / "What's on my network?" |
| `delayed_command` | "Turn off the lights in 15 minutes" / "Mute the TV in an hour" |

## Key Configuration

**`data/rooms.yaml`** — room definitions (devices, lights, Sonos zones per room). Copy from `data/rooms.yaml.example`. If absent, auto-generates a default room from `server/config.py`.
**`server/config.py`** — update `PI_HOST` to your Pi's IP; `TRUSTED_IPS` env var for VPN/admin IPs
**`client/config.py`** — update `SERVER_HOST`, `AUDIO_DEVICE`, `OWW_THRESHOLD`; set `CLIENT_ID`/`ROOM_ID` for multi-client
**`shared/protocol.py`** — endpoint path constants only (no IPs)

Environment variables:
- `ANTHROPIC_API_KEY` (required) — server only
- `SERVER_HOST` / `PI_HOST` / `PI_PORT` — network defaults (192.168.0.4, .3, 8080)
- `SERVER_EXTERNAL_HOST` — PC's LAN IP for Sonos TTS fetch (default: 192.168.0.4)
- `TRUSTED_IPS` — comma-separated additional IPs (VPN, admin)
- `CLIENT_ID` / `ROOM_ID` — multi-client identity (default: "default")
- `GOOGLE_TV_HOST` — TV IP for ADB/remote (default: 192.168.0.20)
- `KOKORO_VOICE` / `KOKORO_SPEED` — TTS voice and speed (default: am_onyx, 1.0)
- `DEFAULT_LOCATION` — weather city (default: Arlington, VA)

No other API keys needed. Weather uses Open-Meteo (free, no account). Smart home uses local LAN only.

## Security Rules

- **Never `shell=True`** in subprocess calls — all commands use list args
- **Never log full transcriptions** — truncate to 50 chars max
- Pi server rejects requests from IPs other than `SERVER_HOST` (except `/api/health`)
- FastAPI rate-limits `/api/process_interaction` to 10 req/min per IP
- Hardware commands whitelisted in `shared/models.py` `HardwareControlRequest`
- All API inputs validated by Pydantic with size limits

## LLM / Conversation

- **Tiered pipeline**: STT → Quality Gate → Intent Router (Tier 1) → LLM (Tier 2+) → TTS
- **Quality gate** (`server/quality_gate.py`): rejects Whisper hallucinations, single non-command words, repetitive text, and long TV dialogue before LLM
- **Intent router** (`server/intent_router.py`): maps unambiguous short phrases ("pause", "lights off", "mute") directly to commands — zero LLM latency
- **LLM**: `tool_choice=auto`, no respond() tool, max 3 rounds. Action commands short-circuit with "Done." (1 API call); narrated commands (weather, timers) get a second call for LLM to read results
- `ChatResult(text, commands_executed)` return type from `llm.chat()`
- `NARRATED_COMMANDS` frozenset in `server/llm.py` controls which commands get a narration round
- `await_followup` heuristic in orchestrator: `endswith('?') and not commands_executed and len(words) < 20`
- History capped at `MAX_CONVERSATION_HISTORY` (10) messages
- `_trim_history()` ensures history always starts with a plain-text user message — tool_result orphans are dropped to avoid Claude API role errors
- Persistent memory stored in unified Brain Store (`data/brain.json`); behavior rules injected into cached base prompt, relevant memories into dynamic block
- Session summarizer runs after each non-follow-up turn to extract facts + episode via `LLM.analyze_conversation()` (single API call, zero extra cost)
- History overflow compresses dropped messages into `_history_summary` injected as `<prior_context>`

### Living Memory System

Three-tier memory architecture inspired by cognitive science (episodic/semantic/procedural) and MemGPT/Letta:

**Tier 1 — Identity Narrative (always in prompt, cached)**
- Living paragraph about the user, stored as single `identity` entry in brain.json
- Regenerated by consolidation engine when memories change (one Haiku call ~$0.001)
- Injected into base system prompt as `<my_person>` — part of Anthropic KV cache, zero extra token cost per interaction
- Solves the "forgotten name" problem: core identity is always visible to the LLM
- `BrainStore.get_identity_narrative()` / `update_identity_narrative()`

**Tier 2 — Episodic Memory (recent interactions)**
- Timestamped interaction summaries with emotional tone, topics, and commands
- Stored as `episode` entries in brain.json (capped at 200, oldest trimmed by `compact()`)
- Recent episodes injected into dynamic prompt as `<recent_episodes>`
- Enables "last time we talked about X" conversational continuity
- Generated by `LLM.analyze_conversation()` — combined with fact extraction in one API call

**Tier 3 — Semantic Memory (existing fact store)**
- Category/key/value facts in brain.json (unchanged)
- Tag-based retrieval for specific queries (`retrieve_relevant()`)
- Auto-extracted by session summarizer after each interaction

**Consolidation Engine (sleep-time processing)**
- Background process triggered every 5 episodes or when identity narrative is missing
- `LLM.generate_identity_narrative()` synthesizes all memories + recent episodes + knowledge gaps into narrative
- `Orchestrator._run_consolidation()` runs as daemon thread — never blocks interactions
- Also triggered at server startup in `main.py` if `brain.should_consolidate()` is True
- Marks processed episodes as consolidated to avoid re-processing

**Knowledge Gap Schema**
- `_PROFILE_SCHEMA` in `brain.py` defines 14 profile slots (name, birthday, relationships, preferences, schedule, home)
- `BrainStore.get_knowledge_gaps()` returns questions for unfilled slots
- Gaps embedded in identity narrative so LLM naturally knows what it doesn't know about the user
- Enables intelligent "ask about me" — fills gaps instead of re-asking stored facts

**Response Texture**
- `_CONFIRMATIONS` dict in `llm.py` — category-aware confirmation pools (light, volume, TV, timer, default)
- `_pick_confirmation(commands)` selects contextual response instead of always "Done."
- All confirmation variants pre-cached in TTS at startup for zero-latency delivery

## Multi-Client Architecture

- **Room config**: `data/rooms.yaml` defines devices per room (Sonos zone, TV host, light groups). Falls back to a default room from `server/config.py` if absent.
- **InteractionContext**: Created per-request in `api.py`, flows through the pipeline. Contains `client_id`, `room` (RoomConfig), `client_type`, `callback_url`, `prefer_sonos`, `tv_state`.
- **ClientRegistry**: Thread-safe dynamic client registration. Clients register at startup via `POST /api/register`. Replaces static `ALLOWED_CLIENT_IPS` for IP allowlisting.
- **RoomStateManager**: Holds per-room mutable state (TV playback, Sonos cache, TTS buffer). Starts per-room TV pollers for rooms with a `tv_host`.
- **Room-aware commands**: Commands that accept `_ctx` parameter use room defaults (light group, Sonos zone, Pi callback URL). Commands without `_ctx` work unchanged.
- **Text client**: `POST /api/text_interaction` — skips STT/TTS, returns text response. For phone/REST API clients.
- **Timer delivery**: Timers store `room_id`, delivered to the correct Pi via ClientRegistry lookup.
- **Backward compat**: All new fields default to `"default"`. Without `rooms.yaml`, everything works exactly as before.

### Adding a new room
1. Add room to `data/rooms.yaml` (copy from `data/rooms.yaml.example`)
2. Set `CLIENT_ID` and `ROOM_ID` env vars on the new Pi
3. Pi auto-registers with server at startup
4. Room-aware commands automatically use the new room's devices

## Interaction Flows

### Normal flow (wake word → response)
1. Pi detects wake word (`OWW_THRESHOLD` × `OWW_TRIGGER_FRAMES` consecutive frames)
2. RMS energy filter rejects low-energy detections (TV/room audio)
3. `_beep("start")` → routes per output config (see beep routing below)
4. VAD records until silence, sends WAV to server `/api/process_interaction`
5. `_beep("end")` on recording complete
6. Server: STT → Quality Gate → Intent Router / LLM → Kokoro TTS → response back
7. Pi plays local audio **or** server routes to Sonos (if `prefer_sonos_output=True`)
8. If `await_followup=True` (heuristic: response ends with `?`, no commands, <12 words, no filler): wait for TTS to finish, play start beep, listen again

### Beep routing (three modes)
```
USE_SONOS_OUTPUT=False:
  _beep(type) → local sox beep on Pi speaker (instant)

USE_SONOS_OUTPUT=True + INDICATOR_LIGHT set:
  _beep(type) → _sonos_beep(type) → POST /api/sonos_beep
    → LIFX flash (instant, ~50ms, silent)

USE_SONOS_OUTPUT=True + INDICATOR_LIGHT=None:
  _beep(type) → _sonos_beep(type) → POST /api/sonos_beep
    → Sonos play_uri beep WAV (1-3s startup lag)
```
**Typical interaction**: only start + end beeps. Error beep on failures. No done/complete signal — user knows interaction is done when TTS stops.

### TV-playing path
- `_last_tv_state` is polled every 5s via ADB (`dumpsys media_session`); up to 5s stale
- **Sticky state**: poller keeps last known good state on ADB failure — never overwrites with "unknown" (root cause of 2026-03-03 incident where assistant talked over TV for 5 min)
- `_get_tv_playback_state()` returns "idle" when ADB works but no media session found (distinct from "unknown" which means ADB failed)
- When playing: quality gate rejects long transcriptions (>40 words); non-critical TTS suppressed; `await_followup` forced False; session summarizer skipped; LLM context includes TV note
- `_is_critical_response()` skips the `?` check when TV is playing — questions from the LLM during TV playback are reactions to ambient audio, not user-critical info
- System prompt `<ambient_speech>` section teaches the LLM to recognize TV/media dialogue in transcriptions regardless of TV state detection
- After any TV command: `suppress_wakeword(20s)` sent to Pi → `client/suppress.py` blocks detection

### Wake word suppression
- Suppression state lives in `client/suppress.py` (module-level, thread-safe)
- Pi Flask server at `/api/suppress_wakeword` sets it; checked at 4 points: top of `_handle_interaction()`, during warmup loop, after detection confirmed, and inside detection loop
- Follow-up paths also check suppression before opening the mic

### Error beep rule
`self._beep("error")` everywhere (never `self.audio.beep_error()` directly) so routing respects `USE_SONOS_OUTPUT` and INDICATOR_LIGHT config.

## Wake Word (OpenWakeWord)

- Trained `.onnx` models live in `oww_models/` — produced by `train_wakeword.py` on the PC
- Base models (melspectrogram + embedding, ~50 MB) download automatically on first run
- Add a new wake word: record samples → train → drop `.onnx` in `oww_models/`
- `WakeWordDetector.predict()` returns `{model_stem: score}` per chunk (0–1 float)
- Tune sensitivity with `OWW_THRESHOLD` in `client/config.py` (default 0.5)
- Training workflow:
  1. Pi: `python record_samples.py`  → `wakeword_samples/positive/*.wav`
  2. PC: `scp -r user@<PI_IP>:~/smart_assistant/wakeword_samples/ wakeword_samples/`
  3. PC: `python onnx_models/wakeword_creation/train_wakeword.py`
  4. Pi: `scp oww_models/igor.onnx pi:smart_assistant/oww_models/`

## Wakeword Improvement Plan

**Quick wins (do now):**
1. Enable Silero VAD pre-filter — add `vad_threshold=0.5` to `Model()` in `wakeword.py`. Non-speech audio never reaches the classifier.
2. Train a custom verifier model — OWW v0.6.0 built-in speaker-specific logistic regression. Needs ~10s speech + false activation clips. Cuts 95% of false alarms.
3. Retrain with real negatives — already have hundreds of TV/speech/silence samples in `wakeword_samples/negative/`.

**Next (2-3 hours):**
4. Synthetic positive generation with Kokoro — generate thousands of "Igor" samples with varied voices/speeds, mix with noise/reverb. Biggest single accuracy boost per OWW community.

**Not worth pursuing:** Whisper-based wake word (too heavy for Pi), Snowboy (dead), SpeechBrain (research toolkit), always-on ASR, few-shot approaches. OpenWakeWord is the right engine — augment, don't replace.

## Roadmap — Future Features

### Sam's List - Don't touch
- learning routines from tool calls, not just speech
- automation choreograph ideas - "good morning"/"goodnight" (as early triggers for circadian rhythym calls), "sexy lightning", "ay carumba"
- bedtime reminders/nudges?
- speak directly into pc mic out
- searching online and adding notes to project files
- smart roomba integration?
- are pre-synthesized speech segments in a different model?
- better method for cleaning up our wakeword samples? all pc silences test as .98+

### "Alive House" Stack (build in order, compounds)
- [ ] Automation Choreography — "movie time" = dim lights + TV on + Sonos vol 30. `data/scenes.yaml`, Tier 1 intent routing, sequential step execution
- [ ] Proactive Intelligence — routines.py pattern detection + scheduler thread + Claude Haiku call (~$0.01/check). Max 2 proactive messages/hr, never during TV
- [ ] Room-to-Room Intercom — "tell the bedroom dinner's ready". TTS via Kokoro, deliver to target room's Pi/Sonos. Broadcast mode for all rooms

### Tier 1: High impact, buildable next
- [x] Reminders/scheduling — persistent reminders via Brain Store (timers survive restart)
- [x] Delayed commands — `delayed_command(command, params, delay)` schedules any device command to run after a delay. Uses timer callback. Whitelist in `_ALLOWED_DELAYED`.
- [x] PC client — `client/pc_main.py`, registers as office_pc/office, Flask callback on :8081, wake word + sample management. Run: `python -m client.pc_main`
- [ ] "Stop" wake word interrupt — second OWW model, playback interruption logic
- [ ] Spotify control (spotipy, needs free developer app registration)
- [ ] Calendar integration — Google Calendar API (read-only to start)
- [ ] Shopping/todo list — shared with phone (Todoist, Google Keep, or self-hosted)

### Tier 2: Learning and growing
- [ ] Emotional voice adaptation — librosa pitch/energy extraction, mood hint in system prompt, adaptive TTS speed
- [ ] Circadian lighting + soundscapes — auto color temp by time of day, ambient audio from Sonos
- [x] Behavioral adaptation — feedback resolved → behavior memory, injected into prompt at runtime
- [x] Richer memory model — unified Brain Store with typed entries, tag-based retrieval, contextual defaults
- [x] Living memory system — identity narrative, episodic memory, consolidation engine, knowledge gap schema, response texture

### Tier 3: Ambitious / transformative (hardware needed)
- [ ] mmWave presence + follow-me audio — $8/room (LD2410B + ESP32), music follows between Sonos zones, auto-lights on enter/leave
- [ ] Visual intelligence — Pi AI Camera ($70, on-chip inference), package detection, visitor recognition, OCR
- [ ] Streaming ASR — WebSocket audio chunks, latency drops from ~3s to ~1.5s (Deepgram Nova, Vosk, Moonshine)
- [ ] Web/API agent — browser or API calls for lookups, research, purchases
- [x] Multiple client support + bedroom Sonos as 2nd output
- [ ] Web dashboard for monitoring
- [ ] Entertainment host — trivia night, D&D dungeon master, scorekeeping, sound effects
- [ ] Puramax2 litterbox control

### Denied - Ideas proposed that I don't like
- [ ] Local LLM fallback — Ollama + Qwen3 4B, try/except on Claude API error, auto-recovery


## Quick Reference

```bash
# Setup
bash setup_server.sh   # PC
bash setup_client.sh   # Pi

# Start
python -m server.main   # PC
python -m client.main   # Pi

# Health
curl http://192.168.0.4:8000/api/health
curl http://192.168.0.3:8080/api/health

# Clear conversation
curl -X POST http://192.168.0.4:8000/api/conversation/clear

# Logs (systemd)
sudo journalctl -u igor-server -f
sudo journalctl -u igor-client -f
```