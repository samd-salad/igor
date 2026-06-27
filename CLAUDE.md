# CLAUDE.md

<!-- doc-verified: 2026-06-26 -->

Guidance for Claude Code when working in this repository.

## Dev Session Workflow

At the start of every session, check `data/brain.json` for open feedback entries and proactively offer to fix them:

```python
from server.brain import init_brain
from server.config import BRAIN_FILE
brain = init_brain(BRAIN_FILE)
for e in brain.list_feedback("open"):
    print(f"#{e['data']['id']}: {e['data']['issue']}")
```

After fixing an item, call `brain.resolve_feedback(N)`, or remind the user to say "resolve feedback #N" to Igor.

## Project Overview

Igor is a **text-in / text-out conversation agent** running as a Home Assistant Custom Conversation Agent. HA owns the entire voice pipeline (wake word, STT, TTS, device control); Igor is just the brain.

- **Host**: Pi5 at `10.0.30.5` as a Docker container on port `4467`
- **LLM**: Claude API (`claude-haiku-4-5-20251001`)
- **Device control**: Home Assistant REST API (lights, media, TV, todo, notify)
- **Memory**: Unified Brain Store (identity narrative, episodes, semantic facts, feedback, reminders)
- **Voice pipeline**: Entirely HA-side — see below

## Architecture

```
[Pi5 @ 10.0.30.5]
├─ Docker: igor                 :4467  ← HA POSTs /conversation/process
├─ systemd: wyoming-openwakeword :10400 ← local wake detection (igor.onnx → igor_v0.1.tflite)
└─ systemd: wyoming-satellite    :10700 ← mic/speaker; HA connects here
          │                             connects to localhost:10400 for wake
       USB mic + USB speaker (plughw:CARD=CODEC,DEV=0)

[Home Assistant @ 10.0.40.5]
├─ Add-on: Wyoming-faster-whisper  (STT on port 10300)
├─ Add-on: Wyoming-piper           (TTS on port 10200)
├─ Integration: Wyoming Protocol → Whisper, Piper, Igor Satellite
├─ Integration: Igor (HACS custom_component) → POST /conversation/process
└─ Voice pipeline: wake → STT → Igor → TTS → satellite speaker

Request flow: user speaks → satellite wake → satellite streams audio to HA →
  Whisper transcribes → HA POSTs text to Igor /conversation/process →
  Igor runs quality_gate → intent_router → Claude LLM (with HA tool calls) →
  response text returned → Piper synthesizes → audio plays through satellite.
```

## Directory Structure

```
igor/
├── server/                      # Igor backend (text-only)
│   ├── main_text.py             # Entry point (python -m server.main_text)
│   ├── api.py                   # FastAPI: /conversation/process, /api/health
│   ├── conversation.py          # ConversationService — pipeline wrapper
│   ├── ha_client.py             # HA REST client (services, templates, area lookup)
│   ├── llm.py                   # Claude API (tool_choice=auto, cached system prompt)
│   ├── brain.py                 # Unified Brain Store (memory, episodes, feedback, reminders)
│   ├── intent_router.py         # Tier 1 direct-match router
│   ├── quality_gate.py          # Hallucination / garbage filter
│   ├── routines.py              # Tool-call frequency logging
│   ├── event_loop.py            # Timer scheduler (no audio — callbacks only)
│   ├── context.py               # InteractionContext (per-request)
│   ├── rooms.py                 # RoomConfig (optional rooms.yaml; HA areas by default)
│   ├── config.py                # Env + constants
│   └── commands/                # Auto-discovered LLM tools
│       ├── base.py
│       ├── light_cmd.py         # light.* services (HA)
│       ├── media_cmd.py         # media_player.* services (HA)
│       ├── tv_cmd.py            # remote.send_command, media_player for TV (HA)
│       ├── todo_cmd.py          # todo.* services (HA)
│       ├── notify_cmd.py        # notify.* services (HA)
│       ├── timer_cmd.py         # set_timer, cancel_timer, list_timers
│       ├── delayed_cmd.py       # delayed_command (schedules any command after N sec)
│       ├── memory_cmd.py        # save_memory, forget_memory
│       ├── feedback_cmd.py      # log_feedback, list_feedback, resolve_feedback
│       ├── weather_cmd.py       # Open-Meteo (no key)
│       ├── network_cmd.py       # LAN scan / device audit
│       ├── time_cmd.py
│       ├── math_cmd.py
│       └── _utils.py            # parse_amount, parse_direction, etc.
├── shared/                      # Pydantic models, shared utils
├── custom_components/igor/      # HA integration (goes in HA's /config/custom_components/)
│   ├── __init__.py
│   ├── conversation.py          # IgorConversationEntity
│   ├── config_flow.py
│   ├── const.py
│   ├── manifest.json
│   └── strings.json
├── deploy/                      # systemd unit files for Pi5 satellite services
│   ├── wyoming-openwakeword.service
│   └── wyoming-satellite.service
├── wakeword/                    # Self-contained wake-word subsystem
│   ├── train.py                 # Outputs both .onnx and .tflite (via onnx2tf)
│   ├── record_samples.py        # Collect positive samples
│   ├── record_negatives.py      # Collect negative samples
│   ├── models/                  # Trained models (.onnx, .tflite — gitignored)
│   └── samples/                 # positive/, negative/ (gitignored)
├── data/                        # Persistent state (brain.json, etc.)
├── docker-compose.yml           # Portainer stack (Pi5)
├── Dockerfile                   # python:3.12-slim, tini, ARM64
├── requirements-server-text.txt # Slim deps (no Whisper/Kokoro/numpy)
├── prompt.py                    # System prompt
├── mcp_server.py                # MCP server for Claude Code (non-audio tools only)
└── .mcp.json                    # MCP server config
```

## Deploy / Run

**Production** — Pi5 Docker stack via Portainer:

```bash
# On the Pi5
cd ~/igor && docker compose up -d --build
# Or via Portainer → Stacks → Igor → Redeploy
# Or (if GitOps is enabled) just git push to main, Portainer auto-deploys
```

Environment variables (set in Portainer stack editor, not committed):
- `ANTHROPIC_API_KEY` — Claude API key
- `HA_TOKEN` — HA long-lived access token
- `HA_URL` — default `http://10.0.40.5:8123`
- `IGOR_API_TOKEN` — shared secret with the HA `custom_components/igor` integration

**Local dev**:

```bash
pip install -r requirements-server-text.txt
ANTHROPIC_API_KEY=... HA_TOKEN=... python -m server.main_text
```

Brain.json persists across container rebuilds via a named Docker volume (`igor_data:/app/data`). Back it up before major changes.

## HA Integration

The HA custom component at `custom_components/igor/` must be copied into HA's `/config/custom_components/igor/` (via Samba/SSH/Studio Code Server). After HA restart, add the integration via Settings → Devices & Services → Add Integration → **Igor**, pointing at `http://10.0.30.5:4467` with your `IGOR_API_TOKEN`.

The HA voice pipeline (Settings → Voice assistants → edit assistant) should have:
- Conversation agent: **Igor**
- Speech-to-text: **faster-whisper**
- Text-to-speech: **piper**
- Wake word: added via the assistant's three-dot menu → **"Add streaming wake word"** → openwakeword → `okay_nabu` (or `igor` once the custom-model TFLite is re-trained and placed in `~/wyoming-openwakeword/custom-models/`)

## Wyoming Satellite on Pi5

The Pi5 runs two systemd services alongside the Igor container:

- **`wyoming-openwakeword`** (port 10400) — loads custom `.tflite` wake models from `~/wyoming-openwakeword/custom-models/`. Service file: `deploy/wyoming-openwakeword.service`.
- **`wyoming-satellite`** (port 10700) — mic/speaker I/O via ALSA, wake-detection via localhost:10400, streams to HA. Service file: `deploy/wyoming-satellite.service`.

Audio device is `plughw:CARD=CODEC,DEV=0` (USB mic + speaker combo). `--mic-auto-gain 0` because we tune the hardware mic level via `alsamixer` once; auto-gain on top causes clipping. Mic at 80% on the CODEC capture control gives ~0.3-0.5 max amplitude — good range for the wake model.

**Important**: wyoming-openwakeword only loads `.tflite` (not `.onnx`). File naming is `<name>_v<version>.tflite` (e.g. `igor_v0.1.tflite`). Train with `wakeword/train.py` which outputs both formats automatically (requires `onnx2tf` for TFLite — `pip install onnx2tf tensorflow tf-keras onnxruntime` in a Python 3.11/3.12 venv since TF doesn't ship wheels for 3.13 yet).

PipeWire gotcha: Raspberry Pi OS Bookworm runs PipeWire per-user by default, but it only touches ALSA `controlC*` and `seq` nodes — it does NOT hold the capture device. If you see `arecord` fail with `Device or resource busy`, the culprit is almost always another capture-using process (e.g. the old Igor client service) holding `pcmC*D*c`. Check with `sudo fuser -v /dev/snd/*`.

## Command System

Commands are auto-discovered from `server/commands/*_cmd.py`. Each is a `Command` subclass (`base.py`) and becomes an LLM tool automatically.

```python
from server.commands.base import Command

class MyCommand(Command):
    name = "my_command"
    description = "What the LLM should know about when to call this"

    @property
    def parameters(self) -> dict:
        return {"thing": {"type": "string", "description": "..."}}

    def execute(self, thing: str) -> str:
        return f"did it: {thing}"
```

- `parameters` returns the **properties dict** (not the full schema — `to_tool()` wraps it)
- All parameters required by default; override `required_parameters` to narrow
- Commands that need room context accept `_ctx=None` — auto-injected by `commands.execute()` via `inspect.signature`
- HA-backed commands use `server.ha_client.get_client()` — returns a lazy-constructed singleton

| Command | Trigger example |
|---------|----------------|
| `get_time` / `calculate` | "What time is it?" / "15% tip on 47" |
| `set_light` / `adjust_brightness` / `set_scene` | "Office lamp to 50%" / "Warmer" / "Movie scene" |
| `play_pause` / `set_volume` / `adjust_volume` / `mute` | "Pause" / "Volume 50" / "Louder" |
| `tv_power` / `tv_key` / `tv_launch` | "TV off" / "Home button" / "Open YouTube" |
| `add_todo` / `list_todos` / `complete_todo` | "Add milk to groceries" |
| `notify` | "Text Ellie I'll be 5 min late" |
| `set_timer` / `cancel_timer` / `list_timers` | "5 minute pasta timer" |
| `save_memory` / `forget_memory` | "Remember I prefer dark roast" |
| `log_feedback` / `list_feedback` / `resolve_feedback` | "Log that" / "Show my change requests" |
| `get_weather` | "Weather?" |
| `scan_network` / `list_known_devices` | "What's on my network?" |
| `delayed_command` | "Turn off lights in 15 minutes" |

## LLM / Conversation

- **Tiered pipeline**: Quality Gate → Intent Router (Tier 1) → LLM (Tier 2+)
- **Quality gate** (`server/quality_gate.py`): rejects hallucinations, single non-command words, repetitive text
- **Intent router** (`server/intent_router.py`): maps unambiguous phrases ("pause", "lights off") directly to commands — zero LLM latency
- **LLM**: `tool_choice=auto`, no `respond()` tool, max 3 rounds. Action commands short-circuit with a confirmation; narrated commands (weather, timers) get a second round for Claude to summarize
- `ChatResult(text, commands_executed)` return type from `llm.chat()`
- `NARRATED_COMMANDS` frozenset in `server/llm.py` controls which commands get a narration round
- History capped at `MAX_CONVERSATION_HISTORY` (10) messages
- Persistent memory in `data/brain.json`; behavior rules injected into cached base prompt, relevant memories into dynamic block
- Session summarizer runs after each non-follow-up turn to extract facts + episode via `LLM.analyze_conversation()` (single API call)

## Living Memory System

Three-tier architecture (see `MEMORY_ROADMAP.md` for the full research and forward plan):

**Identity Narrative** — a living paragraph about the user, stored as a single `identity` entry in brain.json, regenerated by the consolidation engine when memories change. Injected into the base (cached) system prompt as `<my_person>` — zero extra token cost per interaction.

**Episodic Memory** — timestamped interaction summaries (emotional tone, topics, commands). Capped at 200; recent 5 injected into the dynamic prompt as `<recent_episodes>`.

**Semantic Memory** — category/key/value facts with tags. Tag-based retrieval for specific queries.

**Consolidation** — background daemon thread; triggers every 5 unconsolidated episodes. One Haiku call synthesizes memory + recent episodes + knowledge gaps into a new identity narrative. Runs in `ConversationService._run_consolidation`.

**Knowledge Gap Schema** — `_PROFILE_SCHEMA` in `brain.py` defines 14 profile slots (name, birthday, job, partner, household, pets, etc.). `BrainStore.get_knowledge_gaps()` returns questions for unfilled slots, embedded into the narrative so Igor knows what he doesn't know.

## Key Configuration

`server/config.py` — just Claude model, cost rates, history cap, weather default. Everything else is env vars.

Environment variables:
- `ANTHROPIC_API_KEY` (required)
- `HA_TOKEN` (required) — HA long-lived access token
- `HA_URL` (default `http://10.0.40.5:8123`)
- `IGOR_API_TOKEN` — must match the token configured in the HA integration
- `SERVER_HOST` / `SERVER_PORT` (defaults `0.0.0.0:8000`)
- `DEFAULT_LOCATION` — weather (default "Arlington, VA")

No LIFX / Sonos / Google TV / ADB keys — all device control goes through HA now.

## Security Rules

- **Never `shell=True`** in subprocess calls — all commands use list args
- **Never log full transcriptions** — truncate to 50 chars max
- `/conversation/process` requires `X-Igor-Token` header when `IGOR_API_TOKEN` is set
- FastAPI rate-limits `/conversation/process` to 10 req/min per IP
- All API inputs validated by Pydantic with size limits

## Quick Reference

```bash
# Deploy (Pi5)
docker compose up -d --build
# or via Portainer stack → Redeploy

# Health
curl http://10.0.30.5:4467/api/health

# Tail Igor logs
docker logs -f igor

# Tail wake + satellite
journalctl -u wyoming-openwakeword -u wyoming-satellite -f

# Restart wake/satellite after model change
sudo systemctl restart wyoming-openwakeword wyoming-satellite

# MCP tools (Claude Code)
# → list_commands, run_command, test_intent_router, test_quality_gate,
#   test_pipeline, run_benchmark, tail_logs
```
