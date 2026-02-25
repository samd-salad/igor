# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Dr. Butts is a local voice assistant. A **Raspberry Pi** handles audio I/O; a **PC** handles compute (STT, LLM, TTS).

- **STT**: Faster Whisper (`small` model, CPU)
- **LLM**: Claude API (`claude-haiku-4-5-20251001`)
- **TTS**: Piper (`en_US-arctic-medium`)
- **Wake Word**: Sherpa-ONNX keyword spotting (phoneme matching, no training, no account)
- **Server**: FastAPI · **Client callbacks**: Flask

## Architecture

```
Pi (client/)                         PC (server/)
─────────────────────────────────    ─────────────────────────────
Porcupine wake word detection   →    /api/process_interaction
PyAudio VAD recording                  Whisper STT
Flask callback server           ←      Claude LLM + tools
  /api/play_audio                       Piper TTS
  /api/hardware_control          →    /api/play_audio (timer alerts)
  /api/play_beep                 →    /api/hardware_control (volume RPC)
```

## Directory Structure

```
smart_assistant/
├── client/          # Raspberry Pi
│   ├── main.py      # Entry point + main loop
│   ├── wakeword.py  # Sherpa-ONNX keyword spotter
│   ├── audio.py     # PyAudio + beeps (sox)
│   ├── vad_recorder.py
│   ├── hardware.py  # ALSA volume
│   ├── pi_server.py # Flask callback server
│   └── config.py
├── server/          # PC
│   ├── main.py
│   ├── api.py       # FastAPI endpoints + rate limiter
│   ├── orchestrator.py  # STT→LLM→TTS pipeline
│   ├── transcription.py
│   ├── llm.py       # Claude API client
│   ├── synthesis.py # Piper TTS
│   ├── event_loop.py  # Timer thread
│   ├── pi_callback.py # HTTP client → Pi
│   ├── config.py
│   └── commands/    # Auto-discovered LLM tools
│       ├── base.py
│       ├── timer_cmd.py
│       ├── memory_cmd.py
│       ├── math_cmd.py
│       ├── time_cmd.py
│       ├── weather_cmd.py
│       └── network_cmd.py
├── shared/
│   ├── models.py    # Pydantic request/response models
│   ├── protocol.py  # Endpoint path constants
│   └── utils.py
├── sherpa_onnx_models/  # KWS model files (Pi only) — see README for download
├── voices/            # Piper .onnx models (PC only)
├── data/              # Persistent data: memory.json, benchmark.csv
└── prompt.py          # LLM system prompt (Dr. Butts persona)
```

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
- Hardware commands (volume, GPIO) use `self.pi_client` injected at startup via `commands.inject_pi_client()`

| Command | Trigger example |
|---------|----------------|
| `get_time` | "What time is it?" |
| `calculate` | "15% tip on $47" |
| `set_volume` / `get_volume` | "Set volume to 75" (RPC → Pi) |
| `save_memory` / `forget_memory` | "Remember I prefer dark roast" |
| `set_timer` / `cancel_timer` / `list_timers` | "5 minute timer" |
| `get_weather` | "What's the weather?" |
| Network commands | "Scan for new devices" |

## Key Configuration

**`server/config.py`** — update `PI_HOST` to your Pi's IP
**`client/config.py`** — update `SERVER_HOST` to your PC's IP, `AUDIO_DEVICE`, `PORCUPINE_KEYWORD_PATHS`
**`shared/protocol.py`** — endpoint path constants only (no IPs)

Environment variables required:
- `ANTHROPIC_API_KEY` — server only

## Security Rules

- **Never `shell=True`** in subprocess calls — all commands use list args
- **Never log full transcriptions** — truncate to 100 chars max
- Pi server rejects requests from IPs other than `SERVER_HOST` (except `/api/health`)
- FastAPI rate-limits `/api/process_interaction` to 10 req/min per IP
- Hardware commands whitelisted in `shared/models.py` `HardwareControlRequest`
- All API inputs validated by Pydantic with size limits

## LLM / Conversation

- History capped at `MAX_CONVERSATION_HISTORY` (10) messages
- `_trim_history()` ensures history always starts with a plain-text user message — tool_result orphans are dropped to avoid Claude API role errors
- Follow-up mode: LLM appends `[AWAIT]` to signal the client to listen again without a wake word
- Persistent memory injected into system prompt from `data/memory.txt`

## Wake Word (Sherpa-ONNX)

- Model files live in `sherpa_onnx_models/` — download from GitHub releases (see README)
- Keywords are plain text strings in `WAKE_WORDS` in `client/config.py` — no training or account needed
- Uses phoneme matching (CTC transducer); accuracy is good for phonetically distinct phrases
- `WakeWordDetector.predict()` returns `{keyword: 1.0}` on hit, `0.0` otherwise
- Tune sensitivity with `WAKE_THRESHOLD` (default 0.25 — lower = more sensitive)

## Todo

- [ ] Web dashboard for monitoring
- [ ] Multiple Pi support
- [ ] "stop" wakeword interrupt — `.ppn` detected, client needs playback interruption logic
- [ ] integrate prompt into model (avoid reprompting after tool return)
- [ ] Smart lights, TV control, calendar, shopping list
- [ ] commas and quotes not reading right in TTS
- [ ] Multi-user voice interpretation
- [x] swap to Claude API
- [x] wakeword cold-start latency (was Ollama artifact)

## Quick Reference

```bash
# Start
python -m server.main   # PC
python -m client.main   # Pi

# Health
curl http://192.168.0.4:8000/api/health
curl http://192.168.0.3:8080/api/health

# Clear conversation
curl -X POST http://192.168.0.4:8000/api/conversation/clear

# Logs (systemd)
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```
