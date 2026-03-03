# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Dev Session Workflow

At the start of every session, check `data/feedback.json` for open change requests and proactively offer to fix them. Example: "You have 2 open items: response verbosity after light commands, and Sonos reconnect on stale cache. Want me to work through them?"

Read the file with:
```python
import json; print(json.dumps(json.loads(open('data/feedback.json').read()), indent=2))
```

After fixing an item, update its status to `"resolved"` directly in the JSON, or remind the user to say "resolve feedback #N" to Dr. Butts.

## Project Overview

Dr. Butts is a local voice assistant. A **Raspberry Pi** handles audio I/O; a **PC** handles compute (STT, LLM, TTS).

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
Flask callback server           ←      Claude LLM + tools
  /api/play_audio                       Kokoro TTS
  /api/hardware_control          →    /api/play_audio (timer alerts)
  /api/play_beep                 →    /api/hardware_control (volume RPC)
  /api/suppress_wakeword         →    (after TV commands)
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
│   ├── speaker_id.py  # Resemblyzer speaker identification (optional)
│   ├── enroll_speaker.py  # CLI tool for enrolling speaker voice profiles
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
├── data/              # Persistent data: memory.json, benchmark.csv
├── setup_client.sh    # Pi setup script (deps + OWW base model download)
├── setup_server.sh    # PC setup script (deps + voice download)
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
- No-parameter commands must use `execute(self, **_)` to avoid TypeError from command dispatch
- Hardware commands (volume) use `self.pi_client` injected at startup via `commands.inject_pi_client()`

| Command | Trigger example |
|---------|----------------|
| `get_time` | "What time is it?" |
| `calculate` | "15% tip on $47" |
| `set_volume` / `adjust_volume` / `get_volume` | "Your volume to 75" / "Turn yourself up" (RPC → Pi) |
| `set_sonos_volume` / `adjust_sonos_volume` / `sonos_mute` | "TV volume to 50" / "Turn the music up a bit" (Sonos) |
| `set_light` / `set_brightness` / `set_color` / `set_color_temp` | "Turn off the lights" / "Make the lights blue" (LIFX) |
| `adjust_brightness` / `adjust_color_temp` / `shift_hue` | "Slightly brighter" / "A lot warmer" (LIFX relative) |
| `tv_power` | "Turn the TV on/off" (androidtvremote2) |
| `tv_key` | "Go home" / "Mute the TV" (androidtvremote2 nav keys) |
| `tv_launch` / `tv_playback` / `tv_skip` / `tv_search_youtube` | "Open YouTube" / "Pause" / "Skip 30 seconds" (ADB) |
| `save_memory` / `forget_memory` | "Remember I prefer dark roast" |
| `log_feedback` / `list_feedback` / `resolve_feedback` | "Log that" / "I didn't like that response" / "Show my change requests" |
| `set_timer` / `cancel_timer` / `list_timers` | "5 minute timer" |
| `get_weather` | "What's the weather?" |
| Network commands | "Scan for new devices" |

## Key Configuration

**`server/config.py`** — update `PI_HOST` to your Pi's IP
**`client/config.py`** — update `SERVER_HOST`, `AUDIO_DEVICE`, `OWW_THRESHOLD`
**`shared/protocol.py`** — endpoint path constants only (no IPs)

Environment variables required:
- `ANTHROPIC_API_KEY` — server only

No other API keys needed. Weather uses Open-Meteo (free, no account). Smart home uses local LAN only.

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
- `await_followup` bool on the respond tool: client listens again without wake word
- Persistent memory injected into system prompt from `data/memory.json`
- Session summarizer runs after each non-follow-up turn to auto-save facts to memory (skipped when TV is playing)
- History overflow compresses dropped messages into `_history_summary` injected as `<prior_context>`

## Interaction Flows

### Normal flow (wake word → response)
1. Pi detects wake word (`OWW_THRESHOLD` × `OWW_TRIGGER_FRAMES` consecutive frames)
2. RMS energy filter rejects low-energy detections (TV/room audio)
3. `_beep("start")` → Sonos `/api/sonos_beep` → light flash if TV playing, else Sonos beep
4. VAD records until silence, sends WAV to server `/api/process_interaction`
5. `_beep("end")` on recording complete
6. Server: STT → LLM (with tools) → Kokoro TTS → response back
7. Pi plays local audio **or** server routes to Sonos (if `prefer_sonos_output=True`)
8. If `await_followup=True`: client sleeps (tts_dur + 2s), checks suppression, listens again

### Beep routing chain (USE_SONOS_OUTPUT=True)
```
_beep(type) → _sonos_beep(type) → POST /api/sonos_beep
  → play_sonos_beep(type, indicator_light)
    → if TV playing AND indicator_light set: LIFX flash (silent)
    → else: Sonos play_uri beep WAV
```
**Rule**: When TV is playing, ALL audio indicators go through LIFX light flash. No sound on Sonos.

### TV-playing path
- `_last_tv_state` is polled every 5s via ADB (`dumpsys media_session`); up to 5s stale
- When playing: non-critical TTS suppressed (tts_routed=True, no audio); `await_followup` forced False; session summarizer skipped; LLM context includes TV note
- After any TV command: `suppress_wakeword(20s)` sent to Pi → `client/suppress.py` blocks detection

### Wake word suppression
- Suppression state lives in `client/suppress.py` (module-level, thread-safe)
- Pi Flask server at `/api/suppress_wakeword` sets it; checked at top of `_handle_interaction()` and inside detection loop
- Follow-up paths also check suppression before opening the mic

### Error beep rule
`self._beep("error")` everywhere (never `self.audio.beep_error()` directly) so routing respects `USE_SONOS_OUTPUT` and TV state.

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

## Todo

- [ ] Web dashboard for monitoring
- [ ] multiple client support
- [ ] add bedroom sonos as output for 2nd client
- [ ] "stop" wake word interrupt — detected in client, needs playback interruption logic
- [ ] Spotify control (spotipy, needs free developer app registration)
- [ ] Calendar, shopping list
- [ ] Puramax2 litterbox control
- [ ] commas and quotes not reading right in TTS
- [ ] Multi-user voice interpretation
- [ ] better network scanning/testing
- [ ] reminder capability
- [ ] todo list capability - integrated with phone and reminder capability?
- [ ] test tv speaking meaning no word response

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
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```
