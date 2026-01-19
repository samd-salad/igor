# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Raspberry Pi voice assistant called "Dr. Butts" with wake word detection, speech recognition, LLM integration, and text-to-speech. Uses a **client-server architecture** where the Pi handles audio I/O and the PC handles compute-intensive processing.

## Architecture

### Split Design (Pi + PC)

**Raspberry Pi (Client):**
- Wake word detection
- Audio recording (with VAD)
- Audio playback
- Hardware control (volume, etc.)
- HTTP server for callbacks from PC

**PC (Server):**
- Speech-to-text (Faster Whisper)
- LLM processing (Ollama with qwen3:30b)
- Text-to-speech (Piper)
- Command execution
- Timer alerts (with Pi callbacks)
- All persistent data (memory, benchmarks)

### Communication Flow

```
1. Pi: Wake word detected
2. Pi: Record user speech
3. Pi → PC: Send audio (HTTP POST)
4. PC: STT → LLM → Execute commands → TTS
5. PC → Pi: Send response audio (HTTP response)
6. Pi: Play response audio
```

**Timer Alerts:**
```
1. PC: Timer fires in event loop
2. PC: Synthesize alert message
3. PC → Pi: Send audio via callback (HTTP POST)
4. Pi: Play alert audio
```

## Running the Assistant

### Start Server (on PC)

```bash
cd /path/to/smart_assistant
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

python -m server.main
```

### Start Client (on Raspberry Pi)

```bash
cd /path/to/smart_assistant
source .venv/bin/activate

python -m client.main
```

**See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for full deployment instructions.**

## Directory Structure

```
smart_assistant/
├── client/                  # Runs on Raspberry Pi
│   ├── main.py             # Pi entry point
│   ├── audio.py            # Audio I/O
│   ├── wakeword.py         # Wake word detection
│   ├── vad_recorder.py     # Voice activity detection
│   ├── hardware.py         # Hardware control (volume)
│   ├── pi_server.py        # HTTP server for callbacks
│   └── config.py           # Pi configuration
│
├── server/                  # Runs on PC
│   ├── main.py             # Server entry point
│   ├── api.py              # FastAPI endpoints
│   ├── orchestrator.py     # Processing pipeline
│   ├── transcription.py    # STT (Whisper)
│   ├── llm.py              # LLM (Ollama)
│   ├── synthesis.py        # TTS (Piper)
│   ├── event_loop.py       # Timer management
│   ├── pi_callback.py      # Pi callback client
│   ├── config.py           # Server configuration
│   └── commands/           # Command system
│       ├── __init__.py
│       ├── base.py
│       ├── timer_cmd.py
│       ├── memory_cmd.py
│       ├── math_cmd.py
│       ├── time_cmd.py
│       ├── weather_cmd.py
│       ├── system_cmd.py
│       └── network_cmd.py
│
├── shared/                  # Shared code
│   ├── models.py           # Pydantic API models
│   ├── protocol.py         # API endpoints
│   └── utils.py            # Common utilities
│
├── data/                    # Persistent data (on PC)
│   ├── memory.txt
│   ├── known_devices.json
│   └── benchmark.csv
│
├── models/                  # Wake word models (on Pi)
│   └── doctor_butts.onnx
│
├── oww_models/              # OpenWakeWord models (on Pi)
│   ├── melspectrogram.onnx
│   └── embedding_model.onnx
│
├── prompt.py                # LLM system prompt
├── config.py                # Legacy config (deprecated)
└── voice_assistant.py       # Legacy monolithic code (deprecated)
```

## Command System

Commands are auto-discovered in `server/commands/`. To add a new command:

1. Create `server/commands/yourcommand_cmd.py`
2. Subclass `Command` from `server/commands/base.py`
3. Set `name` and `description` class attributes
4. Implement `parameters` property (JSON schema dict)
5. Implement `execute(**kwargs)` method returning a string

The command is automatically registered and exported as an LLM tool.

### Hardware Commands

Commands that need hardware access (volume, etc.) are routed to Pi via RPC:
- Server detects hardware command
- Server calls Pi's `/api/hardware_control` endpoint
- Pi executes command locally
- Pi returns result to server

### Existing Commands

- `get_time` - Current date/time
- `calculate` - Math expressions and unit conversions
- `set_volume` - Set audio volume (RPC to Pi)
- `save_memory` - Save facts to persistent memory
- `set_timer` - Set named timer with duration
- `cancel_timer` - Cancel active timer
- `list_timers` - List all active timers
- `get_weather` - Fetch weather from OpenWeatherMap
- Network monitoring commands (7 total)

## Configuration

### Server Config ([server/config.py](server/config.py))

| Setting | Purpose |
|---------|---------|
| `SERVER_HOST`, `SERVER_PORT` | Server listen address |
| `PI_HOST`, `PI_PORT` | Pi's IP for callbacks |
| `OLLAMA_URL`, `OLLAMA_MODEL` | LLM configuration |
| `WHISPER_MODEL` | STT model size ("base", "small") |
| `PIPER_VOICE` | TTS voice model path |
| `DATA_DIR` | Persistent data location |

### Client Config ([client/config.py](client/config.py))

| Setting | Purpose |
|---------|---------|
| `SERVER_HOST`, `SERVER_PORT` | PC server address |
| `CLIENT_HOST`, `CLIENT_PORT` | Pi listen address |
| `AUDIO_DEVICE` | ALSA device string |
| `WAKE_WORDS` | List of wake words |
| `WAKE_THRESHOLD` | Detection sensitivity (0.0-1.0) |
| `SILENCE_END_DURATION` | Seconds of silence to stop recording |
| `RMS_SILENCE_THRESHOLD` | Silence detection threshold |

## API Endpoints

### Server Endpoints (PC)

- `POST /api/process_interaction` - Process voice interaction
- `GET /api/health` - Health check
- `GET /api/conversation/history` - Get conversation history
- `POST /api/conversation/clear` - Clear conversation history

### Client Endpoints (Pi)

- `POST /api/play_audio` - Play audio (from timer alerts)
- `POST /api/hardware_control` - Execute hardware command
- `POST /api/play_beep` - Play beep sound
- `GET /api/health` - Health check

## Wake Word Models

Custom wake word models go in `models/` as `.onnx` files. The model name must match an entry in `WAKE_WORDS` list in `client/config.py`.

## Conversation State

- LLM maintains last 10 messages in memory
- Persistent memory saved to `data/memory.txt` via `save_memory` command
- Memory loaded into system prompt on each LLM call
- Conversation can be cleared via API

## Performance

Expected latency (measured from [benchmark.csv](data/benchmark.csv)):

**Current (split architecture):**
- STT: ~3-4s (on PC)
- LLM: ~2-10s (Ollama local on PC, was 5-40s over network)
- TTS: ~3-7s (on PC)
- Network transfer: ~0.2-0.5s
- **Total: ~8-20s** (40-67% faster than monolithic)

**Previous (monolithic on Pi):**
- Total: ~13-56s

## Security

- Input validation on all API endpoints (max sizes, whitelists)
- No shell injection (subprocess uses list args)
- Hardware command whitelist
- Path validation for file operations
- Sensitive data truncated in logs
- See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for full analysis

**Trust Model:**
- Trusted home network (no TLS/authentication)
- Physical control of devices assumed
- Not exposed to internet

## Logging

Server and client log to stdout/stderr. View with:

```bash
# Direct execution
python -m server.main
python -m client.main

# Systemd
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```

Performance benchmarks logged to `data/benchmark.csv`.

## Testing

```bash
# Test server health
curl http://192.168.0.4:8000/api/health

# Test Pi health
curl http://192.168.0.3:8080/api/health

# Test end-to-end
# Say: "Doctor Butts, what time is it?"
```

## Troubleshooting

See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for:
- Common issues
- Network connectivity
- Audio device setup
- Service configuration

## Todo

### Completed
- [x] Split architecture (Pi client + PC server)
- [x] Comprehensive logging and benchmarks
- [x] Security hardening
- [x] Faster LLM processing (local vs network)
- [x] Timer alerts with Pi callbacks

### Remaining
- [ ] Incorporate smart light bulbs
- [ ] Turn TV on and off
- [ ] EnviroGrow integration
- [ ] Todo list integration with phone
- [ ] Shopping list integration with phone
- [ ] Calendar integration
- [ ] Litter box integration
- [ ] Web dashboard for monitoring
- [ ] Multiple Pi support (one server, many clients)
- [ ] local and server logging for pi and pc
- [ ] un-expose pi to internet
- [ ] consider different, better STT and TTS
