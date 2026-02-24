# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Dr. Butts is a local voice assistant with personality, using a **client-server architecture** where a Raspberry Pi handles audio I/O and a PC handles compute-intensive processing (STT, LLM, TTS).

**Core Technologies:**
- **STT**: Faster Whisper (base model)
- **LLM**: Ollama (qwen3:30b)
- **TTS**: Piper (en_US-arctic-medium)
- **Wake Word**: Sherpa-ONNX (keyword spotting, no training required)
- **Server**: FastAPI
- **Client**: Flask (for callbacks)

## Architecture

### Split Design

**Raspberry Pi (Client)** - [client/](client/)
- Wake word detection (always listening)
- Audio recording with voice activity detection
- Audio playback
- Hardware control (volume via ALSA)
- HTTP server for callbacks from PC

**PC (Server)** - [server/](server/)
- Speech-to-text (Faster Whisper)
- LLM processing (Ollama local)
- Text-to-speech (Piper)
- Command execution
- Timer alerts (with Pi callbacks)
- All persistent data storage

### Communication Flow

**Normal Interaction:**
```
1. Pi: Continuous wake word detection loop
2. Pi: Detects "Doctor Butts" → beep
3. Pi: Record speech until silence
4. Pi → PC: POST /api/process_interaction {audio_base64, wake_word, timestamp}
5. PC: STT → LLM → Execute commands → TTS
6. PC → Pi: Response {transcription, response_text, audio_base64, timings}
7. Pi: Play response audio → beep
```

**Timer Alerts (Callback):**
```
1. PC: Event loop detects timer expiration
2. PC: Synthesize alert message with TTS
3. PC → Pi: POST /api/play_audio {audio_base64, message, priority}
4. Pi: Play alert audio
```

**Hardware Control (RPC):**
```
1. User: "Set volume to 75"
2. PC: LLM calls set_volume tool
3. PC → Pi: POST /api/hardware_control {command: "set_volume", parameters: {level: 75}}
4. Pi: Execute amixer command
5. Pi → PC: Response {status, result}
```

## Directory Structure

```
smart_assistant/
├── client/                      # Runs on Raspberry Pi
│   ├── main.py                 # Entry point, main loop
│   ├── audio.py                # PyAudio wrapper, beep sounds
│   ├── wakeword.py             # ONNX inference for wake word
│   ├── vad_recorder.py         # Voice activity detection
│   ├── hardware.py             # ALSA volume control
│   ├── pi_server.py            # Flask server for callbacks
│   └── config.py               # Pi configuration
│
├── server/                      # Runs on PC
│   ├── main.py                 # Entry point, service initialization
│   ├── api.py                  # FastAPI app, endpoints
│   ├── orchestrator.py         # Main processing pipeline
│   ├── transcription.py        # Faster Whisper integration
│   ├── llm.py                  # Ollama client
│   ├── synthesis.py            # Piper TTS integration
│   ├── event_loop.py           # Timer management thread
│   ├── pi_callback.py          # HTTP client for Pi callbacks
│   ├── config.py               # Server configuration
│   └── commands/               # Command system
│       ├── __init__.py         # Auto-discovery
│       ├── base.py             # Command base class
│       ├── timer_cmd.py        # Timer commands
│       ├── memory_cmd.py       # Persistent memory
│       ├── math_cmd.py         # Calculator
│       ├── time_cmd.py         # Current time
│       ├── weather_cmd.py      # Weather API
│       ├── system_cmd.py       # Volume control
│       └── network_cmd.py      # Network monitoring
│
├── shared/                      # Shared between client and server
│   ├── models.py               # Pydantic models with validation
│   ├── protocol.py             # API endpoint paths
│   └── utils.py                # Common utilities
│
├── data/                        # Persistent data (server only)
│   ├── memory.txt              # save_memory command storage
│   ├── known_devices.json      # Network monitoring data
│   └── benchmark.csv           # Performance logs
│
├── onnx_models/                 # Wake word models (Pi only)
│   └── doctor_butts.onnx
│
├── oww_models/                  # OpenWakeWord models (Pi only)
│   ├── melspectrogram.onnx     # Auto-downloaded
│   └── embedding_model.onnx    # Auto-downloaded
│
├── voices/                      # Piper TTS models (server only)
│   ├── en_US-arctic-medium.onnx
│   └── en_US-arctic-medium.onnx.json
│
├── prompt.py                    # LLM system prompt
├── requirements-client.txt      # Pi dependencies
├── requirements-server.txt      # PC dependencies
├── README.md                    # User documentation
└── CLAUDE.md                    # This file
```

## Command System

Commands are auto-discovered plugin modules in [server/commands/](server/commands/).

### Creating a New Command

1. Create `server/commands/yourcommand_cmd.py`
2. Subclass `Command` from [server/commands/base.py](server/commands/base.py)
3. Implement required attributes and methods

**Example:**
```python
from server.commands.base import Command

class YourCommand(Command):
    name = "your_command"
    description = "Brief description for LLM to understand when to use this"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "What param1 does"
                }
            },
            "required": ["param1"]
        }

    def execute(self, param1: str) -> str:
        """Execute the command and return a string result."""
        # Your logic here
        return f"Executed with {param1}"
```

The command is automatically registered and becomes available to the LLM as a tool.

### Hardware Commands

Commands that need hardware access (volume, GPIO, etc.) should use the Pi callback system:

```python
def execute(self, level: int) -> str:
    # Access pi_client injected during initialization
    response = self.pi_client.hardware_control("set_volume", {"level": level})
    return f"Volume set to {level}"
```

Set `pi_client` attribute during command instantiation in [server/orchestrator.py](server/orchestrator.py).

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `get_time` | Current date/time | "What time is it?" |
| `calculate` | Math expressions, unit conversions | "Calculate 15% tip on $47" |
| `set_volume` | ALSA volume control (RPC to Pi) | "Set volume to 75" |
| `save_memory` | Save to persistent memory | "Remember I prefer dark roast" |
| `set_timer` | Create named timer | "Set a timer for 5 minutes" |
| `cancel_timer` | Cancel active timer | "Cancel the pasta timer" |
| `list_timers` | List all active timers | "What timers are running?" |
| `get_weather` | Fetch from OpenWeatherMap | "What's the weather?" |
| Network commands | 7 total (scan, list, etc.) | "Scan for new devices" |

## Configuration

### Server Config ([server/config.py](server/config.py))

**Network:**
- `SERVER_HOST`, `SERVER_PORT` - Server listen address (default: 0.0.0.0:8000)
- `PI_HOST`, `PI_PORT` - Pi's IP for callbacks (update for your network!)

**AI Models:**
- `OLLAMA_URL` - Ollama API URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - LLM model (default: qwen3:30b)
- `WHISPER_MODEL` - STT model size (base/small/medium)
- `PIPER_VOICE` - TTS voice model path

**Storage:**
- `DATA_DIR` - Persistent data location (memory, benchmarks)

**Limits:**
- `MAX_CONVERSATION_HISTORY` - Messages to keep (default: 10)

### Client Config ([client/config.py](client/config.py))

**Network:**
- `SERVER_HOST`, `SERVER_PORT` - PC server address (update for your network!)
- `CLIENT_HOST`, `CLIENT_PORT` - Pi listen address (default: 0.0.0.0:8080)

**Audio:**
- `SAMPLE_RATE` - Audio sample rate (16000 Hz)
- `AUDIO_DEVICE` - ALSA device string (find with `arecord -L`)

**Wake Word:**
- `WAKE_WORDS` - List of wake words (default: ["doctor_butts"])
- `WAKE_THRESHOLD` - Detection sensitivity 0.0-1.0 (lower = more sensitive)

**VAD:**
- `SILENCE_END_DURATION` - Seconds of silence to stop recording (default: 2.0)
- `RMS_SILENCE_THRESHOLD` - RMS value for silence detection (default: 1200)
- `MIN_RECORDING` - Minimum recording duration (default: 0.7s)
- `MAX_RECORDING` - Maximum recording duration (default: 15s)

**Timeouts:**
- `REQUEST_TIMEOUT` - HTTP request timeout (default: 60s)

## API Protocol

### Server Endpoints (PC)

**POST /api/process_interaction**
```python
Request: ProcessInteractionRequest
{
  "audio_base64": str,  # WAV audio, max 10MB decoded
  "wake_word": str,     # Name of detected wake word
  "timestamp": float    # Unix timestamp
}

Response: ProcessInteractionResponse
{
  "transcription": str,
  "response_text": str,
  "audio_base64": str,  # TTS audio response
  "commands_executed": List[str],
  "timings": {
    "stt": float,
    "llm": float,
    "tts": float
  },
  "error": Optional[str]
}
```

**GET /api/health**
```python
Response: HealthCheckResponse
{
  "status": "healthy" | "unhealthy",
  "services": {
    "whisper": str,
    "ollama": str,
    "piper": str,
    "pi": str
  },
  "uptime_seconds": float,
  "additional_info": dict
}
```

**GET /api/conversation/history**
```python
Response: {"history": List[Dict]}
```

**POST /api/conversation/clear**
```python
Response: {"status": "success", "message": str}
```

### Client Endpoints (Pi)

**POST /api/play_audio**
```python
Request: PlayAudioRequest
{
  "audio_base64": str,
  "message": str,          # Alert message
  "priority": str          # "alert" | "normal"
}

Response: {"status": "success" | "error"}
```

**POST /api/hardware_control**
```python
Request: HardwareControlRequest
{
  "command": str,          # Whitelisted: "set_volume", "get_volume"
  "parameters": dict
}

Response: {"status": str, "result": Any}
```

**POST /api/play_beep**
```python
Request: PlayBeepRequest
{
  "beep_type": str  # "alert" | "error" | "done" | "start" | "end"
}

Response: {"status": "success"}
```

**GET /api/health**
```python
Response: HealthCheckResponse
```

## Security Guidelines

### Input Validation

All API models use Pydantic with strict validation:
- **Max sizes**: audio_base64 (10MB), transcription (10k chars)
- **Whitelists**: hardware commands, wake words
- **Type safety**: Field validators for base64, enums, etc.

See [shared/models.py](shared/models.py) for validation rules.

### Subprocess Safety

**NEVER use `shell=True`**. Always pass command arguments as a list:

```python
# GOOD
subprocess.run(['piper', '--model', model_path], input=text.encode())

# BAD - VULNERABLE TO INJECTION
subprocess.run(f'echo "{text}" | piper --model {model_path}', shell=True)
```

### CORS Configuration

[server/api.py](server/api.py) restricts origins to Pi's IP:

```python
allowed_origins = [
    f"http://{PI_HOST}:8080",
    f"http://{PI_HOST}",
    "http://192.168.0.3:8080",
    "http://localhost:8080",
]
```

Update if Pi IP changes.

### Logging

Truncate sensitive data:
```python
# GOOD
logger.info(f"Transcribed: '{transcription[:100]}...'")

# BAD - MAY LOG SENSITIVE USER DATA
logger.info(f"Transcribed: '{transcription}'")
```

## Development Guidelines

### Error Handling

- Use try/finally for resource cleanup
- Return error dicts instead of raising for user-facing errors
- Log exceptions with `exc_info=True` for debugging

### File Operations

All persistent data lives in `data/` on server:
- Use `DATA_DIR / "filename.txt"` from [server/config.py](server/config.py)
- Atomic writes for JSON files
- Handle file not found gracefully

### Network Calls

- Always set timeouts
- Handle connection errors
- Log failures, don't crash

### Threading

- Event loop runs in daemon thread
- Clean up on shutdown (see [server/main.py](server/main.py) finally block)
- Use locks for shared state

### Testing

Health check endpoints:
```bash
# Server
curl http://192.168.0.4:8000/api/health

# Client
curl http://192.168.0.3:8080/api/health
```

End-to-end test:
```bash
# Say: "Doctor Butts, what time is it?"
# Check logs for full pipeline execution
```

## Performance

Current benchmarks (from [data/benchmark.csv](data/benchmark.csv)):

| Stage | Time | Notes |
|-------|------|-------|
| STT | ~3-4s | Whisper base on PC |
| LLM | ~2-10s | Ollama local (was 5-40s remote) |
| TTS | ~3-7s | Piper on PC |
| Network | ~0.2-0.5s | Pi ↔ PC transfer |
| **Total** | **~8-20s** | 40-67% faster than monolithic |

**Previous (all on Pi):** 13-56s

## Conversation State

- Last 10 messages kept in memory ([server/llm.py](server/llm.py))
- Persistent memory in `data/memory.txt` injected into system prompt
- Clear via API: `curl -X POST http://192.168.0.4:8000/api/conversation/clear`

## LLM Personality

Defined in [prompt.py](prompt.py):
- **Dr. Butts** - Dry, sardonic, formal vocabulary
- **Rules**: Brevity, no groveling, spoken output only
- **Memory**: Inferential (learns patterns from context)

Prompt is injected as system message on every LLM call. Could be baked into Ollama Modelfile for better persistence.

## Wake Word Detection

Uses Sherpa-ONNX keyword spotting (no per-word training required):
- Pre-trained transducer model (`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`)
- Model auto-downloads to `sherpa_models/` on first run (~6 MB)
- Keywords defined as plain-text phrases in `WAKE_WORDS` / `STOP_WORDS` in [client/config.py](client/config.py)
- Continuous streaming inference on 80ms audio chunks (1280 samples at 16kHz)
- Returns `{keyword: 1.0}` on detection, `{keyword: 0.0}` otherwise
- `WAKE_THRESHOLD` maps to `keywords_threshold` in Sherpa-ONNX (default 0.25)

To add a new keyword: add the phrase string to `WAKE_WORDS` or `STOP_WORDS` in config and restart.

## Troubleshooting

### Common Issues

1. **Wake word always triggers**: Check `WAKE_THRESHOLD`, ensure score actually exceeds threshold in logs
2. **Can't connect to server**: Verify IPs in config match your network, check firewall
3. **Ollama not responding**: `systemctl status ollama` or `ollama serve`
4. **Audio device not found**: `arecord -L`, update `AUDIO_DEVICE` in [client/config.py](client/config.py)

See [README.md](README.md) for full troubleshooting guide.

## Code Modification Patterns

### Adding a Command

1. Create file in [server/commands/](server/commands/)
2. Subclass `Command`, implement `execute()`
3. Restart server (auto-discovered)

### Changing Wake Word

1. Get ONNX model for new wake word
2. Place in `onnx_models/yourword.onnx`
3. Update `WAKE_WORDS` in [client/config.py](client/config.py)
4. Restart client

### Changing LLM Model

1. Pull model: `ollama pull modelname`
2. Update `OLLAMA_MODEL` in [server/config.py](server/config.py)
3. Restart server

### Changing TTS Voice

1. Download Piper voice from https://huggingface.co/rhasspy/piper-voices
2. Place .onnx and .onnx.json in `voices/`
3. Update `PIPER_VOICE` in [server/config.py](server/config.py)
4. Restart server

## Todo List

- [ ] Web dashboard for monitoring
- [ ] Multiple Pi support (one server, many clients)
- [ ] Un-expose Pi to internet
- [ ] consider different STT
- [ ] consider different TTS
- [ ] mcp server-ify?
- [ ] more wake word training
- [ ] potentially different wake word per endpoint
- [x] wakeword initial detection starts llm spoolup — N/A, was Ollama artifact; Claude API has no cold-start
- [ ] integrate prompt into model (no reprompting after command return)
- [x] swap back to claude api
- [ ] "stop" wakeword interrupt handling — keyword detection wired in via Sherpa-ONNX (STOP_WORDS config), client needs logic to interrupt playback on detection
- [ ] fix broken integrations (??)

### Testing
- [ ] commas and quotes not reading right
- [ ] Multi-user voice interpretation
- [ ] question and response without re-waking

### Capabilities and Commands
- [ ] Incorporate smart light bulbs
- [ ] Turn TV on and off
- [ ] EnviroGrow integration
- [ ] Todo list integration with phone
- [ ] Shopping list integration with phone
- [ ] Calendar integration
- [ ] Litter box integration

## Quick Reference

**Start services:**
```bash
# Server (PC)
python -m server.main

# Client (Pi)
python -m client.main
```

**View logs:**
```bash
# Systemd
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```

**Health checks:**
```bash
curl http://192.168.0.4:8000/api/health
curl http://192.168.0.3:8080/api/health
```

**Clear memory:**
```bash
curl -X POST http://192.168.0.4:8000/api/conversation/clear
```
