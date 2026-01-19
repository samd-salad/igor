# smart_assistant
boris

## Architecture Overview

The voice assistant is split between two machines:

- **Raspberry Pi (Client)**: Wake word detection, audio I/O, hardware control
- **PC (Server)**: Speech-to-text, LLM processing, text-to-speech, command execution

## Prerequisites

### On PC (Server)
- Python 3.13+ (Python 3.14 not yet fully supported by onnxruntime)
- Ollama installed and running with qwen3:30b model
- Piper TTS installed
- Faster Whisper model files
- Network connectivity to Raspberry Pi

### On Raspberry Pi (Client)
- Python 3.9+
- USB microphone
- Speaker/audio output via ALSA
- Wake word ONNX models
- Network connectivity to PC

## Installation

### 1. Server Setup (PC)

```bash
- Create virtual environment with Python 3.13
.venv\Scripts\activate # on windows to activate

# Install server dependencies
pip install -r requirements-server.txt

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Verify Piper is installed
piper --version
```

### 2. Client Setup (Raspberry Pi)

```bash
cd /path/to/smart_assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install client dependencies
pip install -r requirements-client.txt

# Verify audio devices
aplay -l
arecord -l

# Test microphone
arecord -d 3 test.wav && aplay test.wav
```

## Configuration

### Server Configuration ([server/config.py](server/config.py))

Key settings:
```python
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000

PI_HOST = "192.168.0.3"  # Your Pi's IP
PI_PORT = 8080

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:30b"

WHISPER_MODEL = "base"  # or "small", "medium"
PIPER_VOICE = "path/to/voice.onnx"
# Invoke-WebRequest https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json -OutFile voices/en_US-arctic-medium.onnx.json
# and then the same thing without json

```

### Client Configuration ([client/config.py](client/config.py))

Key settings:
```python
SERVER_HOST = "192.168.0.4"  # Your PC's IP
SERVER_PORT = 8000

CLIENT_HOST = "0.0.0.0"
CLIENT_PORT = 8080

AUDIO_DEVICE = "plughw:2,0"  # Your audio device
WAKE_WORDS = ["doctor_butts"]
WAKE_THRESHOLD = 0.1
```

## Running

### Start Server (on PC)

```bash
cd /path/to/smart_assistant
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run server
python -m server.main

# Or with uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Expected output:
```
2026-01-19 12:00:00 - server - INFO - Initializing Dr. Butts Voice Assistant Server...
2026-01-19 12:00:05 - server - INFO - Loading Whisper model: base
2026-01-19 12:00:10 - server - INFO - All services initialized successfully
2026-01-19 12:00:10 - server - INFO - Starting server on 0.0.0.0:8000
```

### Start Client (on Raspberry Pi)

```bash
cd /path/to/smart_assistant
source .venv/bin/activate

# Run client
python -m client.main
```

Expected output:
```
2026-01-19 12:00:00 - client - INFO - Initializing Dr. Butts Voice Assistant Client...
2026-01-19 12:00:02 - client - INFO - Loading audio system...
2026-01-19 12:00:03 - client - INFO - Loading wake word detection...
2026-01-19 12:00:04 - client - INFO - All components initialized successfully
2026-01-19 12:00:04 - client - INFO - Pi server started on 0.0.0.0:8080
2026-01-19 12:00:05 - client - INFO - Entering main loop. Listening for wake word...
```

## Testing

### Test Server Health

```bash
curl http://192.168.0.4:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "whisper": "loaded",
    "ollama": "connected",
    "piper": "ready",
    "pi_client": "ready",
    "pi": "reachable"
  },
  "uptime_seconds": 123.45,
  "additional_info": {
    "conversation_messages": 0
  }
}
```

### Test Client Health

```bash
curl http://192.168.0.3:8080/api/health
```

### Test End-to-End

1. Say the wake word: "Doctor Butts"
2. Wait for ascending beep
3. Speak your command
4. Wait for descending beep
5. Server processes (STT → LLM → TTS)
6. Response plays on Pi

## Systemd Services (Optional)

### Server Service

Create `/etc/systemd/system/drbutts-server.service`:

```ini
[Unit]
Description=Dr. Butts Voice Assistant Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/smart_assistant
Environment="PATH=/path/to/smart_assistant/.venv/bin"
ExecStart=/path/to/smart_assistant/.venv/bin/python -m server.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable drbutts-server
sudo systemctl start drbutts-server
sudo systemctl status drbutts-server
```

### Client Service

Create `/etc/systemd/system/drbutts-client.service`:

```ini
[Unit]
Description=Dr. Butts Voice Assistant Client
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart_assistant
Environment="PATH=/home/pi/smart_assistant/.venv/bin"
ExecStart=/home/pi/smart_assistant/.venv/bin/python -m client.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable drbutts-client
sudo systemctl start drbutts-client
sudo systemctl status drbutts-client
```

## Troubleshooting

### Server Issues

**Ollama not connecting:**
```bash
# Check if Ollama is running
systemctl status ollama
# Or start it
ollama serve
```

**Piper not found:**
```bash
# Install Piper
# See: https://github.com/rhasspy/piper
```

**Whisper model download:**
```bash
# Models download automatically on first run
# Or pre-download:
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

### Client Issues

**No audio devices:**
```bash
# List devices
aplay -L
arecord -L

# Update AUDIO_DEVICE in client/config.py
```

**Wake word not detecting:**
- Check models exist in `models/` directory
- Adjust WAKE_THRESHOLD (lower = more sensitive)
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`

**Can't connect to server:**
- Verify server is running: `curl http://192.168.0.4:8000/api/health`
- Check firewall rules
- Verify IP addresses in configs match your network

### Network Issues

**Pi can't reach server:**
```bash
# Test connectivity
ping 192.168.0.4
curl http://192.168.0.4:8000/api/health
```

**Server can't callback to Pi:**
```bash
# Test Pi server
curl http://192.168.0.3:8080/api/health
```

## Performance

Expected latency (after wake word):
- **Network transfer**: ~0.1-0.3s
- **STT (Whisper base)**: ~3-4s
- **LLM (qwen3:30b local)**: ~2-10s
- **TTS (Piper)**: ~3-7s
- **Total**: ~8-20s (vs 11-51s all on Pi)

## Security Notes

- Running on trusted home network (no TLS/auth)
- Input validation on all API endpoints
- Command whitelisting for hardware control
- See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for full analysis

## Logs

### View Server Logs
```bash
# If running directly
python -m server.main

# If using systemd
sudo journalctl -u drbutts-server -f
```

### View Client Logs
```bash
# If running directly
python -m client.main

# If using systemd
sudo journalctl -u drbutts-client -f
```

## Useful Commands

```bash
# Clear conversation history
curl -X POST http://192.168.0.4:8000/api/conversation/clear

# View conversation history
curl http://192.168.0.4:8000/api/conversation/history

# Test timer alert (from server)
# Set timer via voice: "Doctor Butts, set a timer for 1 minute"
```

## Next Steps

1. Test basic interaction
2. Test timer alerts (PC → Pi callbacks)
3. Test volume control (PC → Pi RPC)
4. Monitor performance with benchmark.csv
5. Customize wake word if desired
6. Add more commands as needed

## Support

- Issues: https://github.com/yourusername/smart_assistant/issues
- Documentation: [CLAUDE.md](CLAUDE.md)
- Security: [SECURITY_REVIEW.md](SECURITY_REVIEW.md)
