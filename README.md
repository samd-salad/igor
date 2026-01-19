# Dr. Butts Voice Assistant

A local, privacy-focused voice assistant with personality. Say "Doctor Butts" and ask questions, set timers, control your home, or just chat.

## Features

- **Wake word detection** - Always listening for "Doctor Butts"
- **Natural conversation** - LLM-powered responses with memory
- **Command execution** - Timers, weather, math, volume control, and more
- **Split architecture** - Pi handles audio, PC handles compute (40-67% faster)
- **Privacy-first** - All processing happens locally on your network

## Architecture

- **Raspberry Pi (Client)**: Wake word detection, audio recording/playback, hardware control
- **PC (Server)**: Speech-to-text (Whisper), LLM (Ollama), text-to-speech (Piper)
- **Communication**: HTTP REST API over local network

## Quick Start

### Prerequisites

**On PC (Server):**
- Python 3.13+
- Ollama with qwen3:30b model
- Piper TTS
- Network connectivity to Pi

**On Raspberry Pi (Client):**
- Python 3.9+
- USB microphone and speaker
- Network connectivity to PC

### Installation

#### 1. Server Setup (PC)

```bash
cd /path/to/smart_assistant

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements-server.txt

# Download Piper voice model
# Windows:
Invoke-WebRequest https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx -OutFile voices/en_US-arctic-medium.onnx
Invoke-WebRequest https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json -OutFile voices/en_US-arctic-medium.onnx.json

# Linux/Mac:
wget -P voices/ https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx
wget -P voices/ https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

#### 2. Client Setup (Raspberry Pi)

```bash
cd /path/to/smart_assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-client.txt

# Test audio devices
arecord -l
aplay -l
arecord -d 3 test.wav && aplay test.wav
```

### Configuration

#### Server ([server/config.py](server/config.py))

Update IP addresses to match your network:

```python
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000

PI_HOST = "192.168.0.3"  # Your Pi's IP address
PI_PORT = 8080

PIPER_VOICE = "voices/en_US-arctic-medium.onnx"
```

#### Client ([client/config.py](client/config.py))

```python
SERVER_HOST = "192.168.0.4"  # Your PC's IP address
SERVER_PORT = 8000

AUDIO_DEVICE = "plughw:2,0"  # Run 'arecord -L' to find yours
WAKE_THRESHOLD = 0.1  # Lower = more sensitive (0.0-1.0)
```

### Running

#### Start Server (PC)

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m server.main
```

Expected output:
```
2026-01-19 12:00:00 - server - INFO - Initializing Dr. Butts Voice Assistant Server...
2026-01-19 12:00:05 - server - INFO - Loading Whisper model: base
2026-01-19 12:00:10 - server - INFO - All services initialized successfully
2026-01-19 12:00:10 - server - INFO - Starting server on 0.0.0.0:8000
```

#### Start Client (Pi)

```bash
source .venv/bin/activate
python -m client.main
```

Expected output:
```
2026-01-19 12:00:00 - client - INFO - Initializing Dr. Butts Voice Assistant Client...
2026-01-19 12:00:02 - client - INFO - Loading audio system...
2026-01-19 12:00:03 - client - INFO - Loading wake word detection...
2026-01-19 12:00:04 - client - INFO - All components initialized successfully
2026-01-19 12:00:05 - client - INFO - Entering main loop. Listening for wake word...
```

### Usage

1. Say **"Doctor Butts"** (wake word)
2. Wait for ascending beep
3. Speak your command
4. Wait for descending beep
5. Listen to response

**Example commands:**
- "What time is it?"
- "Set a timer for 5 minutes"
- "What's the weather like?"
- "Calculate 15% tip on $47.32"
- "Remember that I prefer dark roast coffee"
- "Set volume to 75"

## Health Checks

Test server:
```bash
curl http://192.168.0.4:8000/api/health
```

Test client:
```bash
curl http://192.168.0.3:8080/api/health
```

## Systemd Services (Auto-start on Boot)

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

Enable:
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

Enable:
```bash
sudo systemctl enable drbutts-client
sudo systemctl start drbutts-client
sudo systemctl status drbutts-client
```

## Troubleshooting

### Wake word not detecting
- Check models downloaded to `onnx_models/doctor_butts.onnx`
- Adjust `WAKE_THRESHOLD` in [client/config.py](client/config.py) (lower = more sensitive)
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`
- Check logs: wake word score appears in client logs

### Server connection issues
- Verify server is running: `curl http://192.168.0.4:8000/api/health`
- Check firewall rules (allow port 8000 on PC, port 8080 on Pi)
- Verify IP addresses in config files match your network
- Test connectivity: `ping 192.168.0.4` from Pi

### Audio issues on Pi
- List devices: `aplay -l` and `arecord -l`
- Update `AUDIO_DEVICE` in [client/config.py](client/config.py)
- Check ALSA volume: `amixer get PCM`
- Test playback: `speaker-test -c2 -t wav`

### Ollama not responding
- Check if running: `systemctl status ollama` or `ollama serve`
- Verify model: `ollama list` (should show qwen3:30b)
- Test directly: `curl http://localhost:11434/api/tags`

### View logs
```bash
# If running directly
python -m server.main  # Shows logs in terminal
python -m client.main

# If using systemd
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```

## API Reference

### Server Endpoints (PC)

- `POST /api/process_interaction` - Main endpoint for voice processing
- `GET /api/health` - Server health status
- `GET /api/conversation/history` - View conversation history
- `POST /api/conversation/clear` - Clear conversation history

### Client Endpoints (Pi)

- `POST /api/play_audio` - Play audio from timer alerts
- `POST /api/hardware_control` - Execute hardware commands (volume, etc.)
- `POST /api/play_beep` - Play notification beeps
- `GET /api/health` - Client health status

## Utilities

```bash
# Clear conversation history
curl -X POST http://192.168.0.4:8000/api/conversation/clear

# View conversation history
curl http://192.168.0.4:8000/api/conversation/history
```

## Security

- All processing happens locally on your network
- No data sent to cloud services
- Input validation on all API endpoints
- No shell injection vulnerabilities
- Command whitelisting for hardware control
- Designed for trusted home network (no TLS/authentication)