# Dr. Butts Voice Assistant

Local, privacy-focused voice assistant. Say "Doctor Butts" and ask questions, set timers, control volume, or just chat.

- **Pi (Client)**: Wake word detection, audio recording/playback, hardware control
- **PC (Server)**: Whisper STT, Claude LLM, Piper TTS
- **Wake word**: Picovoice Porcupine (custom keywords, no training required)

---

## Setup

### Server (PC)

```bash
python3 -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements-server.txt

# Download Piper voice model
mkdir -p voices
curl -L -o voices/en_US-arctic-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx
curl -L -o voices/en_US-arctic-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json

export ANTHROPIC_API_KEY=sk-ant-...
python -m server.main
```

### Client (Pi)

**1. Get Porcupine keyword files:**
- Create a free account at [console.picovoice.ai](https://console.picovoice.ai)
- Create keywords: "doctor butts" and "stop"
- Download `.ppn` files for **Raspberry Pi** (Cortex-A72 for Pi 4, Cortex-A53 for Pi 3)
- Place in `porcupine_models/`

**2. Install and run:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-client.txt

export PORCUPINE_ACCESS_KEY=your-key-here
python -m client.main
```

### Configuration

Edit `server/config.py`:
```python
PI_HOST = "192.168.0.3"   # Your Pi's IP
PI_PORT = 8080
```

Edit `client/config.py`:
```python
SERVER_HOST = "192.168.0.4"          # Your PC's IP
AUDIO_DEVICE = "plughw:2,0"          # Run 'arecord -L' to find yours
PORCUPINE_KEYWORD_PATHS = [...]       # Paths to your .ppn files
PORCUPINE_SENSITIVITIES = [0.5, 0.5] # Per-keyword sensitivity
```

---

## Usage

1. Say **"Doctor Butts"** → wait for ascending beep
2. Speak your command → wait for descending beep
3. Listen to response

**Examples:** "What time is it?" · "Set a timer for 5 minutes" · "What's the weather?" · "Set volume to 75" · "Remember I prefer dark roast"

---

## Health Checks

```bash
curl http://192.168.0.4:8000/api/health   # Server
curl http://192.168.0.3:8080/api/health   # Pi client
curl -X POST http://192.168.0.4:8000/api/conversation/clear
```

---

## Auto-start (systemd)

Create unit files at `/etc/systemd/system/drbutts-server.service` and `drbutts-client.service`. Key fields:

```ini
[Service]
Environment="ANTHROPIC_API_KEY=..."   # server only
Environment="PORCUPINE_ACCESS_KEY=..." # client only
ExecStart=/path/to/.venv/bin/python -m server.main
Restart=always
```

```bash
sudo systemctl enable drbutts-server && sudo systemctl start drbutts-server
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Wake word not triggering | Adjust `PORCUPINE_SENSITIVITIES` (higher = more sensitive) |
| `.ppn` file not found | Check `PORCUPINE_KEYWORD_PATHS` in `client/config.py` |
| Can't connect to server | Verify IPs in config, check firewall (ports 8000 PC, 8080 Pi) |
| Audio device not found | Run `arecord -L`, update `AUDIO_DEVICE` |
| No API key error | Set `ANTHROPIC_API_KEY` / `PORCUPINE_ACCESS_KEY` env vars |

```bash
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```
