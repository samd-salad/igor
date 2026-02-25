# Dr. Butts Voice Assistant

Local, privacy-focused voice assistant. Say "Doctor Butts" and ask questions, set timers, control volume, or just chat.

- **Pi (Client)**: Sherpa-ONNX wake word, audio recording/playback, hardware control
- **PC (Server)**: Whisper STT, Claude API LLM, Piper TTS

---

## Setup

### Server (PC)

```bash
bash setup_server.sh
export ANTHROPIC_API_KEY=sk-ant-...
python -m server.main
```

### Client (Pi)

```bash
bash setup_client.sh
python -m client.main
```

### Configuration

Edit `server/config.py`:
```python
PI_HOST = "192.168.0.3"   # Your Pi's IP
```

Edit `client/config.py`:
```python
SERVER_HOST = "192.168.0.4"   # Your PC's IP
AUDIO_DEVICE = "plughw:2,0"   # Run 'arecord -L' to find yours
WAKE_WORDS = ["doctor butts", "stop"]
WAKE_THRESHOLD = 0.25         # Lower = more sensitive
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
curl http://192.168.0.4:8000/api/health
curl http://192.168.0.3:8080/api/health
curl -X POST http://192.168.0.4:8000/api/conversation/clear
```

---

## Auto-start (systemd)

```ini
[Service]
Environment="ANTHROPIC_API_KEY=..."   # server only
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
| Wake word not triggering | Lower `WAKE_THRESHOLD` in `client/config.py` |
| Model directory not found | Re-run `setup_client.sh` |
| Can't connect to server | Verify IPs in config, check firewall (ports 8000 PC, 8080 Pi) |
| Audio device not found | Run `arecord -L`, update `AUDIO_DEVICE` |
| No API key error | Set `ANTHROPIC_API_KEY` env var |

```bash
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```
