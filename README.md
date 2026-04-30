# Igor Voice Assistant

Local, privacy-focused voice assistant. Say "Igor" and ask questions, set timers, control lights, or just chat.

- **Pi (Client)**: OpenWakeWord wake word, audio recording/playback, hardware control
- **PC (Server)**: Whisper STT, Claude API LLM, Kokoro TTS
- **Multi-client**: Multiple Pis in different rooms, text clients via REST API

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
# Then follow the wake word training steps below before starting
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
OWW_THRESHOLD = 0.75          # Detection threshold (0–1)
```

### Multi-Room Setup

For multiple clients, create `data/rooms.yaml` (see `data/rooms.yaml.example`):

```yaml
rooms:
  living_room:
    display_name: "Living Room"
    sonos_zone: "Living Room"
    tv_host: "192.168.0.20"
    light_group: ["corner lamp", "table lamp"]
  bedroom:
    display_name: "Bedroom"
    sonos_zone: "Bedroom"
    light_group: ["bedroom lamp"]
```

Each Pi client sets its identity via environment variables:
```bash
CLIENT_ID=bedroom_pi ROOM_ID=bedroom python -m client.main
```

Clients auto-register with the server at startup. Without `rooms.yaml`, a single default room is created from `server/config.py` constants.

---

## Wake Word Training

The wake word detector uses a custom-trained OpenWakeWord model. You need to record samples and train a model before the client will start.

### Step 1 — Record positive samples (on the Pi)

```bash
python wakeword/record_samples.py
```

Follow the prompts. Aim for **150+ samples** across different styles (volume, speed, distance, background noise). Samples are saved to `wakeword/samples/positive/`.

### Step 2 — Transfer samples to PC

```powershell
scp -r samda@10.0.30.5:~/igor/wakeword/samples/ wakeword/
```

### Step 3 — Train the model (on PC)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openwakeword onnx2tf tensorflow tf-keras onnxruntime

python wakeword/train.py
```

Outputs: `wakeword/models/igor.onnx` and `wakeword/models/igor_v0.1.tflite`.

The script uses synthetic negatives (noise/silence) by default. For fewer false positives, record real background audio into `wakeword/samples/negative/*.wav` — the script picks them up automatically.

### Step 4 — Deploy to Pi

```bash
scp wakeword/models/igor_v0.1.tflite samda@10.0.30.5:~/wyoming-openwakeword/custom-models/
ssh samda@10.0.30.5 sudo systemctl restart wyoming-openwakeword wyoming-satellite
```

wyoming-openwakeword loads `.tflite` only; the `.onnx` is kept for reference/future tooling.

---

## Speaker Enrollment (Optional)

Allows the assistant to greet you by name.

```bash
# On the PC (requires resemblyzer):
pip install resemblyzer
python server/enroll_speaker.py enroll "Sam"
python server/enroll_speaker.py list
python server/enroll_speaker.py test
python server/enroll_speaker.py remove "Sam"
```

Follow the prompts — you'll record 5 voice samples in different styles. Resemblyzer is an optional dependency; the assistant works fine without it.

---

## Usage

1. Say **"Igor"** → wait for ascending beep
2. Speak your command → wait for descending beep
3. Listen to response

**Examples:** "What time is it?" · "Set a timer for 5 minutes" · "What's the weather?" · "Turn off the lights" · "Make the lights blue" · "Set volume to 75" · "Remember I prefer dark roast"

### Text Client (REST API)

```bash
curl -X POST http://192.168.0.4:8000/api/text_interaction \
  -H "Content-Type: application/json" \
  -d '{"client_id":"phone","text":"what time is it"}'
```

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
Environment="CLIENT_ID=living_room_pi"  # client only
Environment="ROOM_ID=living_room"       # client only
ExecStart=/path/to/.venv/bin/python -m server.main
Restart=always
```

```bash
sudo systemctl enable igor-server && sudo systemctl start igor-server
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No .onnx models found" | Run wake word training steps above |
| Wake word not triggering | Lower `OWW_THRESHOLD` in `client/config.py` |
| Too many false positives | Raise `OWW_THRESHOLD`, or record more samples and retrain |
| Can't connect to server | Verify IPs in config, check firewall (ports 8000 PC, 8080 Pi) |
| Audio device not found | Run `arecord -L`, update `AUDIO_DEVICE` |
| No API key error | Set `ANTHROPIC_API_KEY` env var |
| Kokoro model not found | Download from kokoro-onnx releases, place in `kokoro/` |

```bash
sudo journalctl -u igor-server -f
sudo journalctl -u igor-client -f
```
