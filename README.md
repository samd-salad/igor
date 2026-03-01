# Dr. Butts Voice Assistant

Local, privacy-focused voice assistant. Say "Doctor Butts" and ask questions, set timers, control volume, or just chat.

- **Pi (Client)**: OpenWakeWord wake word, audio recording/playback, hardware control
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
OWW_THRESHOLD = 0.5            # Detection threshold (0–1). Lower = more sensitive.
```

---

## Wake Word Training

The wake word detector uses a custom-trained OpenWakeWord model. You need to record samples and train a model before the client will start.

### Step 1 — Record positive samples (on the Pi)

```bash
python record_samples.py
```

Follow the prompts. Aim for **150+ samples** across different styles (volume, speed, distance, background noise). This takes about 15 minutes.

Samples are saved to `wakeword_samples/positive/`.

### Step 2 — Transfer samples to PC

```bash
# On PC:
rsync -av user@<PI_IP>:~/smart_assistant/wakeword_samples/ wakeword_samples/
```

### Step 3 — Train the model (on PC)

```bash
# One-time installs (CPU-only torch ~150 MB):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openwakeword

python onnx_models/wakeword_creation/train_wakeword.py
```

Output: `oww_models/doctor_butts.onnx`

The script uses synthetic negatives (noise/silence) by default. For fewer false positives in noisy rooms, record real background audio (TV, music, other speech) into `wakeword_samples/negative/*.wav` — the script picks them up automatically.

### Step 4 — Deploy to Pi

```bash
scp oww_models/doctor_butts.onnx pi@<PI_IP>:~/smart_assistant/oww_models/
```

The client globs `oww_models/*.onnx` on startup — drop any `.onnx` file there and it becomes a wake word.

### Adding more wake words

Repeat the process with a different phrase. Name the output file accordingly (e.g. `stop.onnx`). Multiple models load simultaneously.

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
| "No .onnx models found" | Run wake word training steps above |
| Wake word not triggering | Lower `OWW_THRESHOLD` in `client/config.py` |
| Too many false positives | Raise `OWW_THRESHOLD`, or record more samples and retrain |
| Can't connect to server | Verify IPs in config, check firewall (ports 8000 PC, 8080 Pi) |
| Audio device not found | Run `arecord -L`, update `AUDIO_DEVICE` |
| No API key error | Set `ANTHROPIC_API_KEY` env var |
| Piper voice not found | Re-run `setup_server.sh` |

```bash
sudo journalctl -u drbutts-server -f
sudo journalctl -u drbutts-client -f
```
