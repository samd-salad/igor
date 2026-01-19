# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Raspberry Pi voice assistant called "Dr. Butts" with wake word detection, speech recognition, Claude AI integration, and text-to-speech. Runs as a continuous listening loop on local hardware.

## Running the Assistant

```bash
./voice_assistant.py
# or
python3 voice_assistant.py
```

Requires: USB microphone, speaker via ALSA, and an `ANTHROPIC_API_KEY` environment variable.

## Architecture

**Audio Pipeline Flow:**
1. Wake word detection (continuous streaming via PyAudio)
2. Audio recording (sox → temp WAV file)
3. Speech transcription (Faster Whisper)
4. Claude API call with tool support
5. Tool execution if requested, then follow-up Claude call
6. Text-to-speech response (Piper → aplay)

**Key Files:**
- `voice_assistant.py` - Main loop orchestrating the pipeline
- `config.py` - All tunable parameters (models, thresholds, audio device, paths)
- `prompt.py` - Claude system prompt with personality and instructions
- `wakeword.py` - ONNX-based wake word detection with sliding window inference
- `commands/` - Plugin command system for Claude tool use

## Command System

Commands are auto-discovered via `commands/__init__.py`. To add a new command:

1. Create `commands/yourcommand_cmd.py`
2. Subclass `Command` from `commands/base.py`
3. Set `name` and `description` class attributes
4. Implement `parameters` property (JSON schema dict)
5. Implement `execute(**kwargs)` method returning a string

The command is automatically registered and exported as a Claude tool.

**Existing commands:** `get_time`, `set_volume`, `save_memory`, `load_memory`

## Configuration (config.py)

| Setting | Purpose |
|---------|---------|
| `CLAUDE_MODEL` | Which Claude model to use |
| `WHISPER_MODEL` | Speech recognition model size ("base", "small", etc.) |
| `WAKE_THRESHOLD` | Confidence threshold for wake word (0.0-1.0) |
| `AUDIO_DEVICE` | ALSA device string for recording/playback |
| `MIN_RECORDING` / `MAX_RECORDING` | Speech recording duration bounds |

## Wake Word Models

Custom wake word models go in `models/` as `.onnx` files. The model name must match an entry in `WAKE_WORDS` list in config.py.

## Conversation State

- Maintains last 10 messages in `conversation_history`
- Persistent memory saved to `memory.txt` via the `save_memory` command
- Memory loaded into system prompt on each Claude call

## Todo

- [ ] Incorporate smart light bulbs
- [ ] Turn TV on and off
- [ ] EnviroGrow integration
- [ ] Todo list integration with phone
- [ ] Shopping list integration with phone
- [ ] Calendar integration
- [ ] Litter box integration
