#!/usr/bin/env python3
"""Voice assistant with wake word detection, error handling, and command system."""

import logging
import os
import sys
import time
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Suppress stderr during noisy library imports
_stderr_fd = sys.stderr.fileno()
_null_fd = os.open(os.devnull, os.O_RDWR)
_saved_stderr = os.dup(_stderr_fd)
os.dup2(_null_fd, _stderr_fd)

import warnings
warnings.filterwarnings("ignore")
os.environ["ONNXRUNTIME_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pyaudio
import requests
from faster_whisper import WhisperModel

from config import (
    OLLAMA_URL, OLLAMA_MODEL, WHISPER_MODEL, WAKE_WORDS, WAKE_THRESHOLD,
    SAMPLE_RATE, AUDIO_DEVICE, MODELS_DIR, PIPER_VOICE, TEMP_WAV
)
from prompt import SYSTEM_PROMPT
from wakeword import WakeWordDetector
from vad_recorder import VADRecorder
import commands
from commands.memory_cmd import load_persistent_memory
from benchmark import get_benchmark

# Restore stderr
os.dup2(_saved_stderr, _stderr_fd)
os.close(_null_fd)
os.close(_saved_stderr)

from contextlib import contextmanager

@contextmanager
def suppress_alsa_errors():
    """Suppress ALSA/JACK warnings during PyAudio operations."""
    stderr_fd = sys.stderr.fileno()
    null_fd = os.open(os.devnull, os.O_RDWR)
    saved = os.dup(stderr_fd)
    os.dup2(null_fd, stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(null_fd)
        os.close(saved)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio system configuration."""
    sample_rate: int = SAMPLE_RATE
    device: str = AUDIO_DEVICE
    piper_voice: str = PIPER_VOICE
    temp_wav: str = TEMP_WAV


class Audio:
    """Handles all audio input/output operations."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.pa: Optional[pyaudio.PyAudio] = None
        self.vad_recorder: Optional[VADRecorder] = None

    def initialize(self) -> bool:
        """Initialize PyAudio and VAD recorder. Returns success."""
        try:
            with suppress_alsa_errors():
                self.pa = pyaudio.PyAudio()
            dev_index = self._get_device_index()
            self.vad_recorder = VADRecorder(self.pa, dev_index)
            return True
        except Exception as e:
            log.error(f"Audio initialization failed: {e}")
            return False

    def cleanup(self):
        """Clean up audio resources."""
        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = None

    def _get_device_index(self) -> Optional[int]:
        """Find USB microphone device index."""
        if not self.pa:
            return None
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if "USB" in info["name"] and info["maxInputChannels"] > 0:
                return i
        return None

    def open_stream(self):
        """Open audio input stream for wake word detection."""
        dev_index = self._get_device_index()
        return self.pa.open(
            rate=self.config.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1280,
            input_device_index=dev_index
        )

    def record(self) -> bool:
        """Record audio with VAD. Returns True if speech was captured."""
        if not self.vad_recorder:
            return False
        try:
            return self.vad_recorder.record_with_vad(self.config.temp_wav)
        except Exception as e:
            log.error(f"Recording failed: {e}")
            return False

    def speak(self, text: str) -> bool:
        """Speak text using Piper TTS. Returns success."""
        try:
            safe_text = text.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
            subprocess.run(
                f'echo "{safe_text}" | piper --model {self.config.piper_voice} --output-raw 2>/dev/null | '
                f'aplay -D {self.config.device} -r 22050 -f S16_LE -t raw - 2>/dev/null',
                shell=True,
                stdout=subprocess.DEVNULL,
                timeout=30
            )
            return True
        except subprocess.TimeoutExpired:
            log.error("TTS timed out")
            return False
        except Exception as e:
            log.error(f"TTS failed: {e}")
            return False

    @staticmethod
    def beep_start():
        """Rising tone: 'I'm listening'"""
        subprocess.run(
            'play -n synth 0.12 sine 500:900 vol 0.3',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    @staticmethod
    def beep_end():
        """Falling tone: 'Got it'"""
        subprocess.run(
            'play -n synth 0.12 sine 700:400 vol 0.25',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    @staticmethod
    def beep_done():
        """Two quick chirps: 'Ready'"""
        subprocess.run(
            'play -n synth 0.06 sine 1200 vol 0.2 pad 0 0.04 synth 0.06 sine 1200 vol 0.2',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    @staticmethod
    def beep_error():
        """Low buzz: 'Error occurred'"""
        subprocess.run(
            'play -n synth 0.3 sine 200 vol 0.25',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


class Transcriber:
    """Handles speech-to-text transcription."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[WhisperModel] = None

    def initialize(self) -> bool:
        """Load Whisper model. Returns success."""
        try:
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            return True
        except Exception as e:
            log.error(f"Whisper initialization failed: {e}")
            return False

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file. Returns text or None on failure."""
        if not self.model:
            return None
        try:
            segments, _ = self.model.transcribe(audio_path, beam_size=1)
            text = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as e:
            log.error(f"Transcription failed: {e}")
            return None


class WakeWord:
    """Handles wake word detection."""

    def __init__(self, wake_words: list[str], threshold: float, models_dir):
        self.wake_words = wake_words
        self.threshold = threshold
        self.models_dir = models_dir
        self.detector: Optional[WakeWordDetector] = None

    def initialize(self) -> bool:
        """Load wake word models. Returns success."""
        try:
            models = []
            for wake_word in self.wake_words:
                custom_path = self.models_dir / f"{wake_word}.onnx"
                if custom_path.exists():
                    models.append(str(custom_path))
            if not models:
                log.error("No wake word models found")
                return False
            self.detector = WakeWordDetector(models, threshold=self.threshold)
            return True
        except Exception as e:
            log.error(f"Wake word initialization failed: {e}")
            return False

    def listen(self, audio_stream) -> Optional[str]:
        """Listen for wake word. Returns detected word or None on error."""
        if not self.detector:
            return None
        try:
            while True:
                audio = np.frombuffer(
                    audio_stream.read(1280, exception_on_overflow=False),
                    dtype=np.int16
                )
                prediction = self.detector.predict(audio)
                for word, score in prediction.items():
                    if score > self.threshold:
                        self.detector.reset()
                        return word
        except Exception as e:
            log.error(f"Wake word detection error: {e}")
            return None


class LLM:
    """Handles LLM interactions."""

    def __init__(self, url: str, model: str, max_history: int = 10):
        self.url = url
        self.model = model
        self.max_history = max_history
        self.conversation_history: list[dict] = []

    def _get_system_prompt(self) -> str:
        """Build system prompt with current persistent memory."""
        memory = load_persistent_memory()
        return SYSTEM_PROMPT.format(persistent_memory=memory)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool format to Ollama/OpenAI format."""
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        } for t in tools]

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 thinking tags from response."""
        import re
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def chat(self, user_text: str) -> Optional[str]:
        """Send message to LLM and get response. Returns reply or None on failure."""
        self.conversation_history.append({"role": "user", "content": user_text})

        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        messages = [{"role": "system", "content": self._get_system_prompt()}] + self.conversation_history
        tools = self._convert_tools(commands.get_tools())

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            message = response.json().get("message", {})
        except requests.Timeout:
            log.error("LLM request timed out")
            return None
        except requests.RequestException as e:
            log.error(f"LLM request failed: {e}")
            return None
        except Exception as e:
            log.error(f"LLM error: {e}")
            return None

        # Handle tool calls
        if message.get("tool_calls"):
            tool_results = []
            for tool_call in message["tool_calls"]:
                func = tool_call["function"]
                try:
                    result = commands.execute(func["name"], **func["arguments"])
                except Exception as e:
                    log.error(f"Tool '{func['name']}' failed: {e}")
                    result = f"Error executing {func['name']}: {e}"
                tool_results.append({"role": "tool", "content": result})

            self.conversation_history.append(message)
            self.conversation_history.extend(tool_results)
            messages = [{"role": "system", "content": self._get_system_prompt()}] + self.conversation_history

            try:
                response = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                message = response.json().get("message", {})
            except Exception as e:
                log.error(f"LLM follow-up failed: {e}")
                return None

        reply = self._strip_thinking(message.get("content", ""))
        self.conversation_history.append({"role": "assistant", "content": reply})

        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return reply


@dataclass
class VoiceAssistant:
    """Main voice assistant orchestrator."""

    audio: Audio = field(default_factory=lambda: Audio(AudioConfig()))
    transcriber: Transcriber = field(default_factory=lambda: Transcriber(WHISPER_MODEL))
    wake_word: WakeWord = field(default_factory=lambda: WakeWord(WAKE_WORDS, WAKE_THRESHOLD, MODELS_DIR))
    llm: LLM = field(default_factory=lambda: LLM(OLLAMA_URL, OLLAMA_MODEL))

    def initialize(self) -> bool:
        """Initialize all components. Returns True if successful."""
        print(f"Loading speech recognition ({WHISPER_MODEL})...", end=" ", flush=True)
        if not self.transcriber.initialize():
            print("FAILED")
            return False
        print("OK")

        print("Loading wake word detection...", end=" ", flush=True)
        if not self.wake_word.initialize():
            print("FAILED")
            return False
        print("OK")

        print("Loading audio system...", end=" ", flush=True)
        if not self.audio.initialize():
            print("FAILED")
            return False
        print("OK")

        print("Starting background services...", end=" ", flush=True)
        try:
            from event_loop import get_event_loop
            event_loop = get_event_loop()
            event_loop.enable_network_monitoring()
            print("OK\n")
        except Exception as e:
            log.warning(f"Background services partially failed: {e}")
            print("PARTIAL\n")

        print("=" * 40)
        print(f"  Wake words: {', '.join(WAKE_WORDS)}")
        print(f"  Commands:   {', '.join(commands.get_all_commands().keys())}")
        print("=" * 40)
        print()

        return True

    def cleanup(self):
        """Clean up resources."""
        self.audio.cleanup()

    def notify_error(self, message: str, speak_to_user: bool = True):
        """Log error and optionally notify user via speech."""
        log.error(message)
        if speak_to_user:
            Audio.beep_error()
            self.audio.speak("Sorry, something went wrong. Please try again.")

    def handle_interaction(self) -> bool:
        """Handle one complete interaction cycle. Returns False on fatal error."""
        stream = None
        try:
            # Open audio stream for wake word detection
            stream = self.audio.open_stream()
        except Exception as e:
            log.error(f"Failed to open audio stream: {e}")
            time.sleep(1)  # Prevent tight loop on persistent failure
            return True  # Not fatal, retry

        try:
            print("~ Listening for wake word...")
            detected = self.wake_word.listen(stream)
            if not detected:
                log.warning("Wake word detection returned None")
                return True  # Retry
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"Wake word error: {e}")
            return True  # Retry
        finally:
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass

        print("-" * 40)

        # Pre-warm LLM in background while user speaks
        def warmup_llm():
            try:
                requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "10m"},
                    timeout=30
                )
            except Exception:
                pass  # Best effort, don't block on failure

        threading.Thread(target=warmup_llm, daemon=True).start()

        # Record user speech
        Audio.beep_start()
        if not self.audio.record():
            print("  (no speech detected)")
            Audio.beep_done()
            print()
            return True

        Audio.beep_end()

        benchmark = get_benchmark()

        # Transcribe with timing
        stt_start = time.perf_counter()
        text = self.transcriber.transcribe(TEMP_WAV)
        stt_duration = time.perf_counter() - stt_start

        if not text:
            print("  (transcription empty)")
            Audio.beep_done()
            print()
            return True

        word_count = len(text.split())
        benchmark.log('stt', WHISPER_MODEL, stt_duration, word_count)
        log.info(benchmark.format_stt_log(stt_duration, word_count, WHISPER_MODEL))

        print(f"  You: {text}")

        # Get LLM response with timing
        llm_start = time.perf_counter()
        response = self.llm.chat(text)
        llm_duration = time.perf_counter() - llm_start

        if not response:
            self.notify_error("LLM failed to respond")
            print()
            return True

        benchmark.log('llm', OLLAMA_MODEL, llm_duration)
        log.info(benchmark.format_llm_log(llm_duration, OLLAMA_MODEL))

        print(f"  Assistant: {response}")

        # Speak response with timing
        tts_start = time.perf_counter()
        tts_success = self.audio.speak(response)
        tts_duration = time.perf_counter() - tts_start

        # Extract voice model name from path for logging
        piper_model_name = Path(PIPER_VOICE).stem
        response_word_count = len(response.split())
        benchmark.log('tts', piper_model_name, tts_duration, response_word_count)
        log.info(benchmark.format_tts_log(tts_duration, piper_model_name, response_word_count))

        if not tts_success:
            log.warning("TTS failed, but continuing")

        Audio.beep_done()
        print()
        return True

    def run(self):
        """Main run loop with error recovery."""
        if not self.initialize():
            log.critical("Initialization failed, exiting")
            return

        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            while True:
                try:
                    success = self.handle_interaction()
                    if success:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1

                    if consecutive_errors >= max_consecutive_errors:
                        log.critical(f"Too many consecutive errors ({consecutive_errors}), restarting components...")
                        self.audio.cleanup()
                        time.sleep(2)
                        if self.audio.initialize():
                            consecutive_errors = 0
                            log.info("Audio system restarted successfully")
                        else:
                            log.critical("Failed to restart audio, waiting...")
                            time.sleep(10)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    log.exception(f"Unexpected error in main loop: {e}")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\nBye.")
        finally:
            self.cleanup()


def main():
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
