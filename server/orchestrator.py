"""Main orchestrator for processing voice interactions."""
import logging
import time
import csv
from typing import Optional, Dict, List

from server.transcription import Transcriber
from server.llm import LLM
from server.synthesis import Synthesizer
from server.pi_callback import PiCallbackClient
from server.commands.adb_cmd import _get_tv_playback_state
from server.config import (
    BENCHMARK_FILE, SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD,
    CLAUDE_INPUT_COST_PER_M, CLAUDE_OUTPUT_COST_PER_M,
    SONOS_TTS_OUTPUT, SONOS_DEFAULT_ZONE, SONOS_DISCOVERY_CACHE_TTL,
    SERVER_EXTERNAL_HOST, SERVER_PORT, DATA_DIR,
)
from server.commands.memory_cmd import load_persistent_memory
import server.commands as commands

logger = logging.getLogger(__name__)

# Try to import speaker identification (optional dependency)
try:
    from server.speaker_id import SpeakerIdentifier
    SPEAKER_ID_AVAILABLE = True
except ImportError:
    SPEAKER_ID_AVAILABLE = False
    logger.info("Speaker identification not available (resemblyzer not installed)")


class Orchestrator:
    """Coordinates the full voice interaction pipeline: STT -> LLM -> TTS."""

    def __init__(
        self,
        transcriber: Transcriber,
        llm: LLM,
        synthesizer: Synthesizer,
        pi_client: PiCallbackClient,
        enable_speaker_id: bool = True
    ):
        self.transcriber = transcriber
        self.llm = llm
        self.synthesizer = synthesizer
        self.pi_client = pi_client

        # Initialize speaker identification if available
        self.speaker_identifier = None
        if enable_speaker_id and SPEAKER_ID_AVAILABLE:
            try:
                self.speaker_identifier = SpeakerIdentifier(
                    SPEAKER_EMBEDDINGS_FILE,
                    SPEAKER_SIMILARITY_THRESHOLD
                )
                speakers = self.speaker_identifier.list_speakers()
                if speakers:
                    logger.info(f"Speaker identification enabled. Enrolled: {speakers}")
                else:
                    logger.info("Speaker identification enabled (no speakers enrolled)")
            except Exception as e:
                logger.warning(f"Failed to initialize speaker identification: {e}")

        # Sonos device cache for TTS output routing
        self._sonos_device = None
        self._sonos_cache_time = 0.0

        logger.info("Orchestrator initialized")

    def process_interaction(self, audio_bytes: bytes, wake_word: str, prefer_sonos: bool = False) -> Dict:
        """
        Process a complete voice interaction.

        Pipeline:
        1. Transcribe audio -> text
        2. Send to LLM with tools
        3. Execute any commands
        4. Synthesize response -> audio
        5. Log benchmarks

        Args:
            audio_bytes: WAV audio data from Pi
            wake_word: Wake word that was detected

        Returns:
            Dictionary with transcription, response_text, audio_base64, timings, etc.
        """
        timings = {}
        commands_executed = []

        # Step 1: Speech-to-Text
        logger.info(f"Processing interaction (wake word: {wake_word})")

        # Validate audio size (defense in depth - Pydantic also validates)
        if len(audio_bytes) > 10_000_000:  # 10MB max
            logger.error(f"Audio too large: {len(audio_bytes)} bytes")
            return {
                'transcription': '',
                'response_text': '',
                'audio_base64': '',
                'commands_executed': [],
                'timings': {},
                'speaker': None,
                'await_followup': False,
                'error': 'Audio file too large'
            }

        start = time.time()
        transcription = self.transcriber.transcribe_bytes(audio_bytes)
        timings['stt'] = time.time() - start

        if not transcription:
            logger.error("Transcription failed")
            return {
                'transcription': '',
                'response_text': '',
                'audio_base64': '',
                'commands_executed': [],
                'timings': timings,
                'speaker': None,
                'await_followup': False,
                'error': 'Speech recognition failed'
            }

        # Validate transcription length
        if len(transcription) > 10_000:  # Max 10k characters
            logger.error(f"Transcription too long: {len(transcription)} chars")
            transcription = transcription[:10_000]  # Truncate
            logger.warning("Transcription truncated to 10k chars")

        logger.info(f"Transcribed: '{transcription}'")
        self._log_benchmark('stt', timings['stt'], transcription)

        # Step 1b: Speaker Identification (optional, runs in parallel with STT conceptually)
        speaker_name = None
        speaker_confidence = 0.0
        if self.speaker_identifier:
            try:
                start_speaker = time.time()
                # Convert audio bytes to numpy for speaker ID
                audio_array = self._audio_bytes_to_numpy(audio_bytes)
                if audio_array is not None:
                    result = self.speaker_identifier.identify(audio_array, sample_rate=16000)
                    speaker_name = result.name if result.is_known else None
                    speaker_confidence = result.confidence
                    timings['speaker_id'] = time.time() - start_speaker
                    if speaker_name:
                        logger.info(f"Identified speaker: {speaker_name} ({speaker_confidence:.0%})")
                    else:
                        logger.debug(f"Unknown speaker (best match: {result.confidence:.0%})")
            except Exception as e:
                logger.warning(f"Speaker identification failed: {e}")

        # Step 2: LLM Processing
        start = time.time()
        tools = commands.get_tools()
        persistent_memory = load_persistent_memory()

        # Create tool executor with Pi callback support
        def tool_executor(name: str, **kwargs) -> str:
            """Execute command and track it."""
            commands_executed.append(name)
            return self._execute_command(name, **kwargs)

        llm_result = self.llm.chat(
            user_text=transcription,
            tools=tools,
            tool_executor=tool_executor,
            persistent_memory=persistent_memory,
            speaker=speaker_name
        )
        timings['llm'] = time.time() - start

        if not llm_result:
            logger.error("LLM failed to generate response")
            return {
                'transcription': transcription,
                'response_text': '',
                'audio_base64': '',
                'commands_executed': commands_executed,
                'timings': timings,
                'speaker': speaker_name,
                'await_followup': False,
                'error': 'AI processing failed'
            }

        response_text, await_followup = llm_result

        usage = self.llm.last_usage
        llm_cost = (
            usage["input_tokens"] * CLAUDE_INPUT_COST_PER_M +
            usage["output_tokens"] * CLAUDE_OUTPUT_COST_PER_M
        ) / 1_000_000
        timings['llm_cost'] = llm_cost
        logger.info(f"LLM response generated")
        self._log_benchmark('llm', timings['llm'], cost=llm_cost)

        # Step 3: Text-to-Speech
        start = time.time()
        audio_data = self.synthesizer.synthesize(response_text)
        timings['tts'] = time.time() - start

        if not audio_data:
            logger.error("TTS failed")
            return {
                'transcription': transcription,
                'response_text': response_text,
                'audio_base64': '',
                'commands_executed': commands_executed,
                'timings': timings,
                'speaker': speaker_name,
                'await_followup': await_followup,
                'error': 'Text-to-speech failed'
            }

        logger.info(f"TTS synthesis complete")
        word_count = len(response_text.split())
        self._log_benchmark('tts', timings['tts'], word_count=word_count)

        # Route TTS to Sonos if client requested it and server allows it
        tts_routed = False
        if prefer_sonos and SONOS_TTS_OUTPUT:
            if self._is_critical_response(response_text, commands_executed, await_followup):
                tts_routed = self._route_tts_to_sonos(audio_data)
            else:
                tv_state = _get_tv_playback_state()
                if tv_state == "playing":
                    logger.info("TV is playing — suppressing non-critical Sonos TTS")
                    tts_routed = True  # tell client to skip local playback too
                else:
                    tts_routed = self._route_tts_to_sonos(audio_data)

        # Convert audio to base64 for transmission (empty if Sonos handled it)
        from shared.utils import encode_audio_base64
        audio_base64 = "" if tts_routed else encode_audio_base64(audio_data)

        # Calculate total time (exclude non-time fields like llm_cost)
        timings['total'] = timings.get('stt', 0) + timings.get('llm', 0) + timings.get('tts', 0)
        # Log with detailed statistics
        self._log_interaction_stats(timings, transcription, response_text)

        return {
            'transcription': transcription,
            'response_text': response_text,
            'audio_base64': audio_base64,
            'commands_executed': commands_executed,
            'timings': timings,
            'speaker': speaker_name,
            'await_followup': await_followup,
            'tts_routed': tts_routed,
            'error': None
        }

    def _audio_bytes_to_numpy(self, audio_bytes: bytes):
        """Convert WAV audio bytes to numpy array for speaker identification."""
        import io
        import wave
        import numpy as np

        try:
            # Parse WAV header and extract PCM data
            with io.BytesIO(audio_bytes) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    # Read all frames
                    frames = wav_file.readframes(wav_file.getnframes())
                    # Convert to numpy array (assuming 16-bit PCM)
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
        except Exception as e:
            logger.warning(f"Failed to convert audio bytes to numpy: {e}")
            return None

    @staticmethod
    def _resample_wav_44100(audio_data: bytes) -> bytes:
        """Resample WAV to 44100 Hz for Sonos compatibility (Sonos rejects 22050 Hz)."""
        import io
        import wave
        import numpy as np

        with io.BytesIO(audio_data) as buf:
            with wave.open(buf, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

        if framerate == 44100:
            return audio_data

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        ratio = 44100 / framerate
        n_out = int(len(samples) * ratio)
        x_old = np.arange(len(samples))
        x_new = np.linspace(0, len(samples) - 1, n_out)
        resampled = np.interp(x_new, x_old, samples).astype(np.int16)

        out = io.BytesIO()
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(44100)
            wf.writeframes(resampled.tobytes())
        return out.getvalue()

    @staticmethod
    def _is_critical_response(response_text: str, commands_executed: list, await_followup: bool) -> bool:
        """Return True if TTS must play regardless of TV state.

        Critical: timer actions, informational queries, follow-up prompts, questions.
        Non-critical: pure action acknowledgments (lights, volume, launch, playback).
        """
        if await_followup:
            return True
        if '?' in response_text:
            return True
        CRITICAL_COMMANDS = {
            'set_timer', 'cancel_timer', 'list_timers',
            'get_weather', 'get_time', 'calculate',
            'list_feedback', 'list_sonos', 'list_lights',
            'get_volume', 'save_memory', 'forget_memory',
        }
        return any(cmd in CRITICAL_COMMANDS for cmd in commands_executed)

    def _get_sonos_device(self):
        """Return cached Sonos device, rediscovering if stale. Returns None if unavailable."""
        try:
            import soco
            now = time.time()
            if self._sonos_device is None or now - self._sonos_cache_time > SONOS_DISCOVERY_CACHE_TTL:
                devices = list(soco.discover(timeout=2) or [])
                self._sonos_device = None
                for d in devices:
                    if d.player_name == SONOS_DEFAULT_ZONE:
                        self._sonos_device = d
                        break
                if self._sonos_device is None and devices:
                    self._sonos_device = devices[0]
                self._sonos_cache_time = now
            return self._sonos_device
        except Exception as e:
            logger.warning(f"Sonos discovery failed: {e}")
            return None

    def _sonos_play_uri(self, uri: str, title: str) -> bool:
        """Play a URI on the cached Sonos device with proper DIDL metadata."""
        device = self._get_sonos_device()
        if not device:
            logger.warning("No Sonos device available")
            return False
        from soco.data_structures import DidlItem, DidlResource, to_didl_string
        res = DidlResource(uri=uri, protocol_info="http-get:*:audio/wav:*")
        item = DidlItem(title=title, parent_id="S:", item_id="S:TTS")
        item.resources.append(res)  # resources is not a constructor kwarg — must append
        device.play_uri(uri, meta=to_didl_string(item))
        return True

    def play_sonos_beep(self, beep_type: str) -> bool:
        """Play a beep through Sonos. Fire-and-forget; returns True if dispatched."""
        if not SONOS_TTS_OUTPUT:
            return False
        try:
            uri = f"http://{SERVER_EXTERNAL_HOST}:{SERVER_PORT}/audio/beep/{beep_type}"
            return self._sonos_play_uri(uri, f"beep_{beep_type}")
        except Exception as e:
            logger.warning(f"Sonos beep '{beep_type}' failed: {e}")
            return False

    def _route_tts_to_sonos(self, audio_data: bytes) -> bool:
        """Resample TTS audio, save it, and trigger Sonos playback. Returns True if successful."""
        try:
            if not self._get_sonos_device():
                logger.warning("No Sonos devices found for TTS output")
                return False

            # Sonos requires 44100 Hz; Piper outputs 22050 Hz
            wav_44k = self._resample_wav_44100(audio_data)
            tts_path = DATA_DIR / "tts_latest.wav"
            tmp_path = DATA_DIR / "tts_latest.tmp"
            tmp_path.write_bytes(wav_44k)
            tmp_path.replace(tts_path)

            uri = f"http://{SERVER_EXTERNAL_HOST}:{SERVER_PORT}/audio/tts_latest"
            self._sonos_play_uri(uri, "Dr. Butts")
            logger.info(f"TTS routed to Sonos '{self._sonos_device.player_name}'")
            return True

        except Exception as e:
            logger.warning(f"Sonos TTS routing failed: {e}")
            return False

    def _execute_command(self, name: str, **kwargs) -> str:
        """
        Execute a command with Pi callback support.

        Special handling for hardware commands that need to run on Pi.
        """
        # Hardware commands get routed to Pi via RPC
        if name in ('set_volume', 'get_volume'):
            logger.info(f"Routing hardware command '{name}' to Pi")
            result = self.pi_client.hardware_control(name, kwargs)
            return result if result else f"Failed to execute {name} on Pi"

        # All other commands execute locally
        try:
            return commands.execute(name, **kwargs)
        except Exception as e:
            logger.error(f"Command '{name}' failed: {e}")
            return f"Error: {e}"

    def _log_benchmark(self, stage: str, duration: float, transcription: Optional[str] = None, word_count: Optional[int] = None, cost: Optional[float] = None):
        """Log performance benchmark to CSV."""
        try:
            if transcription and not word_count:
                word_count = len(transcription.split())

            if stage == 'stt':
                model = self.transcriber.model_name
            elif stage == 'llm':
                model = self.llm.model
            elif stage == 'tts':
                model = 'piper'
            else:
                model = 'unknown'

            per_word = (duration / word_count) if word_count and word_count > 0 else None

            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

            if not BENCHMARK_FILE.exists():
                with open(BENCHMARK_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'stage', 'model', 'duration_s', 'word_count', 'per_word_s', 'cost_usd'])

            with open(BENCHMARK_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, stage, model,
                    f"{duration:.3f}",
                    word_count if word_count else '',
                    f"{per_word:.3f}" if per_word else '',
                    f"{cost:.6f}" if cost is not None else '',
                ])

        except Exception as e:
            logger.error(f"Failed to log benchmark: {e}")

    def _load_benchmark_stats(self, stage: str) -> Dict[str, float]:
        """Load historical statistics for a stage from benchmark.csv."""
        durations = []
        per_word_times = []

        if not BENCHMARK_FILE.exists():
            return {'avg_duration': 0, 'avg_per_word': 0}

        try:
            with open(BENCHMARK_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['stage'] == stage:
                        if row['duration_s']:
                            durations.append(float(row['duration_s']))
                        if row['per_word_s']:
                            per_word_times.append(float(row['per_word_s']))

            return {
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'avg_per_word': sum(per_word_times) / len(per_word_times) if per_word_times else 0
            }
        except Exception as e:
            logger.debug(f"Failed to load stats for {stage}: {e}")
            return {'avg_duration': 0, 'avg_per_word': 0}

    def _log_interaction_stats(self, timings: Dict[str, float], transcription: str, response_text: str):
        """Log interaction summary with performance comparison to historical averages."""
        total = timings['total']
        stt_time = timings['stt']
        llm_time = timings['llm']
        tts_time = timings['tts']
        llm_cost = timings.get('llm_cost', 0.0)

        stt_words = len(transcription.split()) if transcription else 0
        tts_words = len(response_text.split()) if response_text else 0
        stt_per_word = stt_time / stt_words if stt_words > 0 else None
        tts_per_word = tts_time / tts_words if tts_words > 0 else None

        stt_stats = self._load_benchmark_stats('stt')
        llm_stats = self._load_benchmark_stats('llm')
        tts_stats = self._load_benchmark_stats('tts')

        def cmp(current, avg) -> str:
            """Return fixed-width 8-char comparison string."""
            if avg == 0:
                return "        "
            pct = ((current - avg) / avg) * 100
            if pct < -5:
                return f"\u2193{min(abs(pct), 999):.0f}%".rjust(8)
            elif pct > 5:
                return f"\u2191{min(pct, 999):.0f}%".rjust(8)
            return "       ~"

        def fmt_time(t: float) -> str:
            return f"{t:9.2f}s"

        def fmt_pw(pw) -> str:
            return f"{pw:.3f}s" if pw is not None else "   —  "

        # ┌───────┬──────────┬────────┬──────────┐
        # │ Stage │   Time   │vs. Avg │ Per Word │
        W = "┌───────┬──────────┬────────┬──────────┐"
        H = "│ Stage │   Time   │vs. Avg │ Per Word │"
        S = "├───────┼──────────┼────────┼──────────┤"
        B = "└───────┴──────────┴────────┴──────────┘"

        def data_row(stage, t, avg_t, pw) -> str:
            return (
                f"│ {stage:<5s} │{fmt_time(t)}│{cmp(t, avg_t)}│{fmt_pw(pw):^10s}│"
            )

        cost_row = f"│ Cost  │ ${llm_cost:.5f} │        │          │"

        logger.info(f"Interaction complete in {total:.2f}s | cost ${llm_cost:.5f}")
        logger.info(W)
        logger.info(H)
        logger.info(S)
        logger.info(data_row("STT", stt_time, stt_stats['avg_duration'], stt_per_word))
        logger.info(data_row("LLM", llm_time, llm_stats['avg_duration'], None))
        logger.info(data_row("TTS", tts_time, tts_stats['avg_duration'], tts_per_word))
        logger.info(cost_row)
        logger.info(B)

    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history from LLM."""
        return self.llm.get_history()

    def clear_conversation_history(self):
        """Clear conversation history."""
        self.llm.clear_history()
