"""Main orchestrator for processing voice interactions."""
import logging
import time
import csv
from typing import Optional, Dict, List

from server.transcription import Transcriber
from server.llm import LLM
from server.synthesis import Synthesizer
from server.pi_callback import PiCallbackClient
from server.config import BENCHMARK_FILE
from server.commands.memory_cmd import load_persistent_memory
import server.commands as commands

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the full voice interaction pipeline: STT -> LLM -> TTS."""

    def __init__(
        self,
        transcriber: Transcriber,
        llm: LLM,
        synthesizer: Synthesizer,
        pi_client: PiCallbackClient
    ):
        self.transcriber = transcriber
        self.llm = llm
        self.synthesizer = synthesizer
        self.pi_client = pi_client
        logger.info("Orchestrator initialized")

    def process_interaction(self, audio_bytes: bytes, wake_word: str) -> Dict:
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
                'error': 'Speech recognition failed'
            }

        # Validate transcription length
        if len(transcription) > 10_000:  # Max 10k characters
            logger.error(f"Transcription too long: {len(transcription)} chars")
            transcription = transcription[:10_000]  # Truncate
            logger.warning("Transcription truncated to 10k chars")

        logger.info(f"Transcribed: '{transcription[:100]}...'")  # Log first 100 chars only
        self._log_benchmark('stt', timings['stt'], transcription)

        # Step 2: LLM Processing
        start = time.time()
        tools = commands.get_tools()
        persistent_memory = load_persistent_memory()

        # Create tool executor with Pi callback support
        def tool_executor(name: str, **kwargs) -> str:
            """Execute command and track it."""
            commands_executed.append(name)
            return self._execute_command(name, **kwargs)

        response_text = self.llm.chat(
            user_text=transcription,
            tools=tools,
            tool_executor=tool_executor,
            persistent_memory=persistent_memory
        )
        timings['llm'] = time.time() - start

        if not response_text:
            logger.error("LLM failed to generate response")
            return {
                'transcription': transcription,
                'response_text': '',
                'audio_base64': '',
                'commands_executed': commands_executed,
                'timings': timings,
                'error': 'AI processing failed'
            }

        logger.info(f"LLM response generated")
        self._log_benchmark('llm', timings['llm'])

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
                'error': 'Text-to-speech failed'
            }

        logger.info(f"TTS synthesis complete")
        word_count = len(response_text.split())
        self._log_benchmark('tts', timings['tts'], word_count=word_count)

        # Convert audio to base64 for transmission
        from shared.utils import encode_audio_base64
        audio_base64 = encode_audio_base64(audio_data)

        # Calculate total time
        timings['total'] = sum(timings.values())
        logger.info(f"Interaction complete in {timings['total']:.2f}s (STT: {timings['stt']:.2f}s, LLM: {timings['llm']:.2f}s, TTS: {timings['tts']:.2f}s)")

        return {
            'transcription': transcription,
            'response_text': response_text,
            'audio_base64': audio_base64,
            'commands_executed': commands_executed,
            'timings': timings,
            'error': None
        }

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

    def _log_benchmark(self, stage: str, duration: float, transcription: Optional[str] = None, word_count: Optional[int] = None):
        """
        Log performance benchmark to CSV.

        Args:
            stage: 'stt', 'llm', or 'tts'
            duration: Duration in seconds
            transcription: For STT, the transcribed text (for word count)
            word_count: For TTS, number of words spoken
        """
        try:
            # Calculate word count if not provided
            if transcription and not word_count:
                word_count = len(transcription.split())

            # Determine model name based on stage
            if stage == 'stt':
                model = self.transcriber.model_name
            elif stage == 'llm':
                model = self.llm.model
            elif stage == 'tts':
                model = 'piper'
            else:
                model = 'unknown'

            # Calculate per-word duration for STT/TTS
            per_word = None
            if word_count and word_count > 0:
                per_word = duration / word_count

            # Append to CSV
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Create file with header if it doesn't exist
            if not BENCHMARK_FILE.exists():
                with open(BENCHMARK_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'stage', 'model', 'duration_s', 'word_count', 'per_word_s'])

            # Append data
            with open(BENCHMARK_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    stage,
                    model,
                    f"{duration:.3f}",
                    word_count if word_count else '',
                    f"{per_word:.3f}" if per_word else ''
                ])

        except Exception as e:
            logger.error(f"Failed to log benchmark: {e}")

    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history from LLM."""
        return self.llm.get_history()

    def set_conversation_history(self, history: List[Dict]):
        """Set conversation history (for restoring state)."""
        self.llm.set_history(history)

    def clear_conversation_history(self):
        """Clear conversation history."""
        self.llm.clear_history()
