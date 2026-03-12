"""Main orchestrator: coordinates STT → Quality Gate → Intent Router → LLM → TTS pipeline.

Each call to process_interaction() runs the full pipeline:
  1. TV state snapshot — taken once at the top so routing decisions later are
     consistent, even if the background poller fires mid-interaction.
  2. Speaker identification — runs concurrently with STT in a background thread.
  3. STT — Faster Whisper transcribes the WAV bytes.
  4. Quality Gate — rejects Whisper hallucinations, ambient TV dialogue, garbage.
  5. Intent Router (Tier 1) — maps unambiguous short phrases ("pause", "lights off")
     directly to commands, skipping the LLM entirely (~0ms vs ~1.5s).
  6. LLM (Tier 2+) — Claude processes the transcription with tool_choice=auto.
     Action commands short-circuit with "Done." (1 API call). Narrated commands
     (weather, timers, etc.) get a second API call for the LLM to read results.
  7. TTS — Kokoro synthesizes the response text to WAV.
  8. Routing — TTS is either sent to Sonos (if prefer_sonos=True) or returned
     as base64 audio for the Pi to play locally.
  9. Session summarizer — background thread extracts memorable facts after
     non-follow-up turns (skipped for Tier 1 matches and TV playback).

TV state and routing:
  The TV playback state is polled every 5s by RoomStateManager per-room TV pollers.
  At the start of each interaction the state is snapshotted from the room's state.
  All routing decisions — quality gate TV filter, LLM context note, await_followup
  override, Sonos suppression, session summarizer skip — use this snapshot.

  When TV is playing:
    - Quality gate rejects long transcriptions (>40 words, likely TV dialogue).
    - LLM gets a note ("TV is playing — this may be ambient speech...").
    - Non-critical TTS is suppressed (pure acknowledgments for light/volume commands).
    - await_followup is forced False (prevents follow-up loop during TV audio).
    - Session summarizer is skipped.

await_followup heuristic:
  Replaces LLM-controlled boolean. Set True when the response ends with '?',
  no commands were executed, and the response is short (<20 words). Overridden
  to False when TV is playing.

Sonos audio routing:
  _is_critical_response() classifies the response; non-critical responses are
  suppressed when TV is playing.  Critical responses always route to Sonos.

Volume command redirect:
  When prefer_sonos=True, set_volume and adjust_volume are redirected to Sonos
  commands (set_sonos_volume, adjust_sonos_volume) since the user's audio goes
  through the Sonos, not the Pi speaker.
"""
import logging
import threading
import time
import csv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional, Dict, List

from server.transcription import Transcriber
from server.llm import LLM
from server.synthesis import Synthesizer
from server.pi_callback import PiCallbackClient
from server.quality_gate import filter_transcription
from server.intent_router import route as route_intent
from server.config import (
    BENCHMARK_FILE, SPEAKER_EMBEDDINGS_FILE, SPEAKER_SIMILARITY_THRESHOLD,
    CLAUDE_INPUT_COST_PER_M, CLAUDE_OUTPUT_COST_PER_M,
    SONOS_TTS_OUTPUT,
    SERVER_EXTERNAL_HOST, SERVER_PORT,
)
from server.brain import get_brain
from server.routines import log_command, get_patterns
from server.context import InteractionContext
import server.commands as commands

logger = logging.getLogger(__name__)


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting that TTS would speak literally.

    Kokoro reads asterisks, hashes, and backticks aloud literally.
    Strip them before synthesis so "**bold**" becomes "bold" in speech.
    """
    import re
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)          # **bold** / *italic*
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ## headers
    text = re.sub(r'`+([^`]+)`+', r'\1', text)            # `code`
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # [link text](url)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)   # - bullet points
    return text.strip()

# Optional dependency — Resemblyzer for speaker voice identification
try:
    from server.speaker_id import SpeakerIdentifier
    SPEAKER_ID_AVAILABLE = True
except ImportError:
    SPEAKER_ID_AVAILABLE = False
    logger.info("Speaker identification not available (resemblyzer not installed)")


class Orchestrator:
    """Coordinates the full voice interaction pipeline: STT → LLM → TTS.

    Per-room state (TV playback, Sonos cache, TTS buffer) is managed by
    RoomStateManager — the orchestrator delegates to it via InteractionContext.
    The orchestrator itself holds only shared resources (STT, LLM, TTS engines).
    """

    def __init__(
        self,
        transcriber: Transcriber,
        llm: LLM,
        synthesizer: Synthesizer,
        pi_client: PiCallbackClient,
        room_state_mgr: 'RoomStateManager | None' = None,
        enable_speaker_id: bool = True
    ):
        self.transcriber = transcriber
        self.llm = llm
        self.synthesizer = synthesizer
        self.pi_client = pi_client
        self._room_state_mgr = room_state_mgr

        # Speaker identification — optional, graceful fallback if not installed
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

        self._benchmark_lock = threading.Lock()

        logger.info("Orchestrator initialized")

    def _get_room_state(self, room_id: str = "default"):
        """Get RoomState for a room, falling back to first available."""
        if self._room_state_mgr:
            return self._room_state_mgr.get_or_default(room_id)
        return None

    @property
    def tts_audio(self) -> bytes:
        """Thread-safe read of the default room's TTS buffer."""
        rs = self._get_room_state()
        return rs.tts_audio if rs else b""

    def get_tts_audio(self, room_id: str = "default") -> bytes:
        """Thread-safe read of a room's TTS buffer."""
        rs = self._get_room_state(room_id)
        return rs.tts_audio if rs else b""

    def _get_pi_client_for_ctx(self, ctx: InteractionContext = None) -> Optional[PiCallbackClient]:
        """Get the appropriate PiCallbackClient for the given context.

        If ctx has a callback_url, creates a client for that URL.
        Otherwise falls back to the legacy singleton pi_client.
        """
        if ctx and ctx.callback_url:
            return PiCallbackClient(ctx.callback_url)
        return self.pi_client

    def route_tts_to_sonos(self, audio_data: bytes) -> bool:
        """Public entry point for Sonos TTS routing (used by event loop for timers).

        Returns True if audio was routed to Sonos, False if caller should
        fall back to Pi local playback.
        """
        if not SONOS_TTS_OUTPUT:
            return False
        return self._route_tts_to_sonos(audio_data)

    def process_interaction(self, audio_bytes: bytes, wake_word: str,
                            prefer_sonos: bool = False, ctx: InteractionContext = None) -> Dict:
        """Process a complete voice interaction: STT → LLM → TTS → routing.

        Args:
            audio_bytes: WAV audio from Pi (16kHz, 16-bit mono).
            wake_word: Wake word model that triggered this interaction.
            prefer_sonos: True when Pi is configured with USE_SONOS_OUTPUT=True.
                          Ignored if ctx is provided (uses ctx.prefer_sonos).
            ctx: InteractionContext for multi-client routing. If None, uses legacy
                 global state (backward compat).

        Returns:
            Dict with transcription, response_text, audio_base64 (or empty if Sonos),
            commands_executed, timings, speaker, await_followup, tts_routed,
            tts_duration_seconds, error.
        """
        timings = {}
        commands_executed = []

        # Use context if provided, otherwise fall back to legacy behavior
        if ctx is not None:
            prefer_sonos = ctx.prefer_sonos
            tv_state = ctx.tv_state
        else:
            # Legacy: snapshot TV state from default room's poller
            rs = self._get_room_state()
            tv_state = rs.tv_state if rs else "unknown"

        logger.info(f"Processing interaction (wake word: {wake_word})")

        # Size validation (defense in depth — Pydantic also validates at the API layer)
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
                'tts_routed': False,
                'tts_duration_seconds': 0.0,
                'error': 'Audio file too large'
            }

        # Step 1a: Start speaker ID in background — runs concurrently with STT.
        # STT is slower (~0.5-2s) so speaker ID (typically ~0.3s) usually finishes
        # before we need the result.
        _speaker_result: Dict = {}

        def _run_speaker_id_task():
            try:
                t0 = time.time()
                audio_array = self._audio_bytes_to_numpy(audio_bytes)
                if audio_array is not None:
                    result = self.speaker_identifier.identify(audio_array, sample_rate=16000)
                    _speaker_result['name'] = result.name if result.is_known else None
                    _speaker_result['confidence'] = result.confidence
                    _speaker_result['duration'] = time.time() - t0
            except Exception as e:
                logger.warning(f"Speaker identification failed: {e}")

        speaker_thread = None
        if self.speaker_identifier:
            speaker_thread = threading.Thread(
                target=_run_speaker_id_task, daemon=True, name="SpeakerID"
            )
            speaker_thread.start()

        # Step 1b: STT (concurrent with speaker ID above)
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
                'tts_routed': False,
                'tts_duration_seconds': 0.0,
                'error': 'Speech recognition failed'
            }

        # Truncate absurdly long transcriptions (e.g. if mic stayed open)
        if len(transcription) > 10_000:
            logger.warning(f"Transcription too long ({len(transcription)} chars), truncating")
            transcription = transcription[:10_000] + " [truncated]"

        logger.info(f"Transcribed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
        self._log_benchmark('stt', timings['stt'], transcription)

        # Collect speaker ID result — join with 2s timeout since STT took longer
        speaker_name = None
        speaker_confidence = 0.0
        if speaker_thread:
            speaker_thread.join(timeout=2.0)
            if _speaker_result:
                speaker_name = _speaker_result.get('name')
                speaker_confidence = _speaker_result.get('confidence', 0.0)
                if 'duration' in _speaker_result:
                    timings['speaker_id'] = _speaker_result['duration']
                if speaker_name:
                    logger.info(f"Identified speaker: {speaker_name} ({speaker_confidence:.0%})")
                else:
                    logger.debug(f"Unknown speaker (best match: {speaker_confidence:.0%})")

        # ---- Quality Gate ----
        tv_playing = (tv_state == "playing")
        filtered = filter_transcription(transcription, tv_playing=tv_playing)
        if filtered is None:
            logger.info("Quality gate rejected transcription")
            # When TV is playing, silence is correct — the rejection is likely TV
            # dialogue and any audio response would interrupt viewing.
            # When TV is NOT playing, the user actually spoke but was unclear.
            # Return a brief TTS nudge so they know to try again (pre-cached,
            # zero synthesis latency).
            if not tv_playing:
                nudge_audio = self.synthesizer.synthesize("Didn't catch that.")
                if nudge_audio:
                    tts_routed = False
                    tts_duration = 0.0
                    if prefer_sonos and SONOS_TTS_OUTPUT:
                        tts_routed = self._route_tts_to_sonos(nudge_audio)
                        if tts_routed:
                            tts_duration = self._wav_duration(self.get_tts_audio())
                    import base64
                    return {
                        'transcription': transcription,
                        'response_text': "Didn't catch that.",
                        'audio_base64': '' if tts_routed else base64.b64encode(nudge_audio).decode(),
                        'commands_executed': [],
                        'timings': timings,
                        'speaker': speaker_name,
                        'await_followup': False,
                        'tts_routed': tts_routed,
                        'tts_duration_seconds': tts_duration,
                        'error': None,
                    }
            return {
                'transcription': transcription,
                'response_text': '',
                'audio_base64': '',
                'commands_executed': [],
                'timings': timings,
                'speaker': speaker_name,
                'await_followup': False,
                'tts_routed': False,
                'tts_duration_seconds': 0.0,
                'error': None,
            }
        transcription = filtered

        # ---- Intent Router (Tier 1) ----
        tier1 = route_intent(transcription)
        if tier1 is not None:
            start = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        self._execute_command,
                        tier1.command, prefer_sonos=prefer_sonos, ctx=ctx, **tier1.params,
                    )
                    result = fut.result(timeout=30.0)
                # Check if the command actually failed — don't mask with hardcoded text.
                # Use startswith to avoid false positives on normal result text
                # that happens to contain "error" or "unknown" as substrings.
                result_lower = result.lower().strip() if result else ""
                _error_prefixes = ('error', 'failed', 'not found', 'not available', 'not connected', 'unknown command')
                if any(result_lower.startswith(p) for p in _error_prefixes):
                    response_text = result  # Surface the actual error to the user
                else:
                    response_text = tier1.response
            except FutureTimeout:
                logger.error(f"Tier 1 command '{tier1.command}' timed out after 30s")
                response_text = f"{tier1.command} timed out."
            except Exception as e:
                logger.error(f"Tier 1 command '{tier1.command}' failed: {e}", exc_info=True)
                response_text = "Sorry, that command failed."
            commands_executed.append(tier1.command)
            timings['llm'] = 0.0
            timings['llm_cost'] = 0.0
            await_followup = False
        else:
            # ---- LLM Processing (Tier 2+) ----
            start = time.time()
            tools = commands.get_tools()
            brain = get_brain()
            behavior_rules = brain.get_behavior_rules()
            relevant = brain.retrieve_relevant(transcription)
            relevant_memories = brain.format_relevant(relevant)
            _recent = brain.get_recent_summaries(limit=3)
            recent_summaries = "\n".join(
                f"- {s['data'].get('text', '')}" for s in _recent
            ) if _recent else ""

            def tool_executor(command_name: str, **kwargs) -> str:
                """Wrapper: execute command, track in commands_executed list.

                IMPORTANT: first arg must NOT be called 'name' — several tools
                (set_timer, cancel_timer) pass 'name' as a keyword argument.
                Python raises "got multiple values for argument 'name'" if the
                positional and keyword arg collide.
                """
                commands_executed.append(command_name)
                return self._execute_command(command_name, prefer_sonos=prefer_sonos, ctx=ctx, **kwargs)

            patterns = get_patterns()

            # Inject TV state note into LLM context when content is playing.
            if tv_playing:
                tv_note = (
                    "TV is currently playing. The transcription almost certainly contains "
                    "TV/media dialogue mixed in. Extract only clear, direct commands "
                    "(pause, volume, lights, etc.) and ignore everything else. "
                    "If no clear command is present, respond with 'Didn't catch a command.' "
                    "Do NOT engage with media dialogue or ask questions. "
                    "2-3 word responses only."
                )
                patterns = tv_note + ("\n" + patterns if patterns else "")

            llm_result = self.llm.chat(
                user_text=transcription,
                tools=tools,
                tool_executor=tool_executor,
                behavior_rules=behavior_rules,
                speaker=speaker_name,
                patterns=patterns,
                relevant_memories=relevant_memories,
                recent_summaries=recent_summaries,
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
                    'tts_routed': False,
                    'tts_duration_seconds': 0.0,
                    'error': 'AI processing failed'
                }

            response_text = llm_result.text
            # commands_executed already populated by tool_executor callback

            # await_followup heuristic: only follow up on short, genuine
            # clarification questions (not filler like "anything else?").
            words = response_text.split()
            await_followup = (
                response_text.rstrip().endswith('?')
                and not commands_executed
                and len(words) < 12
            )
            if await_followup:
                lower = response_text.lower()
                _FILLER = (
                    'anything else', 'help you with', 'else can i',
                    'want me to', 'like to know', 'need anything',
                    'can i do', 'what else', 'something else',
                )
                if any(f in lower for f in _FILLER):
                    await_followup = False

            # Log LLM cost (input + output tokens × per-million price)
            usage = self.llm.last_usage
            llm_cost = (
                usage["input_tokens"] * CLAUDE_INPUT_COST_PER_M +
                usage["output_tokens"] * CLAUDE_OUTPUT_COST_PER_M
            ) / 1_000_000
            timings['llm_cost'] = llm_cost
            logger.info(f"LLM response generated")
            self._log_benchmark('llm', timings['llm'], cost=llm_cost)

        # When TV is playing, block follow-up mode.
        if tv_playing and await_followup:
            logger.info("TV is playing — disabling await_followup to prevent follow-up loop")
            await_followup = False

        # Run session summarizer after non-follow-up turns, when TV isn't playing,
        # and when it wasn't a Tier 1 match (no LLM history to summarize).
        if not await_followup and not tv_playing and tier1 is None:
            _snap = self.llm.get_history_snapshot()
            threading.Thread(
                target=self._run_session_summarizer,
                args=(_snap, list(commands_executed)),
                daemon=True,
                name="SessionSummarizer"
            ).start()

        # Step 3: Text-to-Speech
        start = time.time()
        audio_data = self.synthesizer.synthesize_fast(_strip_markdown(response_text))
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
                'await_followup': False,
                'tts_routed': False,
                'tts_duration_seconds': 0.0,
                'error': 'Text-to-speech failed'
            }

        logger.info(f"TTS synthesis complete")
        word_count = len(response_text.split())

        # Load historical stats BEFORE logging current run so the "vs. avg" column
        # in the stats table compares against past data only (not including this run).
        # Also avoids a Windows file-cache race on immediate re-open.
        _tts_stats_pre = self._load_benchmark_stats('tts')
        self._log_benchmark('tts', timings['tts'], word_count=word_count)

        # Step 4: Route TTS to Sonos or return as base64 for Pi local playback
        tts_routed = False
        tts_suppressed = False
        if prefer_sonos and SONOS_TTS_OUTPUT:
            if self._is_critical_response(response_text, commands_executed, await_followup, tv_playing=tv_playing):
                # Critical responses always play (timers, weather, commands with info)
                tts_routed = self._route_tts_to_sonos(audio_data)
            else:
                # Non-critical: suppress when TV is playing (pure action acknowledgments)
                if tv_playing:
                    logger.info("TV is playing — suppressing non-critical Sonos TTS")
                    tts_routed = True  # tell client to skip local playback too
                    tts_suppressed = True
                else:
                    tts_routed = self._route_tts_to_sonos(audio_data)

        # Pack response for transmission
        from shared.utils import encode_audio_base64
        # If Sonos handled it, return empty audio_base64 (Pi won't play locally)
        audio_base64 = "" if tts_routed else encode_audio_base64(audio_data)
        # Duration used by Pi to know how long to sleep before opening follow-up mic.
        # When TTS is suppressed (TV playing), use 0 — nothing is playing.
        # When Sonos routed, use the Sonos buffer. Otherwise use the local audio.
        if tts_suppressed:
            tts_duration_seconds = 0.0
        elif tts_routed:
            tts_duration_seconds = self._wav_duration(self.get_tts_audio())
        else:
            tts_duration_seconds = self._wav_duration(audio_data)

        timings['total'] = timings.get('stt', 0) + timings.get('llm', 0) + timings.get('tts', 0)

        # Log the stats comparison table in background — pure informational
        threading.Thread(
            target=self._log_interaction_stats,
            args=(timings.copy(), transcription, response_text, _tts_stats_pre),
            daemon=True,
            name="BenchmarkStats",
        ).start()

        return {
            'transcription': transcription,
            'response_text': response_text,
            'audio_base64': audio_base64,
            'commands_executed': commands_executed,
            'timings': timings,
            'speaker': speaker_name,
            'await_followup': await_followup,
            'tts_routed': tts_routed,
            'tts_duration_seconds': tts_duration_seconds,
            'error': None
        }

    def process_text_interaction(self, text: str, ctx: InteractionContext = None) -> Dict:
        """Process a text-only interaction (no STT/TTS).

        Runs the LLM pipeline with the text directly, skipping STT and TTS.
        Used by text clients (phone, REST API).

        Args:
            text: User's text input.
            ctx: InteractionContext for room-aware routing.

        Returns:
            Dict with response_text, commands_executed, await_followup, error.
        """
        commands_executed = []

        prefer_sonos = ctx.prefer_sonos if ctx else False

        logger.info(f"Processing text interaction: '{text[:100]}{'...' if len(text) > 100 else ''}'")

        # Quality gate
        tv_state = ctx.tv_state if ctx else "unknown"
        tv_playing = (tv_state == "playing")
        filtered = filter_transcription(text, tv_playing=tv_playing)
        if filtered is None:
            return {
                'response_text': '',
                'commands_executed': [],
                'await_followup': False,
                'error': None,
            }
        text = filtered

        # Intent router (Tier 1)
        tier1 = route_intent(text)
        if tier1 is not None:
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        self._execute_command,
                        tier1.command, prefer_sonos=prefer_sonos, ctx=ctx, **tier1.params,
                    )
                    result = fut.result(timeout=30.0)
                result_lower = result.lower().strip() if result else ""
                _error_prefixes = ('error', 'failed', 'not found', 'not available', 'not connected', 'unknown command')
                if any(result_lower.startswith(p) for p in _error_prefixes):
                    response_text = result
                else:
                    response_text = tier1.response
            except Exception as e:
                logger.error(f"Tier 1 command '{tier1.command}' failed: {e}", exc_info=True)
                response_text = "Sorry, that command failed."
            commands_executed.append(tier1.command)
            return {
                'response_text': response_text,
                'commands_executed': commands_executed,
                'await_followup': False,
                'error': None,
            }

        # LLM (Tier 2+)
        tools = commands.get_tools()
        brain = get_brain()
        behavior_rules = brain.get_behavior_rules()
        relevant = brain.retrieve_relevant(text)
        relevant_memories = brain.format_relevant(relevant)
        _recent = brain.get_recent_summaries(limit=3)
        recent_summaries = "\n".join(
            f"- {s['data'].get('text', '')}" for s in _recent
        ) if _recent else ""

        def tool_executor(command_name: str, **kwargs) -> str:
            commands_executed.append(command_name)
            return self._execute_command(command_name, prefer_sonos=prefer_sonos, ctx=ctx, **kwargs)

        patterns = get_patterns()
        if tv_playing:
            tv_note = (
                "TV is currently playing. Extract only clear, direct commands "
                "and ignore everything else."
            )
            patterns = tv_note + ("\n" + patterns if patterns else "")

        llm_result = self.llm.chat(
            user_text=text,
            tools=tools,
            tool_executor=tool_executor,
            behavior_rules=behavior_rules,
            patterns=patterns,
            relevant_memories=relevant_memories,
            recent_summaries=recent_summaries,
        )

        if not llm_result:
            return {
                'response_text': '',
                'commands_executed': commands_executed,
                'await_followup': False,
                'error': 'AI processing failed',
            }

        response_text = llm_result.text
        words = response_text.split()
        await_followup = (
            response_text.rstrip().endswith('?')
            and not commands_executed
            and len(words) < 12
        )

        return {
            'response_text': response_text,
            'commands_executed': commands_executed,
            'await_followup': await_followup,
            'error': None,
        }

    def _audio_bytes_to_numpy(self, audio_bytes: bytes):
        """Convert WAV audio bytes to float32 numpy array for speaker identification.

        Resemblyzer's SpeakerIdentifier.identify() expects a float32 array
        normalised to [-1.0, 1.0], not raw int16 PCM bytes.
        """
        import io
        import wave
        import numpy as np

        try:
            with io.BytesIO(audio_bytes) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    # int16 → float32, normalised to [-1, 1]
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
        except Exception as e:
            logger.warning(f"Failed to convert audio bytes to numpy: {e}")
            return None

    @staticmethod
    def _wav_duration(audio_data: bytes) -> float:
        """Return duration of WAV audio in seconds.

        Used to tell the Pi client how long to sleep before opening the follow-up
        microphone (so it doesn't start listening while Sonos is still playing).
        """
        import io
        import wave
        try:
            with io.BytesIO(audio_data) as buf:
                with wave.open(buf, 'rb') as wf:
                    return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    @staticmethod
    def _resample_wav_44100(audio_data: bytes) -> bytes:
        """Resample WAV to 44100 Hz for Sonos compatibility.

        Sonos rejects WAVs at non-44100 Hz with no error — it just silently
        fails to play.  Kokoro outputs at 24000 Hz; this resamples to 44100 Hz.
        Uses linear interpolation (fast, adequate quality for speech).
        """
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
            return audio_data  # Already at target rate — no-op

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
    def _pad_wav_for_sonos(audio_data: bytes, pad_seconds: float = 2.0) -> bytes:
        """Append silence so the clip is long enough for Sonos to start playing.

        Sonos Ray requires ~1+ seconds of audio data in its buffer before it
        transitions from TRANSITIONING to PLAYING state.  Short clips (< ~1s)
        silently fail: device reaches TRANSITIONING, then immediately goes to
        STOPPED without playing anything.  Trailing silence is inaudible but
        provides enough data for Sonos to buffer and begin.

        Args:
            audio_data: WAV bytes at any sample rate.
            pad_seconds: Minimum total duration.  Clips already at or above this
                         are returned unchanged.
        """
        import io
        import wave

        with io.BytesIO(audio_data) as buf:
            with wave.open(buf, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                duration = wf.getnframes() / framerate

        if duration >= pad_seconds:
            return audio_data  # Already long enough

        silence_frames = int((pad_seconds - duration) * framerate) * n_channels
        silence = b'\x00' * (silence_frames * sampwidth)

        out = io.BytesIO()
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(frames + silence)
        return out.getvalue()

    @staticmethod
    def _is_critical_response(response_text: str, commands_executed: list, await_followup: bool, tv_playing: bool = False) -> bool:
        """Determine whether TTS must play regardless of TV state.

        Critical = must play even if TV is on:
          - Follow-up prompts (user must hear the question to respond)
          - Any question in the response text (only when TV is NOT playing)
          - Informational commands (weather, time, timers, calculations)
          - Memory operations (confirmation that something was saved)

        Non-critical = pure action acknowledgment that can be silently suppressed
        when TV is playing (lights changed, volume adjusted, show launched):
          - Light changes, Sonos volume, TV launch/playback commands
          - No question, no timer, no information conveyed

        When TV is playing, questions from the LLM are NOT critical — they're
        likely the LLM responding to ambient TV audio.  Only actual informational
        commands justify interrupting TV playback.
        """
        if await_followup:
            return True
        # Questions bypass suppression ONLY when TV is off — when TV is playing,
        # the LLM's questions are usually reactions to ambient TV dialogue and
        # should not steal Sonos audio control from the TV.
        if not tv_playing and '?' in response_text:
            return True
        CRITICAL_COMMANDS = {
            'set_timer', 'cancel_timer', 'list_timers',
            'get_weather', 'get_time', 'calculate',
            'list_feedback', 'list_sonos', 'list_lights',
            'get_volume', 'save_memory', 'forget_memory',
        }
        return any(cmd in CRITICAL_COMMANDS for cmd in commands_executed)

    def _get_sonos_device(self, room_id: str = "default"):
        """Return cached Sonos device for TTS routing, delegating to RoomState."""
        rs = self._get_room_state(room_id)
        if rs:
            return rs.get_sonos_device()
        return None

    def _sonos_play_uri(self, uri: str, title: str, room_id: str = "default") -> bool:
        """Play a URI on the cached Sonos device with DIDL metadata.

        DIDL (Digital Item Description Language) metadata is required by Sonos
        UPnP/AV stack — without it the device may reject the play_uri call.
        On failure, forces cache invalidation and retries once.

        Args:
            uri: HTTP URL the Sonos device will fetch (must be accessible from LAN).
            title: Display name shown on Sonos controller apps.
            room_id: Room whose Sonos device to use.
        """
        device = self._get_sonos_device(room_id)
        if not device:
            logger.warning("Sonos: no device available for play_uri")
            return False
        from soco.data_structures import DidlItem, DidlResource, to_didl_string
        res = DidlResource(uri=uri, protocol_info="http-get:*:audio/wav:*")
        item = DidlItem(title=title, parent_id="S:", item_id="S:TTS", resources=[res])
        meta = to_didl_string(item)
        logger.info(f"Sonos: play_uri → '{device.player_name}' title='{title}' uri={uri}")
        try:
            device.play_uri(uri, meta=meta)
        except Exception as e:
            # Stale device object (e.g. IP changed after DHCP) — invalidate and retry
            logger.warning(f"Sonos: play_uri failed ({e}), forcing rediscovery and retrying")
            rs = self._get_room_state(room_id)
            if rs:
                rs.invalidate_sonos_cache()
            device = self._get_sonos_device(room_id)
            if not device:
                return False
            device.play_uri(uri, meta=meta)
        logger.info(f"Sonos: play_uri dispatched OK")
        return True

    def play_sonos_beep(self, beep_type: str, indicator_light: str = None) -> bool:
        """Play a beep through Sonos, or flash a LIFX light if configured.

        Called by the /api/sonos_beep endpoint (triggered by Pi client beeps).
        When indicator_light is set, LIFX flash is ALWAYS used — it's instant
        (~50ms) vs Sonos play_uri (1-3s startup lag).  This makes LIFX the
        preferred beep indicator when available, regardless of TV state.
        Without indicator_light, falls back to Sonos audio beeps.
        """
        if not SONOS_TTS_OUTPUT:
            return False
        try:
            if indicator_light:
                # LIFX flash — instant visual feedback, no Sonos startup lag
                threading.Thread(
                    target=self._flash_light_indicator,
                    args=(beep_type, indicator_light),
                    daemon=True
                ).start()
                return True
            uri = f"http://{SERVER_EXTERNAL_HOST}:{SERVER_PORT}/audio/beep/{beep_type}"
            return self._sonos_play_uri(uri, f"beep_{beep_type}")
        except Exception as e:
            logger.warning(f"Sonos beep '{beep_type}' failed: {e}")
            return False

    def _flash_light_indicator(self, beep_type: str, light_name: str):
        """Flash a LIFX light as a silent visual indicator (runs in background thread).

        Used as the beep signal when INDICATOR_LIGHT is configured — provides
        instant feedback (~50ms) vs Sonos play_uri (1-3s startup lag).

        Sequence:
          1. Save current light state (color + power).
          2. Set light to indicator color for ~400ms.
          3. Restore original state.
          4. If light was off, turn it back off after restore.

        Color coding:
          start (listening) → blue
          end (processing)  → green
          error             → red
          alert (timer)     → orange

        Args:
            beep_type: Beep type determines indicator color.
            light_name: LIFX light label to flash (from INDICATOR_LIGHT config).
        """
        _COLORS = {
            'start': [43690, 65535, 65535, 6500],  # blue
            'end':   [21845, 52428, 65535, 4000],  # green
            'error': [0,     65535, 65535, 3500],  # red
            'alert': [10923, 65535, 65535, 3500],  # orange
        }
        color = _COLORS.get(beep_type)
        if color is None:
            return
        try:
            from server.commands.lifx_cmd import _resolve_target
            lights = _resolve_target(light_name)
            if not lights:
                logger.debug(f"Indicator light '{light_name}' not found")
                return
            light = lights[0]
            orig_color = light.get_color()
            orig_power = light.get_power()
            light.set_power(65535, duration=0, rapid=True)
            light.set_color(color, duration=100, rapid=True)
            time.sleep(0.4)
            light.set_color(orig_color, duration=150, rapid=True)
            if orig_power < 1000:
                # Light was off — restore to off after color transition
                time.sleep(0.15)
                light.set_power(0, duration=0, rapid=True)
        except Exception as e:
            logger.debug(f"Light indicator flash failed for '{beep_type}': {e}")

    @staticmethod
    def _append_done_chime(wav_data: bytes) -> bytes:
        """Embed a done-chime tone at the end of TTS audio.

        When await_followup=True, the Pi needs a signal to open the microphone
        after the Sonos finishes playing.  A client-side timer is unreliable
        (Sonos startup latency varies) so we embed the chime directly in the
        WAV — the user hears "sure, what name?" followed by the chime, then
        knows to speak.

        Chime: two 1200 Hz tones with a 40ms gap (short, unobtrusive).
        """
        import io
        import math
        import struct
        import wave

        with io.BytesIO(wav_data) as buf:
            with wave.open(buf, 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

        def _tone(freq, dur, vol=0.20):
            n = int(framerate * dur)
            return [int(vol * 32767 * math.sin(2 * math.pi * freq * i / framerate)) for i in range(n)]

        gap = b'\x00' * int(framerate * 0.12) * sampwidth * channels
        chime_samples = _tone(1200, 0.06) + [0] * int(framerate * 0.04) + _tone(1200, 0.06)
        chime_bytes = struct.pack(f'<{len(chime_samples)}h', *chime_samples)

        out = io.BytesIO()
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(frames + gap + chime_bytes)
        return out.getvalue()

    def _route_tts_to_sonos(self, audio_data: bytes, room_id: str = "default") -> bool:
        """Resample, pad, buffer, and trigger Sonos playback.

        Full pipeline:
          1. Check Sonos device available.
          2. Resample to 44100 Hz.
          3. Pad to minimum 2s duration (Sonos startup buffer requirement).
          4. Store in RoomState TTS buffer (served at /audio/tts/{room_id}).
          5. Call play_uri on Sonos.

        Returns True if Sonos accepted the play_uri call, False otherwise.
        On False, caller falls back to returning audio_base64 for Pi local play.
        """
        try:
            if not self._get_sonos_device(room_id):
                logger.warning("No Sonos devices found for TTS output")
                return False

            wav_44k = self._resample_wav_44100(audio_data)

            # Pad to ensure Sonos can buffer enough data before PLAYING state
            wav_44k = self._pad_wav_for_sonos(wav_44k, pad_seconds=2.0)

            # Store in RoomState TTS buffer — served at /audio/tts/{room_id}
            rs = self._get_room_state(room_id)
            if rs:
                rs.tts_audio = wav_44k

            uri = f"http://{SERVER_EXTERNAL_HOST}:{SERVER_PORT}/audio/tts/{room_id}"
            if not self._sonos_play_uri(uri, "Igor", room_id=room_id):
                return False
            device = self._get_sonos_device(room_id)
            device_name = device.player_name if device else "unknown"
            logger.info(f"TTS routed to Sonos '{device_name}'")
            return True

        except Exception as e:
            logger.warning(f"Sonos TTS routing failed: {e}")
            return False

    def _run_session_summarizer(self, history_snapshot: list, commands_executed: list = None):
        """Extract memorable facts from a completed conversation and auto-save to memory.

        Runs in a background daemon thread — never raises, never blocks interactions.
        Only saves NEW facts (skips keys already present in memory).
        Calls LLM.extract_memories() which makes a bare API call expecting JSON back.

        Examples of extracted facts:
          ("preferences", "coffee", "dark roast")
          ("people", "sister_name", "Laura")
          ("schedule", "morning_run", "weekdays at 6am")
        """
        try:
            facts = self.llm.extract_memories(history_snapshot)
            if not facts:
                return

            from server.brain import get_brain
            from server.commands.memory_cmd import _sanitize

            # Only allow auto-save to these categories — prevents adversarial
            # conversation content from writing to "behavior" (system prompt rules).
            _ALLOWED_AUTO_CATEGORIES = frozenset({
                "preferences", "schedule", "people", "personal", "home", "other",
            })

            brain = get_brain()
            saved = []
            for cat, key, val in facts:
                # Sanitize all fields — values are injected into system prompt
                cat = _sanitize(cat, max_len=50).lower().strip()
                key = _sanitize(key, max_len=50).lower().strip().replace(" ", "_")
                val = _sanitize(val, max_len=500)
                if not (cat and key and val):
                    continue
                if cat not in _ALLOWED_AUTO_CATEGORIES:
                    logger.debug(f"Session summarizer rejected category '{cat}'")
                    continue
                # save_memory handles dedup internally — is_update=True means it existed
                entry_id, is_update = brain.save_memory(cat, key, val)
                if not is_update:
                    saved.append(f"[{cat}][{key}]")

            if saved:
                logger.info(f"Session summarizer saved: {', '.join(saved)}")

            # Generate a brief conversation summary for cross-session recall
            self._save_conversation_summary(history_snapshot, commands_executed or [])
        except Exception as e:
            logger.debug(f"Session summarizer failed (non-critical): {e}")

    def _save_conversation_summary(self, history_snapshot: list, commands_executed: list):
        """Generate a brief summary of the conversation and store in brain.

        Runs inside the session summarizer thread — never raises.
        Builds summary from transcript + commands without an extra LLM call.
        """
        try:
            from server.brain import get_brain
            from server.commands.memory_cmd import _sanitize
            brain = get_brain()

            # Build summary from user messages — sanitize to prevent prompt injection
            user_msgs = []
            for msg in history_snapshot:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip() and msg.get("role") == "user":
                    user_msgs.append(_sanitize(content.strip(), max_len=200))

            if not user_msgs:
                return

            first_msg = user_msgs[0][:100]
            if len(user_msgs) == 1:
                summary = first_msg
            else:
                summary = f"{first_msg} (+{len(user_msgs) - 1} follow-ups)"

            if commands_executed:
                cmds = ", ".join(dict.fromkeys(commands_executed))
                summary += f" [{cmds}]"

            # Extract topic tags from user messages and commands
            topic_tags = set()
            if commands_executed:
                topic_tags.update(c.lower() for c in commands_executed)
            for msg_text in user_msgs:
                topic_tags.update(brain._extract_tags(msg_text))
            topic_tags = list(topic_tags)[:15]

            brain.add_summary(summary, topic_tags)
            logger.debug(f"Conversation summary saved: {summary[:80]}...")
        except Exception as e:
            logger.debug(f"Conversation summary failed (non-critical): {e}")

    # When Sonos is the active audio output, Pi volume commands should control
    # the Sonos instead of the Pi's ALSA mixer (Pi speaker not in use).
    _SONOS_VOLUME_REDIRECT = {
        'set_volume': 'set_sonos_volume',
        'adjust_volume': 'adjust_sonos_volume',
    }

    # TV commands that warrant wake word suppression after execution.
    # Netflix startup, YouTube intro, etc. can immediately retrigger the detector.
    _TV_COMMANDS = {
        'tv_power', 'tv_launch', 'tv_playback', 'tv_skip',
        'tv_search_youtube', 'tv_key', 'tv_adb_connect',
    }

    def _execute_command(self, name: str, prefer_sonos: bool = False,
                         ctx: InteractionContext = None, **kwargs) -> str:
        """Execute a command, handling Sonos redirect and Pi hardware RPC.

        Three paths:
          1. Sonos redirect: if prefer_sonos and command is a volume command,
             transparently redirect to the equivalent Sonos command.
          2. Hardware RPC: set_volume/get_volume must run on Pi (ALSA lives there).
             Forwarded via PiCallbackClient.hardware_control().
          3. Local execution: all other commands run directly on the server.
             After TV commands, suppresses Pi wake word for 20s in a background thread.

        Args:
            name: Command name as called by the LLM.
            prefer_sonos: True when Pi output is routed through Sonos.
            ctx: InteractionContext for room-aware routing (optional).
            **kwargs: Command parameters from LLM tool call.

        Returns:
            String result from the command (shown to LLM as tool_result).
        """
        if ctx is not None:
            prefer_sonos = ctx.prefer_sonos

        # Redirect Pi volume to Sonos when audio output is Sonos
        if prefer_sonos and name in self._SONOS_VOLUME_REDIRECT:
            name = self._SONOS_VOLUME_REDIRECT[name]

        # Volume commands must execute on Pi — ALSA mixer is not accessible from server
        if name in ('set_volume', 'get_volume'):
            logger.info(f"Routing hardware command '{name}' to Pi")
            # Use context callback_url if available, otherwise legacy pi_client
            pi = self._get_pi_client_for_ctx(ctx)
            result = pi.hardware_control(name, kwargs) if pi else None
            threading.Thread(target=log_command, args=(name,), daemon=True).start()
            return result if result else f"Failed to execute {name} on Pi"

        # All other commands execute locally on the server
        # Inject _ctx into kwargs so room-aware commands can use it
        try:
            result = commands.execute(name, _ctx=ctx, **kwargs)
            threading.Thread(target=log_command, args=(name,), daemon=True).start()
        except Exception as e:
            logger.error(f"Command '{name}' failed: {e}", exc_info=True)
            return "Command failed."

        # After TV commands, suppress wake word on Pi so startup audio doesn't retrigger.
        # Runs in a background thread — fire-and-forget, non-blocking.
        if name in self._TV_COMMANDS:
            pi = self._get_pi_client_for_ctx(ctx)
            if pi:
                threading.Thread(
                    target=pi.suppress_wakeword,
                    kwargs={"seconds": 20.0},
                    daemon=True,
                    name="SuppressWakeword",
                ).start()

        return result

    def _log_benchmark(self, stage: str, duration: float, transcription: Optional[str] = None, word_count: Optional[int] = None, cost: Optional[float] = None):
        """Append a performance data point to data/benchmark.csv.

        Creates the file with headers on first write.  Thread-safe via _benchmark_lock.
        Non-critical — errors are logged but never propagate.
        """
        try:
            if transcription and not word_count:
                word_count = len(transcription.split())

            if stage == 'stt':
                model = self.transcriber.model_name
            elif stage == 'llm':
                model = self.llm.model
            elif stage == 'tts':
                model = 'kokoro'
            else:
                model = 'unknown'

            per_word = (duration / word_count) if word_count and word_count > 0 else None

            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

            with self._benchmark_lock:
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
        """Load historical average duration and per-word time for a pipeline stage.

        Used by _log_interaction_stats() to show "vs. avg" deltas in the
        performance table.  Returns zeros if no data exists yet.
        """
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

    def _log_interaction_stats(self, timings: Dict[str, float], transcription: str, response_text: str, tts_stats: Dict = None):
        """Log a formatted performance summary table comparing current vs. historical averages.

        Runs in a background thread — never blocks the response.
        Example output:
          ┌───────┬──────────┬────────┬──────────┐
          │ Stage │   Time   │vs. Avg │ Per Word │
          ├───────┼──────────┼────────┼──────────┤
          │ STT   │   0.62s  │  ↓15%  │  0.031s  │
          │ LLM   │   2.14s  │   ~    │  $0.001  │
          │ TTS   │   0.83s  │  ↑ 5%  │  0.025s  │
          └───────┴──────────┴────────┴──────────┘
        """
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
        if tts_stats is None:
            tts_stats = self._load_benchmark_stats('tts')

        def cmp(current, avg) -> str:
            """Return fixed-width comparison arrow: ↓ fast, ↑ slow, ~ near-average."""
            if avg == 0:
                return "        "
            pct = ((current - avg) / avg) * 100
            if pct < -5:
                return f"\u2193{min(abs(pct), 999):.0f}%".rjust(6)
            elif pct > 5:
                return f"\u2191{min(pct, 999):.0f}%".rjust(6)
            return "       ~"

        def fmt_time(t: float) -> str:
            return f"{t:7.2f}s"

        def fmt_pw(pw) -> str:
            return f"{pw:.3f}s" if pw is not None else "   —  "

        W = "┌───────┬──────────┬────────┬──────────┐"
        H = "│ Stage │   Time   │vs. Avg │ Per Word │"
        S = "├───────┼──────────┼────────┼──────────┤"
        B = "└───────┴──────────┴────────┴──────────┘"

        def data_row(stage, t, avg_t, pw) -> str:
            return f"│ {stage:<5s} │{fmt_time(t)}  │{cmp(t, avg_t)}  │{fmt_pw(pw):^10s}│"

        llm_cost_str = f"${llm_cost:.3f}"
        llm_row = f"│ {'LLM':<5s} │{fmt_time(llm_time)}  │{cmp(llm_time, llm_stats['avg_duration'])}  │{llm_cost_str:^10s}│"

        logger.info(f"Interaction complete in {total:.2f}s")
        logger.info(W)
        logger.info(H)
        logger.info(S)
        logger.info(data_row("STT", stt_time, stt_stats['avg_duration'], stt_per_word))
        logger.info(llm_row)
        logger.info(data_row("TTS", tts_time, tts_stats['avg_duration'], tts_per_word))
        logger.info(B)

    def get_conversation_history(self) -> List[Dict]:
        """Return current LLM conversation history (for API inspection)."""
        return self.llm.get_history()

    def clear_conversation_history(self):
        """Reset LLM conversation history (for API /api/conversation/clear)."""
        self.llm.clear_history()
