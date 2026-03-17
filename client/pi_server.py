"""HTTP server on Pi for receiving callbacks from the PC server.

The Pi runs this Flask server alongside the wake word detection loop.  The PC
server sends requests here to:
  - Play synthesized TTS audio (timer alerts, direct audio delivery)
  - Execute hardware commands (set/get ALSA mixer volume)
  - Play local beep sounds (alert, error, start, end)
  - Suppress wake word detection (after TV commands to prevent false triggers)

Security:
  All endpoints except /api/health reject requests from IPs other than SERVER_HOST.
  The Pi is on a home LAN with a known server IP, so this is a simple effective guard.

Threading:
  Served by Waitress (4 threads) so simultaneous requests (e.g. a timer alert
  arriving while a beep is playing) are handled without blocking each other.
"""
import logging
import time
from flask import Flask, request, jsonify

from shared.models import (
    PlayAudioRequest,
    PlayAudioResponse,
    HardwareControlRequest,
    HardwareControlResponse,
    PlayBeepRequest,
    PlayBeepResponse,
    HealthCheckResponse,
    Status,
    HealthStatus,
    BeepType
)
from shared.utils import decode_audio_base64
from client.hardware import HardwareController
from client.config import SERVER_HOST, WAKE_SAMPLES_DIR

logger = logging.getLogger(__name__)


def create_pi_app(audio_system, start_time: float) -> Flask:
    """Create the Flask Pi callback application.

    Args:
        audio_system: Audio instance from client/audio.py (PyAudio wrapper).
        start_time: Unix timestamp of server startup (for uptime in health check).

    Returns:
        Configured Flask app ready for Waitress.
    """
    app = Flask(__name__)
    hardware = HardwareController()

    @app.before_request
    def restrict_to_server():
        """IP allowlist: only the known PC server may call sensitive endpoints.

        /api/health is exempt so monitoring tools can reach it without credentials.
        All other endpoints return 403 if the caller is not SERVER_HOST.
        """
        if request.endpoint == 'health':
            return  # Health check open for monitoring
        if request.remote_addr != SERVER_HOST:
            logger.warning(f"Rejected request from {request.remote_addr} (expected {SERVER_HOST})")
            return jsonify({'error': 'Unauthorized'}), 403

    def _bad_json():
        return jsonify({'error': 'Invalid JSON body'}), 400

    @app.route('/api/play_audio', methods=['POST'])
    def play_audio():
        """Receive and play TTS audio sent from the PC server.

        Called by PiCallbackClient.play_audio() for timer alerts.
        Decodes base64 WAV and plays it through PyAudio directly.

        Request body: PlayAudioRequest (audio_base64, message, priority)
        Response: PlayAudioResponse (status, played_at, error)
        """
        try:
            data = request.get_json()
            if data is None:
                return _bad_json()
            req = PlayAudioRequest(**data)

            logger.info(f"Received audio playback request: '{req.message[:50]}'")

            try:
                audio_bytes = decode_audio_base64(req.audio_base64)
            except Exception:
                return jsonify(PlayAudioResponse(
                    status=Status.ERROR, played_at=time.time(), error="Invalid audio encoding"
                ).model_dump()), 400

            success = audio_system.play_audio_bytes(audio_bytes)
            if success:
                return jsonify(PlayAudioResponse(status=Status.SUCCESS, played_at=time.time()).model_dump())
            else:
                return jsonify(PlayAudioResponse(
                    status=Status.ERROR, played_at=time.time(), error="Audio playback failed"
                ).model_dump()), 500

        except Exception as e:
            logger.error(f"Error in play_audio: {e}", exc_info=True)
            return jsonify(PlayAudioResponse(
                status=Status.ERROR, played_at=time.time(), error="Internal server error"
            ).model_dump()), 500

    @app.route('/api/hardware_control', methods=['POST'])
    def hardware_control():
        """Execute a hardware command (ALSA volume) on behalf of the PC server.

        The PC server doesn't have access to the Pi's ALSA mixer, so volume
        commands are forwarded here via RPC.  Only 'set_volume' and 'get_volume'
        are whitelisted in HardwareControlRequest — any other command is rejected
        by Pydantic validation before reaching this handler.

        Request body: HardwareControlRequest (command, parameters)
        Response: HardwareControlResponse (status, result, error)
        """
        try:
            data = request.get_json()
            if data is None:
                return _bad_json()
            req = HardwareControlRequest(**data)

            logger.info(f"Received hardware command: {req.command}")

            if req.command == 'set_volume':
                level = req.parameters.get('level', 50)
                result = hardware.set_volume(level)
            elif req.command == 'get_volume':
                result = hardware.get_volume()
            else:
                return jsonify(HardwareControlResponse(
                    status=Status.ERROR, result="", error=f"Unknown command: {req.command}"
                ).model_dump()), 400

            return jsonify(HardwareControlResponse(status=Status.SUCCESS, result=result).model_dump())

        except Exception as e:
            logger.error(f"Error in hardware_control: {e}", exc_info=True)
            return jsonify(HardwareControlResponse(
                status=Status.ERROR, result="", error="Internal server error"
            ).model_dump()), 500

    @app.route('/api/play_beep', methods=['POST'])
    def play_beep():
        """Play a local beep through the Pi's speaker.

        Called by PiCallbackClient.play_beep() for timer alert beeps and by
        the audio module for start/end/error signals when USE_SONOS_OUTPUT=False.
        BeepType enum validation ensures only known types are accepted.

        Request body: PlayBeepRequest (beep_type)
        Response: PlayBeepResponse (status, error)
        """
        try:
            data = request.get_json()
            if data is None:
                return _bad_json()
            req = PlayBeepRequest(**data)

            # Map BeepType enum to the corresponding audio method
            beep_map = {
                BeepType.ALERT: audio_system.beep_alert,
                BeepType.ERROR: audio_system.beep_error,
                BeepType.DONE:  audio_system.beep_done,
                BeepType.START: audio_system.beep_start,
                BeepType.END:   audio_system.beep_end,
            }
            fn = beep_map.get(req.beep_type)
            if fn:
                fn()
                return jsonify(PlayBeepResponse(status=Status.SUCCESS).model_dump())
            else:
                return jsonify(PlayBeepResponse(
                    status=Status.ERROR, error=f"Unknown beep type: {req.beep_type}"
                ).model_dump()), 400

        except Exception as e:
            logger.error(f"Error in play_beep: {e}", exc_info=True)
            return jsonify(PlayBeepResponse(
                status=Status.ERROR, error="Internal server error"
            ).model_dump()), 500

    @app.route('/api/suppress_wakeword', methods=['POST'])
    def suppress_wakeword():
        """Set wake word suppression for N seconds.

        Called by the server after TV commands so startup audio from Netflix,
        YouTube, etc. doesn't immediately retrigger the wake word detector.
        Writes to client/suppress.py's module-level state which is checked
        by the main detection loop.

        Request body: {"seconds": float}  (default 20.0)
        Response: {"status": "ok", "seconds": float}
        """
        try:
            data = request.get_json() or {}
            seconds = min(float(data.get('seconds', 20.0)), 300.0)  # cap at 5 min
            from client.suppress import suppress as _suppress
            _suppress(seconds)
            logger.info(f"Wake word suppressed for {seconds:.0f}s")
            return jsonify({'status': 'ok', 'seconds': seconds})
        except Exception as e:
            logger.error(f"Error in suppress_wakeword: {e}")
            return jsonify({'error': 'Internal server error'}), 500

    @app.route('/api/relabel_wakeword', methods=['POST'])
    def relabel_wakeword():
        """Move the most recent auto-saved wake word sample to negative/.

        Called when the user reports a false positive ("that wasn't me").
        Finds the newest auto_*.wav in positive/, moves it to negative/.
        """
        try:
            pos_dir = WAKE_SAMPLES_DIR  # .../wakeword_samples/positive/
            neg_dir = pos_dir.parent / "negative"

            # Find the most recent auto-saved sample
            auto_files = sorted(pos_dir.glob("auto_*.wav"), key=lambda f: f.stat().st_mtime)
            if not auto_files:
                return jsonify({'status': 'error', 'error': 'No auto-saved samples to relabel.'}), 404

            latest = auto_files[-1]
            neg_dir.mkdir(parents=True, exist_ok=True)

            # Find next available index in negative/ (max + 1, not len,
            # to avoid overwriting if files were deleted non-sequentially)
            existing_neg = list(neg_dir.glob("auto_*.wav"))
            if existing_neg:
                max_idx = max(
                    int(f.stem.split("_")[1]) for f in existing_neg
                    if f.stem.split("_")[1].isdigit()
                )
                next_idx = max_idx + 1
            else:
                next_idx = 0
            dest = neg_dir / f"auto_{next_idx:04d}.wav"

            latest.rename(dest)
            logger.info(f"Relabeled wake word sample: {latest.name} → negative/{dest.name}")
            return jsonify({'status': 'success', 'message': f'Moved {latest.name} to negative samples.'})

        except Exception as e:
            logger.error(f"Error relabeling wake word sample: {e}", exc_info=True)
            return jsonify({'status': 'error', 'error': 'Failed to relabel sample.'}), 500

    @app.route('/api/health', methods=['GET'])
    def health():
        """Liveness check — open to all IPs for monitoring.

        Reports audio subsystem status and uptime.  The 'pa' attribute on the
        audio system is the PyAudio instance — None if not yet initialized.
        """
        try:
            return jsonify(HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services={'audio': 'ready' if audio_system.pa else 'not_initialized', 'hardware': 'ready'},
                uptime_seconds=time.time() - start_time,
                additional_info={}
            ).model_dump())
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify(HealthCheckResponse(
                status=HealthStatus.UNHEALTHY, services={}, uptime_seconds=0,
                additional_info={'error': 'Health check failed'}
            ).model_dump()), 500

    @app.route('/', methods=['GET'])
    def root():
        return jsonify({'service': 'Igor Pi Client', 'status': 'running'})

    return app


def run_pi_server(audio_system, host: str = None, port: int = None):
    """Start the Pi callback server (blocking — run in a daemon thread).

    Uses Waitress instead of Flask's dev server for production stability.
    4 threads handles concurrent requests (e.g. a timer alert arriving
    while a beep is already playing).

    Args:
        audio_system: Audio instance to use for playback.
        host: Interface to bind on (default from CLIENT_HOST config).
        port: Port to listen on (default from CLIENT_PORT config).
    """
    from client.config import CLIENT_HOST, CLIENT_PORT
    host = host or CLIENT_HOST
    port = port or CLIENT_PORT
    start_time = time.time()
    app = create_pi_app(audio_system, start_time)
    logger.info(f"Starting Pi server on {host}:{port}")
    from waitress import serve
    serve(app, host=host, port=port, threads=4)
