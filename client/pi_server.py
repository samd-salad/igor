"""HTTP server on Pi for receiving callbacks from PC."""
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
from client.config import SERVER_HOST

logger = logging.getLogger(__name__)


def create_pi_app(audio_system, start_time: float) -> Flask:
    app = Flask(__name__)
    hardware = HardwareController()

    @app.before_request
    def restrict_to_server():
        """Only allow requests from the known PC server."""
        if request.endpoint == 'health':
            return  # Health check open for monitoring
        if request.remote_addr != SERVER_HOST:
            logger.warning(f"Rejected request from {request.remote_addr} (expected {SERVER_HOST})")
            return jsonify({'error': 'Unauthorized'}), 403

    def _bad_json():
        return jsonify({'error': 'Invalid JSON body'}), 400

    @app.route('/api/play_audio', methods=['POST'])
    def play_audio():
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
        try:
            data = request.get_json()
            if data is None:
                return _bad_json()
            req = PlayBeepRequest(**data)

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

    @app.route('/api/health', methods=['GET'])
    def health():
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
        return jsonify({'service': 'Dr. Butts Pi Client', 'status': 'running'})

    return app


def run_pi_server(audio_system, host: str = '0.0.0.0', port: int = 8080):
    start_time = time.time()
    app = create_pi_app(audio_system, start_time)
    logger.info(f"Starting Pi server on {host}:{port}")
    from waitress import serve
    serve(app, host=host, port=port, threads=4)
