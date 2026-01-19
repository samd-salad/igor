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

logger = logging.getLogger(__name__)


def create_pi_app(audio_system, start_time: float) -> Flask:
    """
    Create Flask app for Pi HTTP server.

    Args:
        audio_system: Audio instance for playing sounds
        start_time: Server start timestamp

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    hardware = HardwareController()

    @app.route('/api/play_audio', methods=['POST'])
    def play_audio():
        """Play audio sent from PC (for timer alerts, etc.)."""
        try:
            data = request.get_json()
            req = PlayAudioRequest(**data)

            logger.info(f"Received audio playback request: '{req.message[:50]}...'")

            # Decode audio
            try:
                audio_bytes = decode_audio_base64(req.audio_base64)
            except Exception as e:
                logger.error(f"Failed to decode audio: {e}")
                return jsonify(PlayAudioResponse(
                    status=Status.ERROR,
                    played_at=time.time(),
                    error="Invalid audio encoding"
                ).model_dump()), 400

            # Play audio
            success = audio_system.play_audio_bytes(audio_bytes)

            if success:
                logger.info("Audio played successfully")
                return jsonify(PlayAudioResponse(
                    status=Status.SUCCESS,
                    played_at=time.time(),
                    error=None
                ).model_dump())
            else:
                logger.error("Failed to play audio")
                return jsonify(PlayAudioResponse(
                    status=Status.ERROR,
                    played_at=time.time(),
                    error="Audio playback failed"
                ).model_dump()), 500

        except Exception as e:
            logger.error(f"Error handling play_audio: {e}", exc_info=True)
            return jsonify(PlayAudioResponse(
                status=Status.ERROR,
                played_at=time.time(),
                error=str(e)
            ).model_dump()), 500

    @app.route('/api/hardware_control', methods=['POST'])
    def hardware_control():
        """Execute hardware command (volume, etc.)."""
        try:
            data = request.get_json()
            req = HardwareControlRequest(**data)

            logger.info(f"Received hardware command: {req.command}({req.parameters})")

            # Execute command
            if req.command == 'set_volume':
                level = req.parameters.get('level', 50)
                result = hardware.set_volume(level)
            elif req.command == 'get_volume':
                result = hardware.get_volume()
            else:
                logger.error(f"Unknown hardware command: {req.command}")
                return jsonify(HardwareControlResponse(
                    status=Status.ERROR,
                    result="",
                    error=f"Unknown command: {req.command}"
                ).model_dump()), 400

            logger.info(f"Hardware command result: {result}")
            return jsonify(HardwareControlResponse(
                status=Status.SUCCESS,
                result=result,
                error=None
            ).model_dump())

        except Exception as e:
            logger.error(f"Error handling hardware_control: {e}", exc_info=True)
            return jsonify(HardwareControlResponse(
                status=Status.ERROR,
                result="",
                error=str(e)
            ).model_dump()), 500

    @app.route('/api/play_beep', methods=['POST'])
    def play_beep():
        """Play beep sound."""
        try:
            data = request.get_json()
            req = PlayBeepRequest(**data)

            logger.debug(f"Received beep request: {req.beep_type}")

            # Play appropriate beep
            if req.beep_type == BeepType.ALERT:
                audio_system.beep_alert()
            elif req.beep_type == BeepType.ERROR:
                audio_system.beep_error()
            elif req.beep_type == BeepType.DONE:
                audio_system.beep_done()
            elif req.beep_type == BeepType.START:
                audio_system.beep_start()
            elif req.beep_type == BeepType.END:
                audio_system.beep_end()
            else:
                logger.error(f"Unknown beep type: {req.beep_type}")
                return jsonify(PlayBeepResponse(
                    status=Status.ERROR,
                    error=f"Unknown beep type: {req.beep_type}"
                ).model_dump()), 400

            return jsonify(PlayBeepResponse(
                status=Status.SUCCESS,
                error=None
            ).model_dump())

        except Exception as e:
            logger.error(f"Error handling play_beep: {e}", exc_info=True)
            return jsonify(PlayBeepResponse(
                status=Status.ERROR,
                error=str(e)
            ).model_dump()), 500

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        try:
            uptime = time.time() - start_time
            return jsonify(HealthCheckResponse(
                status=HealthStatus.HEALTHY,
                services={
                    'audio': 'ready' if audio_system.pa else 'not_initialized',
                    'hardware': 'ready'
                },
                uptime_seconds=uptime,
                additional_info={}
            ).model_dump())
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify(HealthCheckResponse(
                status=HealthStatus.UNHEALTHY,
                services={},
                uptime_seconds=0,
                additional_info={'error': str(e)}
            ).model_dump()), 500

    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint."""
        return jsonify({
            'service': 'Dr. Butts Voice Assistant - Pi Client',
            'status': 'running',
            'endpoints': {
                'health': '/api/health',
                'play_audio': '/api/play_audio',
                'hardware_control': '/api/hardware_control',
                'play_beep': '/api/play_beep'
            }
        })

    return app


def run_pi_server(audio_system, host: str = '0.0.0.0', port: int = 8080):
    """
    Run the Pi HTTP server.

    Args:
        audio_system: Audio instance
        host: Host to bind to
        port: Port to listen on
    """
    start_time = time.time()
    app = create_pi_app(audio_system, start_time)

    logger.info(f"Starting Pi server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
