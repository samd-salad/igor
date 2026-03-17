"""PC callback server — receives play_audio, play_beep, suppress_wakeword from the server.

Mirrors client/pi_server.py but uses PCAudio (PyAudio/Windows) instead of ALSA.
Runs on CLIENT_PORT (8081) to avoid conflict with the server on 8000.
"""
import base64
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify

from client.pc_audio import PCAudio
from client.pc_config import SERVER_HOST, WAKE_SAMPLES_DIR
from client.suppress import suppress, is_suppressed

logger = logging.getLogger(__name__)


def create_pc_app(audio: PCAudio) -> Flask:
    """Create the Flask callback server app."""
    app = Flask(__name__)

    def _restrict_to_server():
        """Reject requests not from the server IP."""
        remote = request.remote_addr
        if remote not in (SERVER_HOST, "127.0.0.1", "::1"):
            return jsonify({"error": "forbidden"}), 403
        return None

    @app.before_request
    def check_ip():
        return _restrict_to_server()

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "client": "pc"})

    @app.route("/api/play_audio", methods=["POST"])
    def play_audio():
        """Receive and play TTS audio from the server."""
        data = request.get_json()
        if not data or "audio_base64" not in data:
            return jsonify({"error": "missing audio_base64"}), 400

        wav_bytes = base64.b64decode(data["audio_base64"])
        threading.Thread(
            target=audio.play_wav_bytes, args=(wav_bytes,),
            daemon=True, name="PlayAudio",
        ).start()
        return jsonify({"status": "playing"})

    @app.route("/api/play_beep", methods=["POST"])
    def play_beep():
        """Play a beep sound."""
        data = request.get_json() or {}
        beep_type = data.get("type", "alert")

        beep_map = {
            "start": audio.beep_start,
            "end": audio.beep_end,
            "error": audio.beep_error,
            "alert": audio.beep_alert,
        }
        func = beep_map.get(beep_type, audio.beep_alert)
        threading.Thread(target=func, daemon=True, name="Beep").start()
        return jsonify({"status": "ok"})

    @app.route("/api/suppress_wakeword", methods=["POST"])
    def suppress():
        """Suppress wake word detection for a duration."""
        data = request.get_json() or {}
        seconds = data.get("seconds", 20)
        suppress(seconds)
        logger.info(f"Wake word suppressed for {seconds}s")
        return jsonify({"status": "suppressed", "seconds": seconds})

    @app.route("/api/hardware_control", methods=["POST"])
    def hardware_control():
        """Hardware control RPC — stub for PC (no ALSA)."""
        return jsonify({"error": "Hardware volume control not available on PC"}), 501

    @app.route("/api/relabel_wakeword", methods=["POST"])
    def relabel_wakeword():
        """Move the most recent auto-saved wake word sample to negative/."""
        neg_dir = WAKE_SAMPLES_DIR.parent / "negative"
        neg_dir.mkdir(parents=True, exist_ok=True)

        auto_files = sorted(
            WAKE_SAMPLES_DIR.glob("pc_auto_*.wav"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not auto_files:
            return jsonify({"status": "no samples to relabel"})

        latest = auto_files[0]
        dest = neg_dir / latest.name
        latest.rename(dest)
        logger.info(f"Relabeled wake word sample: {latest.name} → negative/")
        return jsonify({"status": "relabeled", "file": latest.name})

    return app


def start_callback_server(audio: PCAudio, host: str, port: int):
    """Start the callback server in a background thread."""
    app = create_pc_app(audio)

    def _run():
        from waitress import serve
        logger.info(f"PC callback server starting on {host}:{port}")
        serve(app, host=host, port=port, threads=4, _quiet=True)

    thread = threading.Thread(target=_run, daemon=True, name="PCCallbackServer")
    thread.start()
    return thread
