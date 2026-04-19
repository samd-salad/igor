# syntax=docker/dockerfile:1.7

# Slim, multi-arch (linux/amd64 + linux/arm64) image for Igor in HA
# conversation-agent mode. No Whisper, no Kokoro, no audio deps — HA's
# voice pipeline handles STT/TTS and posts transcribed text here.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000

WORKDIR /app

# System deps kept minimal. tini gives us a proper PID 1 for signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-server-text.txt ./
RUN pip install -r requirements-server-text.txt

COPY server ./server
COPY shared ./shared
COPY prompt.py ./

# Data directory is a mount point — brain.json, logs, rooms.yaml live here.
RUN mkdir -p /app/data

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "server.main_text"]
