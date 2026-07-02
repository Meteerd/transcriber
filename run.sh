#!/usr/bin/env bash
# Start the transcriber server (NeMo container: Sortformer v2.1 + per-language ASR).
# This is the ONLY way the server runs. Nothing auto-starts on boot or login.
# Stop it any time with: ./stop.sh
#
# Escape hatch: TRANSCRIBER_LEGACY=1 ./run.sh  runs the old venv Whisper+pyannote
# pipeline instead of the container (kept as a fallback).
set -euo pipefail
cd "$(dirname "$0")"

HOST="${TRANSCRIBER_HOST:-127.0.0.1}"
PORT="${TRANSCRIBER_PORT:-8765}"

# --- Legacy fallback: old venv pipeline -------------------------------------
if [ "${TRANSCRIBER_LEGACY:-0}" = "1" ]; then
  if pgrep -f "[u]vicorn app.main:app" > /dev/null; then
    echo "Transcriber (legacy) already running. Stop with ./stop.sh"; exit 1
  fi
  source .venv/bin/activate
  echo "Starting transcriber (LEGACY venv) at http://${HOST}:${PORT}"
  exec uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level info
fi

# --- Default: NeMo container -------------------------------------------------
IMAGE="transcriber-nemo:latest"
NAME="transcriber"
HOST_HF="$(cd .. && pwd)/huggingface"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Transcriber is already running (container '$NAME'). Stop it first with: ./stop.sh"
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE (first run only, a few minutes)…"
  docker build -f Dockerfile.nemo -t "$IMAGE" .
fi

echo "Starting transcriber at http://${HOST}:${PORT}  (NeMo: Sortformer + Parakeet/Canary/Qwen)"
echo "Press Ctrl-C to stop. (Or run ./stop.sh in another terminal.)"
exec docker run --rm --name "$NAME" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env \
  -e TRANSCRIBER_BACKEND=nemo \
  -e HF_HOME=/hfcache \
  -e FFMPEG_BINARY=ffmpeg \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOST_HF":/hfcache \
  -v "$(pwd)":/workspace/transcriber \
  -p "${HOST}:${PORT}:8765" \
  "$IMAGE"
