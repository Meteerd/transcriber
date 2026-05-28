#!/usr/bin/env bash
# Start the transcriber server.
# This is the ONLY way the server runs. Nothing auto-starts on boot or login.
# Stop it any time with: ./stop.sh  (or Ctrl-C if running in foreground)
set -euo pipefail
cd "$(dirname "$0")"

HOST="${TRANSCRIBER_HOST:-127.0.0.1}"
PORT="${TRANSCRIBER_PORT:-8765}"
SERVER_PATTERN="[u]vicorn app.main:app"

# Refuse to run if already up
if pgrep -f "$SERVER_PATTERN" > /dev/null; then
  echo "Transcriber is already running."
  echo "Stop it first with: ./stop.sh"
  exit 1
fi

source .venv/bin/activate
echo "Starting transcriber at http://${HOST}:${PORT}"
echo "Press Ctrl-C to stop. (Or run ./stop.sh in another terminal.)"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level info
