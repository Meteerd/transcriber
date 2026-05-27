#!/usr/bin/env bash
# Start the transcriber server.
# This is the ONLY way the server runs. Nothing auto-starts on boot or login.
# Stop it any time with: ./stop.sh  (or Ctrl-C if running in foreground)
set -euo pipefail
cd "$(dirname "$0")"

# Refuse to run if already up
if pgrep -f "uvicorn app.main:app" > /dev/null; then
  echo "Transcriber is already running at http://127.0.0.1:8765"
  echo "Stop it first with: ./stop.sh"
  exit 1
fi

source .venv/bin/activate
echo "Starting transcriber at http://127.0.0.1:8765"
echo "Press Ctrl-C to stop. (Or run ./stop.sh in another terminal.)"
exec uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level info
