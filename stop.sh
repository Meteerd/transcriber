#!/usr/bin/env bash
# Stop the transcriber server (container and/or legacy venv process).
NAME="transcriber"
stopped=0

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop "$NAME" >/dev/null 2>&1 && stopped=1
fi

SERVER_PATTERN="[u]vicorn app.main:app"
if pgrep -f "$SERVER_PATTERN" > /dev/null; then
  pkill -f "$SERVER_PATTERN"; sleep 1
  pgrep -f "$SERVER_PATTERN" > /dev/null && pkill -9 -f "$SERVER_PATTERN"
  stopped=1
fi

if [ "$stopped" = "1" ]; then
  echo "Transcriber stopped."
else
  echo "Transcriber wasn't running."
fi
