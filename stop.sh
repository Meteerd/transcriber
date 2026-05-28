#!/usr/bin/env bash
# Stop the transcriber server.
SERVER_PATTERN="[u]vicorn app.main:app"

if pgrep -f "$SERVER_PATTERN" > /dev/null; then
  pkill -f "$SERVER_PATTERN"
  sleep 1
  if pgrep -f "$SERVER_PATTERN" > /dev/null; then
    pkill -9 -f "$SERVER_PATTERN"
  fi
  echo "Transcriber stopped."
else
  echo "Transcriber wasn't running."
fi
