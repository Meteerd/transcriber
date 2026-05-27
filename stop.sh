#!/usr/bin/env bash
# Stop the transcriber server.
if pgrep -f "uvicorn app.main:app" > /dev/null; then
  pkill -f "uvicorn app.main:app"
  sleep 1
  if pgrep -f "uvicorn app.main:app" > /dev/null; then
    pkill -9 -f "uvicorn app.main:app"
  fi
  echo "Transcriber stopped."
else
  echo "Transcriber wasn't running."
fi
