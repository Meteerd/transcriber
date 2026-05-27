#!/usr/bin/env bash
# Transcribe one or more files via CLI (no server).
# Usage: ./cli.sh path/to/audio.m4a [more.m4a ...]
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
exec python cli.py "$@"
