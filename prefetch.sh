#!/usr/bin/env bash
# Download Whisper model weights into the local Hugging Face cache.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
exec python scripts/prefetch_models.py "$@"
