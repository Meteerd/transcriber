"""Download model weights before running a transcription job."""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "full": "mlx-community/whisper-large-v3-mlx",
}


def main() -> int:
    p = argparse.ArgumentParser(description="Prefetch Whisper models into the local HF cache.")
    p.add_argument(
        "--quality",
        choices=("turbo", "full", "all"),
        default="full",
        help="Which MLX Whisper model to prefetch. Default: full.",
    )
    args = p.parse_args()

    qualities = ("turbo", "full") if args.quality == "all" else (args.quality,)
    for quality in qualities:
        repo_id = MLX_MODELS[quality]
        print(f"Downloading {quality}: {repo_id}", flush=True)
        path = snapshot_download(repo_id)
        print(f"Cached at {path}", flush=True)

    print("Done. App restarts will reuse the cached model.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
