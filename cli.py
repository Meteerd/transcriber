"""One-shot CLI transcriber.

Usage:
    ./cli.sh [--engine mlx|faster-whisper] [--lang tr|tr-en|en|hu|...]
             [--quality turbo|full] [--speakers 2] FILE [FILE ...]

Examples:
    ./cli.sh interview.m4a
    ./cli.sh --lang tr-en --quality full dilan-call.m4a
    ./cli.sh --engine faster-whisper --quality full --speakers 2 meeting.m4a
    ./cli.sh --lang en *.m4a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.transcribe import (
    FASTER_WHISPER_MODEL_FULL,
    FASTER_WHISPER_MODEL_TURBO,
    MIXED_LANGUAGE_PROMPTS,
    WHISPER_MODEL_TURBO,
    WHISPER_MODEL_FULL,
    transcribe_file,
)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Local transcription with speaker labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="+", help="Audio/video files to transcribe")
    p.add_argument(
        "--engine", "-e", choices=("mlx", "faster-whisper"), default="mlx",
        help="mlx for Apple Silicon, faster-whisper for CUDA/CPU. Default: mlx.",
    )
    p.add_argument(
        "--lang", "-l", default=None,
        help="ISO 639-1 code (en, tr, hu, de, fr, …), or tr-en/hu-en for mixed calls. Omit for auto-detect.",
    )
    p.add_argument(
        "--quality", "-q", choices=("turbo", "full"), default="turbo",
        help="turbo (fast) or full (best for TR/HU). Default: turbo.",
    )
    p.add_argument(
        "--speakers", "-s", type=int, default=None,
        help="Exact speaker count hint for diarization, e.g. 2 for a two-person call.",
    )

    args = p.parse_args()

    if args.speakers is not None and args.speakers < 1:
        print("ERROR: --speakers must be a positive integer", file=sys.stderr)
        return 1

    if args.engine == "faster-whisper":
        model = FASTER_WHISPER_MODEL_FULL if args.quality == "full" else FASTER_WHISPER_MODEL_TURBO
    else:
        model = WHISPER_MODEL_FULL if args.quality == "full" else WHISPER_MODEL_TURBO
    language = args.lang
    initial_prompt = None
    if args.lang in MIXED_LANGUAGE_PROMPTS:
        language = None
        initial_prompt = MIXED_LANGUAGE_PROMPTS[args.lang]
    out_dir = Path(__file__).parent / "transcripts"
    out_dir.mkdir(exist_ok=True)

    paths = [Path(a).expanduser().resolve() for a in args.files]
    missing = [pp for pp in paths if not pp.exists()]
    if missing:
        for pp in missing:
            print(f"ERROR: not found: {pp}", file=sys.stderr)
        return 1

    print(f"Engine: {args.engine}")
    print(f"Model: {model.rsplit('/', 1)[-1]}")
    print(f"Language: {args.lang or 'auto-detect'}")
    print(f"Speakers: {args.speakers or 'auto'}")
    print()

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")
        print("-" * 60)

        def cb(stage: str, frac: float | None) -> None:
            pct = f"{round((frac or 0) * 100):>3}%"
            print(f"  [{pct}] {stage}")

        try:
            out_path, result = transcribe_file(
                path, output_dir=out_dir, progress=cb,
                engine=args.engine, language=language, initial_prompt=initial_prompt, model=model,
                num_speakers=args.speakers,
            )
            print(f"  ✓ Saved → {out_path}")
            print(f"    {result.num_speakers} speaker(s), "
                  f"{result.duration_seconds:.0f}s, {result.language or 'auto'}")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
