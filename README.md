# Transcriber

Local-first meeting transcription for phone recordings and exported audio/video files.

This app is intentionally small: drop an `.m4a`, `.mp3`, `.mp4`, `.wav`, etc. into a local web UI or run the CLI, then get a Markdown transcript with speaker labels that is easy to paste into AI agents.

## What it is good at

- Private batch transcription of iPhone Voice Memos / meeting recordings.
- Speaker diarization with `pyannote.audio`.
- Apple Silicon inference with `mlx-whisper`.
- NVIDIA/CUDA inference with `faster-whisper` for machines such as an RTX 5090.
- Language hints for multilingual calls, especially English, Turkish, and Hungarian.
- Exact speaker-count hints when you know the call had 1, 2, or 3 speakers.

## Recommended Quality Stack

For best multilingual meeting quality:

- **ASR:** Whisper large-v3 full for important calls.
- **Fast ASR:** Whisper large-v3-turbo for quick drafts.
- **Apple Silicon runtime:** `mlx-whisper`.
- **NVIDIA runtime:** `faster-whisper` / CTranslate2 with `float16`.
- **Diarization:** `pyannote/speaker-diarization-3.1`.
- **Best practical setting:** set the language explicitly and set the exact speaker count when known.

For English-only benchmarking on NVIDIA GPUs, also test NVIDIA Parakeet/Canary outside this app. They are promising, but Whisper large-v3 remains the safest baseline for multilingual meeting notes.

## Setup: Apple Silicon

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN`. You must accept the Hugging Face terms for:

- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

Start the local UI:

```bash
./run.sh
```

Open <http://127.0.0.1:8765>. The server only binds to localhost.

## Setup: NVIDIA / CUDA

On Linux with CUDA 12 + cuDNN 9:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cuda.txt
cp .env.example .env
```

Set:

```bash
WHISPER_ENGINE=faster-whisper
FASTER_WHISPER_DEVICE=cuda
FASTER_WHISPER_COMPUTE_TYPE=float16
```

CLI example for a two-person investor call:

```bash
./cli.sh --engine faster-whisper --quality full --lang tr --speakers 2 call.m4a
```

## Use

Web UI:

```bash
./run.sh
```

CLI:

```bash
./cli.sh --lang en --quality turbo meeting.m4a
./cli.sh --lang tr --quality full --speakers 2 investor-call.m4a
./cli.sh --engine faster-whisper --quality full --speakers 2 meeting.m4a
```

Outputs are saved under `./transcripts/`.

## Model Notes

- `turbo` maps to `mlx-community/whisper-large-v3-turbo` on MLX and `large-v3-turbo` on faster-whisper.
- `full` maps to `mlx-community/whisper-large-v3-mlx` on MLX and `large-v3` on faster-whisper.
- Turkish/Hungarian important calls should usually use `--quality full` plus `--lang tr` or `--lang hu`.
- Speaker labels improve when you pass `--speakers 2` for two-person calls.

## What This Is Not

This is not a Fireflies/Otter/Vexa-style meeting bot that joins Zoom, Google Meet, or Teams. It is the local, high-quality file transcription core. Projects like Vexa solve live meeting bot infrastructure; this project focuses on local files from your phone and high-quality transcripts for AI context.

## Files

- `app/main.py` - FastAPI server.
- `app/transcribe.py` - ffmpeg, diarization, ASR, speaker assignment, Markdown output.
- `static/index.html` - local drag-and-drop UI.
- `cli.py` / `cli.sh` - one-shot transcription without the web server.
- `requirements.txt` - Apple Silicon default.
- `requirements-cuda.txt` - NVIDIA/CUDA default.

## Public Repo Hygiene

The repo ignores:

- `.env`
- `.venv/`
- `uploads/`
- `transcripts/`
- caches and bytecode

Do not commit real recordings, transcripts, or Hugging Face tokens.
