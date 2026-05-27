# Research Notes: Local Meeting Transcription

Last reviewed: 2026-05-28

## Goal

High-quality local transcription for phone-recorded meetings, with speaker labels, multilingual support, and a path to benchmark on both Apple Silicon and NVIDIA GPUs.

## Recommended Baseline

Use Whisper large-v3 full plus pyannote diarization for important multilingual calls.

Why:

- Whisper remains the most reliable multilingual baseline across English, Turkish, Hungarian, and mixed-language calls.
- `large-v3-turbo` is fast and useful for drafts, but full `large-v3` is the quality-first choice.
- pyannote diarization is language-agnostic and supports known speaker-count hints.
- On Apple Silicon, MLX gives the simplest local acceleration path.
- On NVIDIA GPUs, faster-whisper/CTranslate2 gives a mature CUDA path and exposes word timestamps.
- RTX 5090 / Blackwell requires a PyTorch wheel new enough for `sm_120`; this repo uses the PyTorch cu128 index for the CUDA requirements.

## Projects Reviewed

### WhisperX

WhisperX combines faster-whisper, VAD, forced phoneme alignment, and pyannote diarization. It is the closest research-backed open-source pipeline to this app's goal, especially for long-form audio where timestamp drift affects speaker assignment.

Takeaway for this repo: keep the current lightweight pipeline, but consider adding WhisperX-style forced alignment as the next accuracy upgrade for CUDA machines.

Source: https://github.com/m-bain/whisperX

### faster-whisper

faster-whisper is the best practical CUDA runtime for Whisper in this app. It supports `large-v3`, `large-v3-turbo`, GPU FP16, CPU/int8 fallback, VAD, and word timestamps.

Takeaway for this repo: add `faster-whisper` as the NVIDIA backend.

Source: https://github.com/SYSTRAN/faster-whisper

### NVIDIA NeMo, Parakeet, Canary

NVIDIA's Parakeet and Canary models are strong ASR candidates for a 5090 benchmark. Parakeet is high-throughput; Canary-1B-v2 supports 25 European languages, punctuation, capitalization, and word/segment timestamps.

Takeaway for this repo: benchmark these on the 5090, especially for English. Keep Whisper as the default multilingual baseline until Turkish/Hungarian meeting quality is proven better on your data.

Sources:

- https://github.com/NVIDIA-NeMo/NeMo
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/canary-1b-v2

### pyannote.audio

`pyannote/speaker-diarization-3.1` is the conservative open-source diarization default. It accepts `num_speakers`, `min_speakers`, and `max_speakers`, which is important for meeting calls where you often know the speaker count.

Newer `speaker-diarization-community-1` is worth evaluating later, but it changes the dependency surface. This repo keeps 3.1 as the stable default for now.

Sources:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/speaker-diarization-community-1

### Vexa

Vexa is the reference project for live meeting bot infrastructure: Google Meet, Teams, Zoom, real-time transcripts, WebSockets, MCP, and self-hosting.

Takeaway for this repo: Vexa answers "how do I join live meetings?" This repo answers "how do I locally transcribe recordings from my phone at the highest quality?" They are complementary, not duplicates.

Source: https://github.com/Vexa-ai/vexa

## Benchmark Plan for RTX 5090

Use the same set of real calls and compare:

1. `faster-whisper large-v3`, FP16, language fixed, speaker count fixed.
2. `faster-whisper large-v3-turbo`, FP16, language fixed, speaker count fixed.
3. WhisperX `large-v3` with forced alignment and pyannote.
4. NVIDIA Parakeet-TDT-0.6B-v3 for English-heavy calls.
5. NVIDIA Canary-1B-v2 for multilingual European-language calls.

Score manually on what matters:

- Names, company names, and technical terms.
- Speaker-turn correctness.
- Turkish/Hungarian accuracy.
- Hallucinations during silence.
- Paragraph readability for AI context.
- End-to-end time.

## Current Product Direction

Keep the app boring and dependable:

- Local-only by default.
- No hidden cloud calls.
- No real recordings in git.
- Quality knobs visible in the UI.
- Exact speaker-count hint visible in the UI.
- CUDA path available without breaking the Mac path.
