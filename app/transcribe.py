"""Core transcription + diarization pipeline.

Flow:
  any audio/video → ffmpeg (16kHz mono WAV + highpass + loudness normalization)
                  → pyannote diarization (speaker turns)
                  → MLX Whisper or faster-whisper word-level transcription
                  → per-word speaker assignment with smoothing across short interruptions
                  → split segments where speaker actually changes mid-sentence,
                    preserving punctuation/capitalization from segment.text
                  → merge consecutive same-speaker blocks → markdown
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

WHISPER_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"
WHISPER_MODEL_FULL = "mlx-community/whisper-large-v3-mlx"
FASTER_WHISPER_MODEL_TURBO = "large-v3-turbo"
FASTER_WHISPER_MODEL_FULL = "large-v3"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", WHISPER_MODEL_TURBO)
WHISPER_ENGINE = os.environ.get("WHISPER_ENGINE", "mlx").lower()
HF_TOKEN = os.environ.get("HF_TOKEN")
DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
)
DIARIZATION_DEVICE = os.environ.get("DIARIZATION_DEVICE", "auto").lower()
FASTER_WHISPER_DEVICE = os.environ.get("FASTER_WHISPER_DEVICE", "auto").lower()
FASTER_WHISPER_COMPUTE_TYPE = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "float16")
FASTER_WHISPER_BEAM_SIZE = int(os.environ.get("FASTER_WHISPER_BEAM_SIZE", "5"))
FASTER_WHISPER_VAD = os.environ.get("FASTER_WHISPER_VAD", "1").lower() not in {"0", "false", "no"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _env_float_or_none(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in {"none", "off", "false", "0"}:
        return None
    return float(raw)


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = tuple(float(v.strip()) for v in raw.split(",") if v.strip())
    return values or default


MLX_CONDITION_ON_PREVIOUS_TEXT = _env_bool("MLX_CONDITION_ON_PREVIOUS_TEXT", False)
MLX_HALLUCINATION_SILENCE_THRESHOLD = _env_float_or_none(
    "MLX_HALLUCINATION_SILENCE_THRESHOLD",
    2.0,
)
MLX_TEMPERATURES = _env_float_tuple("MLX_TEMPERATURES", (0.0, 0.2, 0.4, 0.6))

# Per-language initial prompts to bias Whisper toward correct vocab + character
# set. Especially helps avoid mid-call language flip (greetings in English →
# rest of call wrongly transcribed in English). Keep short — long prompts can
# make Whisper paraphrase.
LANGUAGE_PROMPTS = {
    "tr": (
        "Turkish business conversation that may include English startup, VC, "
        "AI, and technology terms. Preserve both Turkish and English words."
    ),
    "hu": (
        "Hungarian business conversation that may include English startup, VC, "
        "AI, and technology terms. Preserve both Hungarian and English words."
    ),
    "en": None,  # English is the default training language — no prompt needed
}

MIXED_LANGUAGE_PROMPTS = {
    "tr-en": (
        "Mixed Turkish and English business conversation. Preserve code-switching, "
        "proper names, startup terms, VC terms, and technology vocabulary exactly."
    ),
    "hu-en": (
        "Mixed Hungarian and English business conversation. Preserve code-switching, "
        "proper names, startup terms, VC terms, and technology vocabulary exactly."
    ),
}

# Smoothing: a one-off short interjection from a different speaker is treated as
# noise — we keep the surrounding speaker. Tunable via env.
MIN_RUN_WORDS = int(os.environ.get("MIN_RUN_WORDS", "2"))
MIN_RUN_SECONDS = float(os.environ.get("MIN_RUN_SECONDS", "0.8"))

TranscriptionEngine = Literal["mlx", "faster-whisper"]
ProgressCb = Callable[[str, float | None], None]  # (stage, fraction 0..1 or None)


# --------------------------------------------------------------------------- #
# Pyannote pipeline (loaded lazily, kept in memory across requests)
# --------------------------------------------------------------------------- #
_diarizer = None
_faster_whisper_models: dict[tuple[str, str, str], object] = {}


def _get_diarizer():
    global _diarizer
    if _diarizer is None:
        if not HF_TOKEN:
            raise RuntimeError(
                f"HF_TOKEN not set. Add it to .env — required to download "
                f"{DIARIZATION_MODEL} from HuggingFace."
            )
        import torch
        import torchaudio

        # pyannote.audio 3.x still references this as an eager type annotation,
        # while newer torchaudio CUDA builds no longer export it at top-level.
        if not hasattr(torchaudio, "AudioMetaData"):
            torchaudio.AudioMetaData = object
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=HF_TOKEN,
        )
        device = _resolve_torch_device(torch, DIARIZATION_DEVICE)
        if device != "cpu":
            pipeline.to(torch.device(device))
        _diarizer = pipeline
    return _diarizer


def _resolve_torch_device(torch_module, requested: str) -> str:
    """Resolve a torch device name from an env/user setting."""
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested in {"cuda", "mps", "cpu"}:
        return requested
    raise ValueError("device must be one of: auto, cuda, mps, cpu")


# --------------------------------------------------------------------------- #
# Audio prep
# --------------------------------------------------------------------------- #
def _ffmpeg_binary() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """Decode to 16 kHz mono PCM WAV with phone-recording-friendly cleanup:

    - highpass 80 Hz: kill HVAC/handling rumble that confuses VAD
    - dynaudnorm: gentle dynamic range compression so the quiet speaker doesn't disappear
    - loudnorm (EBU R128): even out volume across speakers / recording distances

    Note: we don't run a learned denoiser (DeepFilterNet etc.) — those have been
    shown to *degrade* Whisper WER by stripping frequencies the model relies on.
    """
    af = "highpass=f=80,dynaudnorm=f=200:g=15,loudnorm=I=-16:TP=-1.5:LRA=11"
    cmd = [
        _ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-af", af,
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def diarize(
    wav_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    pipeline = _get_diarizer()
    kwargs = {
        k: v
        for k, v in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if v is not None
    }
    annotation = pipeline(str(wav_path), **kwargs)
    turns: list[Turn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(Turn(start=float(segment.start), end=float(segment.end), speaker=str(speaker)))
    turns.sort(key=lambda t: t.start)
    return turns


def _speaker_at(t: float, turns: list[Turn]) -> str:
    """Return the speaker whose diarization turn contains t. Nearest fallback."""
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    if not turns:
        return "SPEAKER_??"
    return min(turns, key=lambda x: min(abs(x.start - t), abs(x.end - t))).speaker


# --------------------------------------------------------------------------- #
# Transcription — request word-level timestamps for accurate speaker assignment
# --------------------------------------------------------------------------- #
def transcribe_segments(
    wav_path: Path,
    *,
    engine: TranscriptionEngine | None = None,
    model: str | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    progress: ProgressCb | None = None,
    progress_start: float = 0.55,
    progress_end: float = 0.90,
) -> dict:
    """Transcribe with word_timestamps. Returns Whisper-shaped segment dicts.

    Args:
        engine:         "mlx" for Apple Silicon, "faster-whisper" for CUDA/CPU.
        model:          HF repo id; defaults to env WHISPER_MODEL.
        language:       ISO 639-1 code ("tr", "hu", "en"); None = auto-detect.
        initial_prompt: free-text bias prompt; auto-filled from LANGUAGE_PROMPTS
                        if language is given and no explicit prompt provided.
    """
    if initial_prompt is None and language:
        initial_prompt = LANGUAGE_PROMPTS.get(language)

    selected_engine = _normalize_engine(engine or WHISPER_ENGINE)
    if selected_engine == "faster-whisper":
        return _transcribe_faster_whisper(
            wav_path,
            model=model,
            language=language,
            initial_prompt=initial_prompt,
            progress=progress,
            progress_start=progress_start,
            progress_end=progress_end,
        )

    return _transcribe_mlx_whisper(
        wav_path,
        model=model,
        language=language,
        initial_prompt=initial_prompt,
    )


def _normalize_engine(engine: str) -> TranscriptionEngine:
    engine = engine.lower().strip()
    if engine in {"mlx", "mlx-whisper", "mlx_whisper"}:
        return "mlx"
    if engine in {"faster", "faster-whisper", "faster_whisper", "cuda"}:
        return "faster-whisper"
    raise ValueError("engine must be one of: mlx, faster-whisper")


def _transcribe_mlx_whisper(
    wav_path: Path,
    *,
    model: str | None,
    language: str | None,
    initial_prompt: str | None,
) -> dict:
    """Transcribe with mlx-whisper on Apple Silicon."""

    kwargs: dict = {
        "path_or_hf_repo": model or WHISPER_MODEL,
        "word_timestamps": True,
        "condition_on_previous_text": MLX_CONDITION_ON_PREVIOUS_TEXT,
        "temperature": MLX_TEMPERATURES,
        "verbose": False,
    }
    if MLX_HALLUCINATION_SILENCE_THRESHOLD is not None:
        kwargs["hallucination_silence_threshold"] = MLX_HALLUCINATION_SILENCE_THRESHOLD
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    import mlx_whisper

    return mlx_whisper.transcribe(str(wav_path), **kwargs)


def _resolve_faster_whisper_device() -> str:
    if FASTER_WHISPER_DEVICE != "auto":
        return FASTER_WHISPER_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_faster_whisper_model(model: str | None):
    from faster_whisper import WhisperModel

    model_name = model or FASTER_WHISPER_MODEL_TURBO
    device = _resolve_faster_whisper_device()
    compute_type = FASTER_WHISPER_COMPUTE_TYPE
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    key = (model_name, device, compute_type)
    if key not in _faster_whisper_models:
        _faster_whisper_models[key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return _faster_whisper_models[key]


def _transcribe_faster_whisper(
    wav_path: Path,
    *,
    model: str | None,
    language: str | None,
    initial_prompt: str | None,
    progress: ProgressCb | None,
    progress_start: float,
    progress_end: float,
) -> dict:
    """Transcribe with faster-whisper/CTranslate2 on CUDA or CPU."""
    fw_model = _get_faster_whisper_model(model)
    model_label = (model or FASTER_WHISPER_MODEL_TURBO).rsplit("/", 1)[-1]
    device_label = _resolve_faster_whisper_device().upper()
    kwargs: dict = {
        "beam_size": FASTER_WHISPER_BEAM_SIZE,
        "word_timestamps": True,
        "vad_filter": FASTER_WHISPER_VAD,
        "condition_on_previous_text": False,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    segments_iter, info = fw_model.transcribe(str(wav_path), **kwargs)
    duration = float(getattr(info, "duration_after_vad", 0.0) or getattr(info, "duration", 0.0) or 0.0)
    progress_span = max(0.0, progress_end - progress_start)
    segments = []
    for idx, seg in enumerate(segments_iter):
        words = [
            {
                "word": w.word,
                "start": float(w.start),
                "end": float(w.end),
                "probability": float(getattr(w, "probability", 0.0)),
            }
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        segments.append({
            "id": idx,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": words,
        })
        if progress and duration > 0:
            done_ratio = min(max(float(seg.end) / duration, 0.0), 1.0)
            progress(
                f"Transcribing with {device_label} / {model_label} ({_fmt_ts(seg.end)} / {_fmt_ts(duration)})…",
                min(progress_start + done_ratio * progress_span, progress_end),
            )

    return {
        "text": " ".join(s["text"].strip() for s in segments if s["text"].strip()),
        "segments": segments,
        "language": getattr(info, "language", language),
    }


class TranscriptionQualityError(RuntimeError):
    """Raised when Whisper returns an obvious hallucination instead of speech."""


_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÁÉÍÓÖŐÚÜŰáéíóöőúüű]+")


def _words_for_quality_check(whisper_result: dict) -> list[str]:
    text = whisper_result.get("text") or " ".join(
        str(seg.get("text", "")) for seg in whisper_result.get("segments", [])
    )
    return [w.casefold() for w in _WORD_RE.findall(text)]


def _detect_repetition_hallucination(whisper_result: dict) -> str | None:
    """Return a human-readable reason if the transcript is clearly bogus."""
    words = _words_for_quality_check(whisper_result)
    if len(words) < 120:
        return None

    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    top_ratio = top_count / len(words)
    unique_ratio = len(counts) / len(words)

    segment_texts = [
        str(seg.get("text", "")).strip().casefold()
        for seg in whisper_result.get("segments", [])
        if str(seg.get("text", "")).strip()
    ]
    repeated_greeting_segments = sum(
        1
        for text in segment_texts
        if text.count("merhaba") >= 3 or text.count("hello") >= 3 or text.count("hi") >= 3
    )

    if top_ratio >= 0.35 and unique_ratio <= 0.12:
        return (
            f"Whisper repetition loop detected: '{top_word}' is "
            f"{top_ratio:.0%} of all words and vocabulary diversity is "
            f"{unique_ratio:.0%}."
        )
    if repeated_greeting_segments >= 5 and top_word in {"merhaba", "hello", "hi"}:
        return f"Whisper greeting loop detected: repeated '{top_word}' across many segments."
    return None


def assert_transcription_quality(whisper_result: dict) -> None:
    reason = _detect_repetition_hallucination(whisper_result)
    if reason:
        raise TranscriptionQualityError(
            reason
            + " The transcript was rejected instead of being saved. "
            + "Use the mixed-language option or faster-whisper/CUDA for this file."
        )


# --------------------------------------------------------------------------- #
# Word-level speaker assignment with smoothing + segment text re-splicing
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class _WordRun:
    """A consecutive run of words assigned to the same speaker within a segment."""
    speaker: str
    start_idx: int  # inclusive index into text_words
    end_idx: int    # exclusive
    start: float
    end: float


_WHITESPACE_SPLIT = re.compile(r"\s+")


def _split_text_preserving_punctuation(text: str) -> list[str]:
    """Split segment.text by whitespace, keeping punctuation glued to tokens.

    "Hey, thanks for joining." → ["Hey,", "thanks", "for", "joining."]
    """
    return [t for t in _WHITESPACE_SPLIT.split(text.strip()) if t]


def _smooth_runs(runs: list[_WordRun]) -> list[_WordRun]:
    """Merge a short interjection back into surrounding speaker.

    Pattern: [A, B(tiny), A] → [A] (B is absorbed into A).
    A run is "tiny" if it has fewer than MIN_RUN_WORDS words AND lasts less
    than MIN_RUN_SECONDS. Both ends of the sandwich must be the same speaker.
    """
    if len(runs) < 3:
        return runs

    out: list[_WordRun] = []
    i = 0
    while i < len(runs):
        # Look ahead for the sandwich pattern
        if (
            i + 2 < len(runs)
            and runs[i].speaker == runs[i + 2].speaker
            and runs[i + 1].speaker != runs[i].speaker
            and (runs[i + 1].end_idx - runs[i + 1].start_idx) < MIN_RUN_WORDS
            and (runs[i + 1].end - runs[i + 1].start) < MIN_RUN_SECONDS
        ):
            # Absorb middle into the outer speaker. Combine all three.
            merged = _WordRun(
                speaker=runs[i].speaker,
                start_idx=runs[i].start_idx,
                end_idx=runs[i + 2].end_idx,
                start=runs[i].start,
                end=runs[i + 2].end,
            )
            out.append(merged)
            i += 3
        else:
            out.append(runs[i])
            i += 1
    return out


def _segment_to_blocks(seg: dict, turns: list[Turn]) -> list[Block]:
    """Convert one whisper segment into 1+ blocks, splitting where the speaker
    actually changes within the segment (preserving punctuated text)."""
    text = seg["text"].strip()
    if not text:
        return []

    text_words = _split_text_preserving_punctuation(text)
    whisper_words = seg.get("words") or []

    # If we don't have word timings, fall back to dominant-speaker for the segment.
    if not whisper_words:
        spk = _dominant_speaker(float(seg["start"]), float(seg["end"]), turns)
        return [Block(spk, float(seg["start"]), float(seg["end"]), text)]

    # Best case: word count matches the punctuated split → 1:1 alignment.
    if len(whisper_words) == len(text_words):
        word_speakers = [
            _speaker_at((float(w["start"]) + float(w["end"])) / 2, turns)
            for w in whisper_words
        ]

        # Build runs of consecutive same-speaker words.
        runs: list[_WordRun] = []
        i = 0
        while i < len(word_speakers):
            j = i
            while j < len(word_speakers) and word_speakers[j] == word_speakers[i]:
                j += 1
            runs.append(_WordRun(
                speaker=word_speakers[i],
                start_idx=i,
                end_idx=j,
                start=float(whisper_words[i]["start"]),
                end=float(whisper_words[j - 1]["end"]),
            ))
            i = j

        runs = _smooth_runs(runs)

        return [
            Block(
                speaker=r.speaker,
                start=r.start,
                end=r.end,
                text=" ".join(text_words[r.start_idx:r.end_idx]),
            )
            for r in runs
        ]

    # Length mismatch (rare — whisper tokenized differently than whitespace split):
    # fall back to dominant-speaker for the whole segment with original text.
    spk = _dominant_speaker(float(seg["start"]), float(seg["end"]), turns)
    return [Block(spk, float(seg["start"]), float(seg["end"]), text)]


def _dominant_speaker(seg_start: float, seg_end: float, turns: list[Turn]) -> str:
    """Pick the speaker whose turns overlap [seg_start, seg_end] the most."""
    overlap: dict[str, float] = defaultdict(float)
    for t in turns:
        if t.end < seg_start or t.start > seg_end:
            continue
        ov = min(t.end, seg_end) - max(t.start, seg_start)
        if ov > 0:
            overlap[t.speaker] += ov

    if overlap:
        return max(overlap, key=overlap.get)
    if not turns:
        return "SPEAKER_??"
    seg_mid = (seg_start + seg_end) / 2
    return min(turns, key=lambda t: min(abs(t.start - seg_mid), abs(t.end - seg_mid))).speaker


def merge_segments_with_speakers(whisper_result: dict, turns: list[Turn]) -> list[Block]:
    """Walk whisper segments, split at speaker boundaries, then merge consecutive
    same-speaker blocks across segments."""
    raw: list[Block] = []
    for seg in whisper_result.get("segments", []):
        raw.extend(_segment_to_blocks(seg, turns))

    if not raw:
        return raw

    # Merge consecutive same-speaker blocks
    merged: list[Block] = [raw[0]]
    for b in raw[1:]:
        if b.speaker == merged[-1].speaker:
            merged[-1] = Block(
                speaker=merged[-1].speaker,
                start=merged[-1].start,
                end=b.end,
                text=(merged[-1].text + " " + b.text).strip(),
            )
        else:
            merged.append(b)
    return merged


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def to_markdown(
    blocks: list[Block],
    *,
    source_name: str,
    language: str | None,
    duration_seconds: float,
) -> str:
    lines = [
        f"# {source_name}",
        "",
        f"- **Transcribed:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Duration:** {_fmt_ts(duration_seconds)}",
        f"- **Language:** {language or 'auto'}",
        f"- **Speakers detected:** {len({b.speaker for b in blocks})}",
        "",
        "---",
        "",
    ]
    for b in blocks:
        lines.append(f"**{b.speaker}** [{_fmt_ts(b.start)} – {_fmt_ts(b.end)}]")
        lines.append("")
        lines.append(b.text)
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptionResult:
    markdown: str
    language: str | None
    duration_seconds: float
    num_speakers: int
    num_blocks: int


def transcribe_file(
    src: Path,
    *,
    output_dir: Path,
    source_name: str | None = None,
    progress: ProgressCb | None = None,
    engine: TranscriptionEngine | None = None,
    language: str | None = None,
    model: str | None = None,
    initial_prompt: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[Path, TranscriptionResult]:
    """Full pipeline. Returns (markdown_path, result).

    Args:
        language: ISO 639-1 hint (e.g. "tr" for Turkish). None = Whisper auto-detect.
        engine:   "mlx" for Apple Silicon, "faster-whisper" for CUDA/CPU.
        model:    model id/name; None = use selected engine default.
        num_speakers: exact speaker count hint for pyannote when known.
    """
    def _p(stage: str, frac: float | None = None) -> None:
        if progress:
            progress(stage, frac)

    src = Path(src)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_engine = _normalize_engine(engine or WHISPER_ENGINE)
    default_model = WHISPER_MODEL if selected_engine == "mlx" else FASTER_WHISPER_MODEL_TURBO
    model_label = (model or default_model).rsplit("/", 1)[-1]

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        _p("Preparing audio (highpass + normalize)…", 0.05)
        to_wav_16k_mono(src, wav)

        _p("Running speaker diarization…", 0.15)
        turns = diarize(
            wav,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        _p(f"Loading model and transcribing with {selected_engine} / {model_label}…", 0.55)
        whisper_result = transcribe_segments(
            wav,
            engine=selected_engine,
            model=model,
            language=language,
            initial_prompt=initial_prompt,
            progress=progress,
        )
        assert_transcription_quality(whisper_result)

        _p("Assigning speakers per word…", 0.92)
        blocks = merge_segments_with_speakers(whisper_result, turns)

        duration = max((b.end for b in blocks), default=0.0)
        if whisper_result.get("segments"):
            duration = max(duration, float(whisper_result["segments"][-1]["end"]))

        display_name = source_name or src.stem
        md = to_markdown(
            blocks,
            source_name=display_name,
            language=whisper_result.get("language"),
            duration_seconds=duration,
        )

        out_path = output_dir / f"{display_name}.md"
        n = 1
        while out_path.exists():
            out_path = output_dir / f"{display_name}-{n}.md"
            n += 1
        out_path.write_text(md, encoding="utf-8")

        _p("Done", 1.0)

        result = TranscriptionResult(
            markdown=md,
            language=whisper_result.get("language"),
            duration_seconds=duration,
            num_speakers=len({b.speaker for b in blocks}),
            num_blocks=len(blocks),
        )
        return out_path, result
