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

import logging
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

log = logging.getLogger("transcriber")

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
# Embedding model used to stitch speaker identities across chunks (see
# _diarize_chunked). Same model pyannote 3.1 uses internally for embeddings.
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
)
# Long recordings are diarized in windows then stitched, because pyannote's
# global clustering collapses to a single speaker on long, monologue-heavy
# files (measured: a 37-min 2-person call diarized as 98%/2% in one pass but
# 28%/72% when chunked). Files at or below the chunk length use a single pass.
DIARIZATION_CHUNK_SECONDS = float(os.environ.get("DIARIZATION_CHUNK_SECONDS", "480"))
# Cosine-distance threshold for grouping per-chunk speakers into global speakers
# when the speaker count is unknown (auto). Lower = more speakers.
DIARIZATION_STITCH_THRESHOLD = float(os.environ.get("DIARIZATION_STITCH_THRESHOLD", "0.55"))
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
_embedder = None
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


def _get_embedder():
    """Speaker-embedding inference, used to stitch per-chunk speakers into global
    identities. Cached across requests. Returns a callable Inference(window=whole)."""
    global _embedder
    if _embedder is None:
        import torch
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            torchaudio.AudioMetaData = object
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
        inference = Inference(model, window="whole")
        device = _resolve_torch_device(torch, DIARIZATION_DEVICE)
        if device != "cpu":
            inference.to(torch.device(device))
        _embedder = inference
    return _embedder


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


# Heavy chain for ASR. Whisper transcribes better when the quiet speaker is
# leveled up, so we keep dynamic + loudness normalization here.
ASR_AUDIO_FILTER = "highpass=f=80,dynaudnorm=f=200:g=15,loudnorm=I=-16:TP=-1.5:LRA=11"

# Light chain for diarization: highpass ONLY. dynaudnorm/loudnorm flatten the
# per-speaker volume + spectral cues pyannote clusters on, and can collapse a
# multi-person call onto one speaker. Measured on a real stereo VC recording:
# the heavy chain gave a 95%/5% (one speaker swallows everything) split, while
# highpass-only gave a healthy 64%/36% two-speaker split. Both chains are
# time-preserving (no resampling/trim), so word timestamps from the ASR wav
# still line up with diarization turns from this one.
DIARIZATION_AUDIO_FILTER = "highpass=f=70"


def _ffmpeg_to_wav(src: Path, dst: Path, audio_filter: str) -> Path:
    """Decode any input to 16 kHz mono PCM WAV applying ``audio_filter``.

    Note: we don't run a learned denoiser (DeepFilterNet etc.) — those have been
    shown to *degrade* Whisper WER by stripping frequencies the model relies on.
    """
    cmd = [
        _ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-af", audio_filter,
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """ASR-oriented decode: 16 kHz mono + highpass + dynaudnorm + loudnorm."""
    return _ffmpeg_to_wav(src, dst, ASR_AUDIO_FILTER)


def to_wav_16k_mono_for_diarization(src: Path, dst: Path) -> Path:
    """Diarization-oriented decode: 16 kHz mono + highpass only (no level
    normalization), preserving the cues pyannote needs to separate speakers."""
    return _ffmpeg_to_wav(src, dst, DIARIZATION_AUDIO_FILTER)


# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def _clean_turns(turns: list[Turn]) -> list[Turn]:
    """Sort and drop zero/near-zero-duration artifacts (pyannote emits these at
    speaker boundaries; they cause tie-breaking issues in _speaker_at)."""
    turns.sort(key=lambda t: t.start)
    return [t for t in turns if (t.end - t.start) >= 0.05]


def _annotation_to_turns(annotation, offset: float = 0.0) -> list[Turn]:
    out: list[Turn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        out.append(Turn(
            start=float(segment.start) + offset,
            end=float(segment.end) + offset,
            speaker=str(speaker),
        ))
    return out


def diarize(
    wav_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Diarize a WAV. Long files are processed in windows and stitched, because
    pyannote's global clustering collapses to one speaker on long recordings."""
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = object
    info = torchaudio.info(str(wav_path))
    duration = info.num_frames / info.sample_rate

    if duration > DIARIZATION_CHUNK_SECONDS * 1.5:
        log.info("Long recording (%.0fs) — using chunked diarization", duration)
        return _diarize_chunked(
            wav_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    return _diarize_single(
        wav_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def _diarize_single(
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
    return _clean_turns(_annotation_to_turns(annotation))


def _diarize_chunked(
    wav_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Diarize in fixed windows, then stitch per-window speakers into global
    identities via voice embeddings.

    pyannote's one-pass global clustering collapses on long, imbalanced
    recordings. Diarizing each window independently keeps each clustering
    problem small (so it stays accurate), and matching speakers across windows
    by embedding similarity recovers consistent global speaker labels. Any
    over-split inside a window (e.g. a monologue window forced to 2 speakers) is
    healed when both local speakers map to the same global cluster.
    """
    import numpy as np
    import torch
    import torchaudio
    from pyannote.core import Segment
    from sklearn.cluster import AgglomerativeClustering

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = object

    pipeline = _get_diarizer()
    embedder = _get_embedder()
    wav, sr = torchaudio.load(str(wav_path))
    total = wav.shape[1] / sr
    chunk = DIARIZATION_CHUNK_SECONDS

    # Per-window hint: pass the user's speaker count through so dialogue windows
    # split reliably. Over-splitting in monologue windows is healed by stitching.
    win_kwargs = {
        k: v
        for k, v in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if v is not None
    }

    local_turns: list[tuple[float, float, tuple[int, str]]] = []
    emb_rows: list = []
    emb_keys: list[tuple[int, str]] = []

    ci = 0
    start = 0.0
    while start < total:
        seg_len = min(chunk, total - start)
        clip = wav[:, int(start * sr): int((start + seg_len) * sr)]
        clip_in = {"waveform": clip, "sample_rate": sr}
        annotation = pipeline(clip_in, **win_kwargs)

        by_spk: dict[str, list[tuple[float, float]]] = {}
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            local_turns.append((start + segment.start, start + segment.end, (ci, speaker)))
            by_spk.setdefault(speaker, []).append((segment.start, segment.end))

        # One embedding per local speaker = its longest segment in this window.
        for speaker, segs in by_spk.items():
            segs.sort(key=lambda x: x[1] - x[0], reverse=True)
            ls, le = segs[0]
            ls = max(0.0, ls)
            le = min(seg_len, le)
            if le - ls < 0.5:
                continue
            try:
                vec = embedder.crop(clip_in, Segment(ls, le))
                emb_rows.append(np.asarray(vec).reshape(-1))
                emb_keys.append((ci, speaker))
            except Exception:
                log.exception("embedding failed for window %d speaker %s", ci, speaker)

        ci += 1
        start += chunk

    if not emb_rows:
        return _clean_turns([Turn(s, e, "SPEAKER_00") for s, e, _ in local_turns])

    # Cluster per-window speakers into global identities by embedding similarity.
    X = np.vstack(emb_rows)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if len(emb_rows) == 1:
        labels = [0]
    elif num_speakers is not None:
        n = min(num_speakers, len(emb_rows))
        labels = AgglomerativeClustering(
            n_clusters=n, metric="cosine", linkage="average"
        ).fit_predict(X)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DIARIZATION_STITCH_THRESHOLD,
            metric="cosine",
            linkage="average",
        ).fit_predict(X)

    keymap = {emb_keys[i]: f"SPEAKER_{int(labels[i]):02d}" for i in range(len(emb_keys))}

    # Remap local turns → global speakers. A local speaker with no embedding
    # (too short) falls back to the dominant global speaker of its window.
    win_global_counts: dict[int, dict[str, float]] = {}
    for s, e, (cidx, spk) in local_turns:
        g = keymap.get((cidx, spk))
        if g is not None:
            win_global_counts.setdefault(cidx, {})[g] = (
                win_global_counts.setdefault(cidx, {}).get(g, 0.0) + (e - s)
            )

    global_turns: list[Turn] = []
    for s, e, (cidx, spk) in local_turns:
        g = keymap.get((cidx, spk))
        if g is None:
            counts = win_global_counts.get(cidx)
            g = max(counts, key=counts.get) if counts else "SPEAKER_00"
        global_turns.append(Turn(s, e, g))

    turns = _clean_turns(global_turns)
    dist: dict[str, float] = {}
    for t in turns:
        dist[t.speaker] = dist.get(t.speaker, 0.0) + (t.end - t.start)
    log.info("Chunked diarization (%d windows): %s",
             ci, {k: f"{v:.0f}s" for k, v in sorted(dist.items())})
    return turns


def _speaker_at(t: float, turns: list[Turn]) -> str:
    """Return the speaker whose diarization turn contains t.

    Fallback strategy for words that land in a gap between turns:
    find the turn whose boundary is nearest to t, preferring the START of
    the next turn over the END of the previous turn — this correctly handles
    the common case where Whisper places a word's timestamp slightly before
    pyannote's speaker-change boundary (both tools have ~200ms timing error).
    """
    # Pass 1: exact containment
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker

    if not turns:
        return "SPEAKER_??"

    # Pass 2: gap fallback — find the prev and next turns bracketing t, then
    # choose whichever boundary is closer, biased toward the START of the next
    # turn (not the END of the previous one).
    prev_turn: Turn | None = None
    next_turn: Turn | None = None
    for turn in turns:
        if turn.end <= t:
            if prev_turn is None or turn.end > prev_turn.end:
                prev_turn = turn
        elif turn.start >= t:
            if next_turn is None or turn.start < next_turn.start:
                next_turn = turn

    if prev_turn is None:
        return next_turn.speaker  # type: ignore[union-attr]
    if next_turn is None:
        return prev_turn.speaker

    dist_to_prev_end = t - prev_turn.end      # how long ago prev speaker finished
    dist_to_next_start = next_turn.start - t  # how soon next speaker starts

    # Prefer the next speaker when it's within 0.6 s and the gap is small
    # (covers the typical Whisper/pyannote timestamp offset).
    if next_turn.speaker != prev_turn.speaker and dist_to_next_start <= 0.6:
        return next_turn.speaker

    return prev_turn.speaker if dist_to_prev_end <= dist_to_next_start else next_turn.speaker


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
) -> dict:
    """Transcribe with faster-whisper/CTranslate2 on CUDA or CPU."""
    fw_model = _get_faster_whisper_model(model)
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

    # Length mismatch — whisper tokenized differently than whitespace split.
    # Fall back to dominant-speaker for the whole segment.
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
        wav_diar = Path(tmp) / "audio_diar.wav"
        _p("Preparing audio (highpass + normalize)…", 0.05)
        # Two streams from the same source, same timeline: a normalized one for
        # ASR, and a lightly-filtered one for diarization (heavy normalization
        # erases the per-speaker cues pyannote needs and can collapse a call
        # onto a single speaker).
        to_wav_16k_mono(src, wav)
        to_wav_16k_mono_for_diarization(src, wav_diar)

        _p("Running speaker diarization…", 0.15)
        turns = diarize(
            wav_diar,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        spk_time: dict[str, float] = {}
        for t in turns:
            spk_time[t.speaker] = spk_time.get(t.speaker, 0.0) + (t.end - t.start)
        log.info("Diarization: %d turn(s), distribution: %s",
                 len(turns), {k: f"{v:.0f}s" for k, v in sorted(spk_time.items())})

        _p(f"Loading model and transcribing with {selected_engine} / {model_label}…", 0.55)
        whisper_result = transcribe_segments(
            wav,
            engine=selected_engine,
            model=model,
            language=language,
            initial_prompt=initial_prompt,
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
