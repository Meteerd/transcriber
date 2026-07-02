"""NeMo-based transcription pipeline — runs inside the nvcr.io/nvidia/nemo container.

Diarization: streaming Sortformer v2.1 (best "who-talked-when", handles overlap).
ASR: per-language router chosen from the UI Language dropdown —
  * English / Hungarian-mixed / other European / auto  -> Parakeet-TDT-0.6b-v3
        (noise-robust transducer, native word timestamps -> precise turn mapping)
  * Hungarian (pure)                                    -> Canary-1b-v2
  * Turkish / Tr-En / Tr-Hu (medical code-switching)    -> Qwen3-ASR-1.7B + glossary
        (LLM decoder keeps embedded English/Latin terms verbatim; `context=` biasing)

Drop-in for transcribe.transcribe_file, plus a `context` glossary argument.
Models are lazy-loaded singletons, kept resident (all fit in the 4090's 24 GB).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.transcribe import Block, TranscriptionResult, to_markdown

log = logging.getLogger("transcriber.nemo")
ProgressCb = Callable[[str, "float | None"], None]

SORTFORMER_MODEL = os.environ.get("SORTFORMER_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2.1")
PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
CANARY_MODEL = os.environ.get("CANARY_MODEL", "nvidia/canary-1b-v2")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")

# A persistent default glossary for code-switched medical speech; the UI can add
# to it per-upload. Passed as Qwen3-ASR `context=` to preserve exact term spelling.
DEFAULT_CONTEXT = os.environ.get("TRANSCRIBER_CONTEXT", "")

# Language dropdown value -> (strategy, model_key, force_lang)
#   strategy "wordts"  : whole-file transcribe with word timestamps (Parakeet)
#   strategy "chunked" : transcribe each diarized speaker turn (Canary / Qwen)
_ROUTER: dict[str, tuple[str, str, str | None]] = {
    "en":    ("wordts",  "parakeet", "en"),
    "hu":    ("chunked", "canary",   "hu"),
    "hu-en": ("wordts",  "parakeet", None),
    "tr":    ("chunked", "qwen",     "Turkish"),
    "tr-en": ("chunked", "qwen",     None),
    "tr-hu": ("chunked", "qwen",     None),
    "de":    ("wordts",  "parakeet", "de"),
    "fr":    ("wordts",  "parakeet", "fr"),
    "es":    ("wordts",  "parakeet", "es"),
    "it":    ("wordts",  "parakeet", "it"),
    "ru":    ("chunked", "qwen",     "Russian"),
    "ar":    ("chunked", "qwen",     "Arabic"),
    "zh":    ("chunked", "qwen",     "Chinese"),
    "ja":    ("chunked", "qwen",     "Japanese"),
}
# auto / unknown -> Qwen (broadest coverage incl. Turkish, plus context biasing)
_DEFAULT_ROUTE = ("chunked", "qwen", None)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


# --------------------------------------------------------------------------- #
# Model singletons
# --------------------------------------------------------------------------- #
_models: dict[str, object] = {}


def _torch():
    import torch
    return torch


def _get_diarizer():
    if "diar" not in _models:
        from nemo.collections.asr.models import SortformerEncLabelModel
        m = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL).eval()
        torch = _torch()
        if torch.cuda.is_available():
            m = m.to("cuda")
        _models["diar"] = m
        log.info("Loaded diarizer %s", SORTFORMER_MODEL)
    return _models["diar"]


def _get_parakeet():
    if "parakeet" not in _models:
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL)
        # Long meetings: switch the FastConformer encoder to limited-context local
        # attention so memory grows linearly (global attention is quadratic and
        # OOMs on long audio). We still chunk below as a hard safety bound.
        try:
            m.change_attention_model("rel_pos_local_attn", [256, 256])
            log.info("Parakeet: enabled local attention for long audio")
        except Exception as e:  # pragma: no cover - depends on NeMo version
            log.warning("Parakeet: could not set local attention: %s", e)
        _models["parakeet"] = m
        log.info("Loaded ASR %s", PARAKEET_MODEL)
    return _models["parakeet"]


def _get_canary():
    if "canary" not in _models:
        import nemo.collections.asr as nemo_asr
        _models["canary"] = nemo_asr.models.ASRModel.from_pretrained(CANARY_MODEL)
        log.info("Loaded ASR %s", CANARY_MODEL)
    return _models["canary"]


def _get_qwen():
    if "qwen" not in _models:
        from qwen_asr import Qwen3ASRModel
        torch = _torch()
        _models["qwen"] = Qwen3ASRModel.from_pretrained(
            QWEN_MODEL, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=440,
        )
        log.info("Loaded ASR %s", QWEN_MODEL)
    return _models["qwen"]


# --------------------------------------------------------------------------- #
# Audio prep
# --------------------------------------------------------------------------- #
def _to_wav_16k_mono(src: Path, dst: Path) -> None:
    """Decode any input to 16 kHz mono PCM WAV. Light touch — Sortformer and the
    transducer/LLM ASRs are noise-robust, so we avoid heavy normalization (which
    historically erased the per-speaker cues diarization needs)."""
    ff = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    cmd = [ff, "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
           "-af", "highpass=f=60", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
def _norm_speaker(raw: str) -> str:
    # "speaker_0" -> "SPEAKER_00"
    digits = "".join(c for c in raw if c.isdigit())
    return f"SPEAKER_{int(digits):02d}" if digits else raw.upper()


def diarize(wav: Path) -> list[Turn]:
    diar = _get_diarizer()
    out = diar.diarize(audio=[str(wav)], batch_size=1)
    pred = out[0] if isinstance(out, (list, tuple)) and out else out
    turns: list[Turn] = []
    for seg in pred or []:
        if isinstance(seg, str):
            p = seg.split()
            if len(p) >= 3:
                turns.append(Turn(float(p[0]), float(p[1]), _norm_speaker(p[2])))
    turns.sort(key=lambda t: t.start)
    return turns


def _speaker_at(t: float, turns: list[Turn]) -> str:
    for tr in turns:
        if tr.start <= t <= tr.end:
            return tr.speaker
    if not turns:
        return "SPEAKER_00"
    return min(turns, key=lambda tr: min(abs(tr.start - t), abs(tr.end - t))).speaker


def _merge_speaker_turns(turns: list[Turn], max_gap: float = 0.75,
                         max_len: float = 40.0) -> list[Turn]:
    """Collapse the many small Sortformer segments into contiguous speaker turns
    for chunked ASR. Merge same-speaker segments separated by < max_gap, capping
    a merged turn at max_len seconds so ASR chunks stay a sane size."""
    if not turns:
        return []
    merged = [Turn(turns[0].start, turns[0].end, turns[0].speaker)]
    for tr in turns[1:]:
        last = merged[-1]
        if (tr.speaker == last.speaker and tr.start - last.end <= max_gap
                and tr.end - last.start <= max_len):
            last.end = max(last.end, tr.end)
        else:
            merged.append(Turn(tr.start, tr.end, tr.speaker))
    return merged


# --------------------------------------------------------------------------- #
# ASR strategies
# --------------------------------------------------------------------------- #
def _blocks_from_wordts(words: list[dict], turns: list[Turn]) -> list[Block]:
    """Parakeet path: assign each word to the speaker whose turn holds its
    midpoint, then merge consecutive same-speaker words into blocks."""
    raw: list[Block] = []
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        spk = _speaker_at((s + e) / 2, turns)
        tok = str(w["word"])
        if raw and raw[-1].speaker == spk:
            raw[-1].end = e
            raw[-1].text = (raw[-1].text + " " + tok).strip()
        else:
            raw.append(Block(spk, s, e, tok))
    return raw


# Transcribe long audio in windows so memory stays bounded regardless of length.
_PARAKEET_WINDOW_S = float(os.environ.get("PARAKEET_WINDOW_S", "480"))


def _parakeet_words(hyps, offset: float) -> list[dict]:
    ts = getattr(hyps[0], "timestamp", None) or {}
    out: list[dict] = []
    for w in ts.get("word", []):
        out.append({"word": w["word"],
                    "start": float(w["start"]) + offset,
                    "end": float(w["end"]) + offset})
    return out


def _asr_parakeet_wordts(wav: Path, turns: list[Turn], force_lang: str | None) -> list[Block]:
    asr = _get_parakeet()
    data, sr = _load_audio(wav)
    dur = len(data) / sr if sr else 0.0
    words: list[dict] = []
    if dur <= _PARAKEET_WINDOW_S:
        words = _parakeet_words(asr.transcribe([str(wav)], timestamps=True), 0.0)
    else:
        import soundfile as sf
        start = 0.0
        while start < dur:
            a = int(start * sr)
            b = min(len(data), int((start + _PARAKEET_WINDOW_S) * sr))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                sf.write(tf.name, data[a:b], sr)
                words += _parakeet_words(asr.transcribe([tf.name], timestamps=True), start)
            start += _PARAKEET_WINDOW_S
    if not words:  # no timestamps -> whole file as one dominant-speaker block
        spk = turns[0].speaker if turns else "SPEAKER_00"
        end = turns[-1].end if turns else 0.0
        return [Block(spk, 0.0, end, "")]
    return _blocks_from_wordts(words, turns)


def _load_audio(wav: Path):
    import soundfile as sf
    data, sr = sf.read(str(wav), dtype="float32")
    return data, sr


def _asr_chunked(wav: Path, turns: list[Turn], model_key: str,
                 force_lang: str | None, context: str) -> list[Block]:
    """Canary/Qwen path: transcribe each merged speaker turn independently, so
    speaker attribution is exact by construction and code-switching stays intact."""
    data, sr = _load_audio(wav)
    merged = _merge_speaker_turns(turns)
    blocks: list[Block] = []
    if model_key == "qwen":
        qwen = _get_qwen()
    else:
        canary = _get_canary()
    for tr in merged:
        a = max(0, int(tr.start * sr))
        b = min(len(data), int(tr.end * sr))
        if b - a < int(0.2 * sr):  # skip sub-200ms slivers
            continue
        clip = data[a:b]
        if model_key == "qwen":
            res = qwen.transcribe(audio=(clip, sr), language=force_lang, context=context or "")
            text = (res[0].text or "").strip()
        else:  # canary — needs a temp wav + source/target lang
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                sf.write(tf.name, clip, sr)
                out = canary.transcribe([tf.name], source_lang=force_lang or "en",
                                        target_lang=force_lang or "en", pnc="yes")
                h = out[0]
                text = (getattr(h, "text", None) or str(h)).strip()
        if text:
            blocks.append(Block(tr.speaker, tr.start, tr.end, text))
    return blocks


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
def transcribe_file(
    src: Path,
    *,
    output_dir: Path,
    source_name: str | None = None,
    progress: ProgressCb | None = None,
    engine: str | None = None,        # ignored (kept for signature compat)
    language: str | None = None,
    model: str | None = None,         # ignored — model is chosen by the router
    initial_prompt: str | None = None,
    context: str | None = None,       # glossary for Qwen context-biasing
    num_speakers: int | None = None,  # Sortformer auto-detects up to 4
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[Path, TranscriptionResult]:
    def _p(stage: str, frac: float | None = None) -> None:
        if progress:
            progress(stage, frac)

    src = Path(src)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang = (language or "auto").lower()
    strategy, model_key, force_lang = _ROUTER.get(lang, _DEFAULT_ROUTE)
    ctx = "\n".join(x for x in (DEFAULT_CONTEXT, context) if x).strip()
    label = {"parakeet": PARAKEET_MODEL, "canary": CANARY_MODEL, "qwen": QWEN_MODEL}[model_key]

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        _p("Preparing audio…", 0.05)
        _to_wav_16k_mono(src, wav)

        _p("Running speaker diarization (Sortformer v2.1)…", 0.15)
        turns = diarize(wav)
        dist: dict[str, float] = {}
        for t in turns:
            dist[t.speaker] = dist.get(t.speaker, 0.0) + (t.end - t.start)
        log.info("Diarization: %d segs, %s", len(turns),
                 {k: f"{v:.0f}s" for k, v in sorted(dist.items())})

        _p(f"Transcribing with {label.rsplit('/', 1)[-1]}…", 0.5)
        if strategy == "wordts":
            blocks = _asr_parakeet_wordts(wav, turns, force_lang)
        else:
            blocks = _asr_chunked(wav, turns, model_key, force_lang, ctx)

        # merge consecutive same-speaker blocks
        _p("Assembling transcript…", 0.9)
        final: list[Block] = []
        for b in blocks:
            if final and final[-1].speaker == b.speaker:
                final[-1].end = b.end
                final[-1].text = (final[-1].text + " " + b.text).strip()
            else:
                final.append(b)

        duration = max((b.end for b in final), default=(turns[-1].end if turns else 0.0))
        display_name = source_name or src.stem
        md = to_markdown(final, source_name=display_name, language=lang, duration_seconds=duration)

        out_path = output_dir / f"{display_name}.md"
        n = 1
        while out_path.exists():
            out_path = output_dir / f"{display_name}-{n}.md"
            n += 1
        out_path.write_text(md, encoding="utf-8")
        _p("Done", 1.0)

        result = TranscriptionResult(
            markdown=md,
            language=lang,
            duration_seconds=duration,
            num_speakers=len({b.speaker for b in final}),
            num_blocks=len(final),
        )
        return out_path, result
