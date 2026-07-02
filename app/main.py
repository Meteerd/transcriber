"""FastAPI server: drag-drop upload → background transcription → polled status."""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # Must come BEFORE importing transcribe (env vars read at import time)

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.transcribe import (
    FASTER_WHISPER_MODEL_FULL,
    FASTER_WHISPER_MODEL_TURBO,
    MIXED_LANGUAGE_PROMPTS,
    WHISPER_MODEL_TURBO,
    WHISPER_MODEL_FULL,
)

# Engine backend. "nemo" = the containerized Sortformer + per-language ASR router
# (Parakeet / Canary / Qwen3-ASR). Anything else keeps the legacy Whisper+pyannote
# pipeline. Selected by env so the same codebase runs in both environments.
_BACKEND = os.environ.get("TRANSCRIBER_BACKEND", "").lower()
if _BACKEND == "nemo":
    from app.nemo_pipeline import transcribe_file
else:
    from app.transcribe import transcribe_file

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
TRANSCRIPTS = ROOT / "transcripts"
STATIC = ROOT / "static"
# Persists "last used names" across sessions so the UI can pre-fill the rename inputs
NAME_MEMORY = ROOT / "transcripts" / ".last-names.json"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "2048"))
UPLOADS.mkdir(exist_ok=True)
TRANSCRIPTS.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transcriber")

app = FastAPI(title="Local Transcriber")


# --------------------------------------------------------------------------- #
# Runtime / accelerator detection.
#
# The engine is decided by THIS machine's hardware, never by the user. On the
# CUDA box it locks to faster-whisper/CUDA; on a Mac it locks to mlx/Apple
# Silicon. The UI shows the locked engine as a read-only badge — there is no
# engine dropdown. The user only configures language / quality / speakers.
# --------------------------------------------------------------------------- #
_runtime_cache: dict[str, str] | None = None


def detect_runtime() -> dict[str, str]:
    """Return the engine + human labels for the hardware this server runs on.

    Cached after first call. Detection order: CUDA GPU → Apple Silicon (MPS) →
    CPU fallback (honoring an explicit WHISPER_ENGINE env if set).
    """
    global _runtime_cache
    if _runtime_cache is not None:
        return _runtime_cache

    engine = (os.environ.get("WHISPER_ENGINE") or "faster-whisper").lower()
    engine_label = "CPU"
    accelerator = "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            engine = "faster-whisper"
            engine_label = "CUDA"
            try:
                accelerator = torch.cuda.get_device_name(0)
            except Exception:
                accelerator = "CUDA GPU"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            engine = "mlx"
            engine_label = "Apple Silicon"
            accelerator = "MLX"
        else:
            engine_label = "Apple Silicon" if engine == "mlx" else "CPU"
            accelerator = "MLX" if engine == "mlx" else "CPU"
    except Exception:
        engine_label = "Apple Silicon" if engine == "mlx" else "CPU"
        accelerator = "Apple Silicon (MLX)" if engine == "mlx" else "CPU"

    _runtime_cache = {
        "engine": engine,
        "engine_label": engine_label,
        "accelerator": accelerator,
    }
    log.info("Runtime engine locked to %s (%s / %s)", engine, engine_label, accelerator)
    return _runtime_cache


@app.middleware("http")
async def require_access_token(request: Request, call_next):
    token = os.environ.get("TRANSCRIBER_ACCESS_TOKEN")
    if not token:
        return await call_next(request)

    header = request.headers.get("authorization", "")
    scheme, _, encoded = header.partition(" ")
    username = password = ""
    if scheme.lower() == "basic" and encoded:
        try:
            import base64

            decoded = base64.b64decode(encoded).decode("utf-8")
            username, _, password = decoded.partition(":")
        except Exception:
            username = password = ""

    if secrets.compare_digest(username, "transcriber") and secrets.compare_digest(password, token):
        return await call_next(request)

    return JSONResponse(
        {"detail": "Authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Transcriber"'},
    )

# Job registry — in-memory only. Restart = jobs lost (transcripts persist on disk).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _stage_key(stage: str) -> str:
    s = stage.lower()
    if "preparing" in s or "audio" in s:
        return "prepare"
    if "diarization" in s:
        return "diarize"
    if "transcribing" in s or "model" in s:
        return "transcribe"
    if "assigning" in s:
        return "assign"
    if "done" in s:
        return "done"
    return "queued"


def _update_job(job_id: str, **kv: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kv)


def _run_job(
    job_id: str,
    upload_path: Path,
    original_name: str,
    *,
    engine: str | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    model: str | None = None,
    num_speakers: int | None = None,
    context: str | None = None,
) -> None:
    def progress(stage: str, frac: float | None) -> None:
        _update_job(
            job_id,
            stage=stage,
            stage_key=_stage_key(stage),
            progress=frac,
            updated_at=datetime.now().isoformat(),
        )
        log.info("[%s] %s %s", job_id[:8], stage, f"({frac*100:.0f}%)" if frac else "")

    try:
        _update_job(
            job_id,
            status="running",
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        extra = {"context": context} if _BACKEND == "nemo" else {}
        out_path, result = transcribe_file(
            upload_path,
            output_dir=TRANSCRIPTS,
            source_name=Path(original_name).stem,
            progress=progress,
            engine=engine,
            language=language,
            initial_prompt=initial_prompt,
            model=model,
            num_speakers=num_speakers,
            **extra,
        )
        _update_job(
            job_id,
            status="done",
            finished_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            stage_key="done",
            transcript_path=str(out_path),
            transcript_name=out_path.name,
            markdown=result.markdown,
            language=result.language,
            duration_seconds=result.duration_seconds,
            num_speakers=result.num_speakers,
            num_blocks=result.num_blocks,
            speakers=_detect_speakers_in_md(result.markdown),
        )
        log.info("[%s] complete → %s", job_id[:8], out_path)
    except Exception as e:
        log.exception("[%s] failed", job_id[:8])
        _update_job(
            job_id,
            status="error",
            error=str(e),
            finished_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
    finally:
        # Best-effort cleanup of upload
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/config")
async def runtime_config() -> JSONResponse:
    """Tell the UI which engine this machine is locked to, for the badge."""
    rt = detect_runtime()
    if _BACKEND == "nemo":
        return JSONResponse(
            {
                "engine": "nemo",
                "engine_label": "NeMo",
                "accelerator": f"Sortformer + Parakeet/Qwen · {rt['accelerator']}",
            }
        )
    return JSONResponse(
        {
            "engine": rt["engine"],
            "engine_label": rt["engine_label"],
            "accelerator": rt["accelerator"],
        }
    )


_VALID_QUALITY = {"turbo", "full"}
_VALID_ENGINES = {"mlx", "faster-whisper"}
_VALID_LANGS = {
    None,
    "",
    "auto",
    "en",
    "tr",
    "tr-en",
    "tr-hu",
    "hu",
    "hu-en",
    "de",
    "fr",
    "es",
    "it",
    "ar",
    "ru",
    "zh",
    "ja",
}


def _model_for(engine: str, quality: str) -> str:
    if engine == "faster-whisper":
        return FASTER_WHISPER_MODEL_FULL if quality == "full" else FASTER_WHISPER_MODEL_TURBO
    return WHISPER_MODEL_FULL if quality == "full" else WHISPER_MODEL_TURBO


def _parse_speakers(raw: str | None) -> int | None:
    if raw in (None, "", "auto"):
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise HTTPException(400, "speakers must be auto or a positive integer") from e
    if value < 1 or value > 20:
        raise HTTPException(400, "speakers must be between 1 and 20")
    return value


def _language_settings(raw: str | None) -> tuple[str | None, str | None, str]:
    if raw in (None, "", "auto"):
        return None, None, "auto"
    if raw in MIXED_LANGUAGE_PROMPTS:
        return None, MIXED_LANGUAGE_PROMPTS[raw], raw
    return raw, None, raw


def _copy_upload_with_limit(file: UploadFile, upload_path: Path) -> None:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    with upload_path.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large. MAX_UPLOAD_MB={MAX_UPLOAD_MB}")
            f.write(chunk)


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("auto"),  # accepted for backward compat, but ignored
    language: str = Form("auto"),
    quality: str = Form("turbo"),  # legacy; NeMo backend picks the model by language
    speakers: str = Form("auto"),
    context: str = Form(""),  # glossary of names/terms for code-switch biasing (NeMo)
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "No filename")
    # The engine is fixed by this machine's hardware, not the client. Whatever
    # the form sends is ignored — the CUDA box always runs faster-whisper/CUDA,
    # a Mac always runs mlx.
    engine = detect_runtime()["engine"]
    if quality not in _VALID_QUALITY:
        raise HTTPException(400, f"quality must be one of {_VALID_QUALITY}")
    if language not in _VALID_LANGS:
        raise HTTPException(400, f"unsupported language {language!r}")

    lang, initial_prompt, language_hint = _language_settings(language)
    model = _model_for(engine, quality)
    num_speakers = _parse_speakers(speakers)
    # The NeMo router keys on the full language code (incl. mixed variants like
    # "tr-en"/"tr-hu"); the legacy pipeline uses the resolved single-lang hint.
    job_language = language_hint if _BACKEND == "nemo" else lang

    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name  # strip any path
    upload_path = UPLOADS / f"{job_id}_{safe_name}"

    _copy_upload_with_limit(file, upload_path)

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "original_name": safe_name,
            "status": "queued",
            "stage": "Queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "language_hint": language_hint,
            "engine": engine,
            "model_quality": quality,
            "speaker_hint": num_speakers or "auto",
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, upload_path, safe_name),
        kwargs={
            "engine": engine,
            "language": job_language,
            "initial_prompt": initial_prompt,
            "model": model,
            "num_speakers": num_speakers,
            "context": context,
        },
        daemon=True,
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        # Return a copy so we don't leak mutable dict
        return JSONResponse(dict(job))


@app.get("/jobs")
async def list_jobs() -> JSONResponse:
    with _jobs_lock:
        # Newest first, omit large markdown body
        items = []
        for job in sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True):
            slim = {k: v for k, v in job.items() if k != "markdown"}
            items.append(slim)
        return JSONResponse(items)


@app.get("/transcripts/{name}")
async def download_transcript(name: str) -> FileResponse:
    path = TRANSCRIPTS / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


_SPEAKER_LABEL_RE = re.compile(r"^\*\*(SPEAKER_\d+)\*\*(\s*\[)", re.MULTILINE)


def _detect_speakers_in_md(md: str) -> list[str]:
    """Return the unique SPEAKER_NN labels found in a transcript, in first-seen order."""
    seen: list[str] = []
    for m in _SPEAKER_LABEL_RE.finditer(md):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def _apply_renames_to_md(md: str, mapping: dict[str, str]) -> str:
    """Replace **SPEAKER_NN** labels with **<new name>** in the markdown.

    Only replaces in the speaker-label position (line-start `**SPEAKER_NN**[`) so
    that any speaker IDs that happen to appear elsewhere in the body aren't touched.
    """
    def repl(m: re.Match[str]) -> str:
        old = m.group(1)
        new = mapping.get(old, old).strip() or old
        return f"**{new}**{m.group(2)}"
    return _SPEAKER_LABEL_RE.sub(repl, md)


def _load_name_memory() -> dict[str, str]:
    if not NAME_MEMORY.exists():
        return {}
    try:
        return json.loads(NAME_MEMORY.read_text())
    except Exception:
        return {}


def _save_name_memory(d: dict[str, str]) -> None:
    NAME_MEMORY.write_text(json.dumps(d, indent=2))


@app.get("/last-names")
async def last_names() -> JSONResponse:
    """Return the last-used name mapping so the UI can pre-fill rename inputs."""
    return JSONResponse(_load_name_memory())


@app.post("/jobs/{job_id}/rename")
async def rename_speakers(
    job_id: str,
    mapping: dict[str, str] = Body(...),
) -> JSONResponse:
    """Apply speaker renames to a job's transcript. Writes the new markdown to disk
    and updates the in-memory job state. Body is a {old_label: new_name} dict."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        if job.get("status") != "done":
            raise HTTPException(409, "Job not finished")
        old_md = job.get("markdown") or ""
        transcript_name = job.get("transcript_name")

    new_md = _apply_renames_to_md(old_md, mapping)
    if transcript_name:
        (TRANSCRIPTS / transcript_name).write_text(new_md, encoding="utf-8")

    with _jobs_lock:
        _jobs[job_id]["markdown"] = new_md
        # Recount speakers after renaming
        _jobs[job_id]["num_speakers"] = len(set(_detect_speakers_in_md(new_md)) |
                                            set(v.strip() for v in mapping.values() if v.strip()))

    # Remember these names for next time
    mem = _load_name_memory()
    for k, v in mapping.items():
        v = v.strip()
        if v:
            mem[k] = v
    _save_name_memory(mem)

    return JSONResponse({"ok": True, "markdown": new_md})


@app.post("/open-folder")
async def open_folder() -> JSONResponse:
    """Open the transcripts folder in Finder (macOS)."""
    try:
        subprocess.Popen(["open", str(TRANSCRIPTS)])
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/transcripts")
async def list_transcripts() -> JSONResponse:
    files = sorted(TRANSCRIPTS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse([
        {"name": p.name, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
         "size": p.stat().st_size}
        for p in files
    ])


# Mount any extra static files (not strictly needed since we inline index.html)
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
