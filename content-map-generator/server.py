"""FastAPI backend — video upload/download, pipeline, status, chat."""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Request

from fastapi.responses import StreamingResponse

from analyzer._logging import get_logger
from analyzer.source_resolver import download_source, resolve_source

log = get_logger(__name__)

app = FastAPI(title="Content Map Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR        = Path("data/uploads")
INTERMEDIATE_ROOT = Path("data/intermediate")
PREDICTIONS_ROOT  = Path("data/predictions")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _initial_status(path: Path, message: str = "Queued — starting…") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"stage_num": 0, "total_stages": 9, "message": message,
                   "percent": 0, "status": "processing"}, f)


def _write_status(path: Path, stage_num: int, message: str, percent: int, status: str = "processing") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"stage_num": stage_num, "total_stages": 9,
                   "message": message, "percent": percent, "status": status}, f)


def _error_status(path: Path, message: str) -> None:
    with open(path, "w") as f:
        json.dump({"stage_num": 0, "total_stages": 9,
                   "message": message, "percent": 0, "status": "error"}, f)


def _find_video_file(job_id: str) -> Path | None:
    for ext in ("mp4", "mkv", "webm", "mov", "avi", "m4v"):
        p = UPLOAD_DIR / f"{job_id}.{ext}"
        if p.exists():
            return p
    return None


def _safe_job_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", value):
        raise HTTPException(400, "Invalid job id")
    return value


# ── Local file upload ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    whisper_model: str = "base",
    skip_visual: bool = False,
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    job_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(video.filename or "upload.mp4").stem.lower()).strip("_") or "upload"
    dest   = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(video.file, f)
    await video.close()

    status_path = INTERMEDIATE_ROOT / job_id / "status.json"
    _initial_status(status_path)
    log.info("Queued local job %s ← %s", job_id, dest)

    threading.Thread(
        target=_run_pipeline_thread,
        args=(dest, job_id, whisper_model, skip_visual),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "processing"}


# ── External URL import (YouTube, Twitch, direct media, yt-dlp sources) ───────

@app.post("/api/import-url")
async def import_url(
    source_url: str = Form(...),
    analyze: bool = Form(True),
    stream_only: bool = Form(False),
    whisper_model: str = Form("base"),
    skip_visual: bool = Form(False),
):
    """Resolve an external URL for playback and optionally queue segmentation."""
    try:
        source = resolve_source(source_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Source resolution failed")
        raise HTTPException(502, f"Could not resolve source: {exc}") from exc

    job_id = source["job_id"]
    status_path = INTERMEDIATE_ROOT / job_id / "status.json"
    playback_url = source.get("playback_url") or source.get("source_url")
    can_analyze = bool(source.get("can_analyze"))

    if stream_only or not analyze or not can_analyze:
        reason = "Live or stream-only source. Playback is available; offline segmentation is not queued."
        if can_analyze and not analyze:
            reason = "Playback only. Segmentation was not requested."
        _write_status(status_path, 0, reason, 100, status="streaming")
        return {
            "job_id": job_id,
            "status": "streaming",
            "analysis_supported": can_analyze,
            "playback_url": playback_url,
            "source": source,
        }

    _initial_status(status_path, "Queued — resolving and downloading source…")
    threading.Thread(
        target=_download_and_run_thread,
        args=(source, whisper_model, skip_visual, status_path),
        daemon=True,
    ).start()

    log.info("Queued URL job %s <- %s", job_id, source.get("webpage_url") or source_url)
    return {
        "job_id": job_id,
        "status": "processing",
        "analysis_supported": True,
        "playback_url": playback_url,
        "source": source,
    }


# ── YouTube compatibility route ───────────────────────────────────────────────

@app.post("/api/youtube")
async def download_youtube(youtube_url: str = Form(...)):
    """Backward-compatible wrapper for older player builds."""
    return await import_url(
        source_url=youtube_url,
        analyze=True,
        stream_only=False,
        whisper_model="base",
        skip_visual=False,
    )


# ── Status + result ───────────────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job_id = _safe_job_id(job_id)
    p = INTERMEDIATE_ROOT / job_id / "status.json"
    if not p.exists():
        raise HTTPException(404, "Job not found")
    with open(p) as f:
        return json.load(f)


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    job_id = _safe_job_id(job_id)
    p = PREDICTIONS_ROOT / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    with open(p) as f:
        return json.load(f)


# ── Video file serving (for YouTube downloads) ────────────────────────────────

@app.get("/api/video/{job_id}")
def serve_video(job_id: str, request: Request):
    """Stream a server-side video file with HTTP Range support."""
    job_id = _safe_job_id(job_id)
    path = _find_video_file(job_id)
    if not path:
        raise HTTPException(404, "Video file not found on server")

    mime = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".m4v": "video/mp4",
    }.get(path.suffix.lower(), "video/mp4")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    # Browser asked for a byte range: return 206 Partial Content.
    if range_header:
        try:
            units, range_spec = range_header.split("=", 1)
            if units.strip().lower() != "bytes":
                raise ValueError("Unsupported range unit")

            start_str, end_str = (range_spec.split("-", 1) + [""])[:2]
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1

            if start < 0 or end < start or start >= file_size:
                raise ValueError("Invalid byte range")

            end = min(end, file_size - 1)
            chunk_size = end - start + 1

            def iterfile():
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        data = f.read(min(1024 * 1024, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(
                iterfile(),
                status_code=206,
                media_type=mime,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(chunk_size),
                },
            )
        except Exception as exc:
            log.warning("Bad Range header %r for job %s: %s", range_header, job_id, exc)

    # Full-file fallback.
    return FileResponse(
        str(path),
        media_type=mime,
        headers={"Accept-Ranges": "bytes"},
    )
# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []


@app.post("/api/chat/{job_id}")
def chat_endpoint(job_id: str, body: ChatRequest):
    """Chat about the analyzed video. Returns reply + optional seek_to timestamp."""
    pred_path = PREDICTIONS_ROOT / f"{job_id}.json"
    if not pred_path.exists():
        raise HTTPException(404, "Analysis not complete — run analysis first.")
    from analyzer.chat import chat
    return chat(body.message, body.history, job_id)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── React static build ────────────────────────────────────────────────────────

_STATIC = Path("player/dist")
if _STATIC.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")


# ── Background worker (local upload) ─────────────────────────────────────────

def _run_pipeline_thread(video_path: Path, job_id: str, whisper_model: str, skip_visual: bool) -> None:
    status_path = INTERMEDIATE_ROOT / job_id / "status.json"
    try:
        from analyzer.pipeline import run_pipeline
        run_pipeline(video_path, whisper_model=whisper_model,
                     skip_visual=skip_visual, status_path=status_path)
    except Exception as exc:
        log.exception("Pipeline failed for job %s", job_id)
        _error_status(status_path, str(exc))


def _download_and_run_thread(source: dict, whisper_model: str, skip_visual: bool, status_path: Path) -> None:
    job_id = source["job_id"]
    try:
        def progress(percent: int, message: str) -> None:
            _write_status(status_path, 0, message, percent)

        video_path = download_source(
            source["source_url"],
            UPLOAD_DIR,
            job_id,
            progress=progress,
        )

        from analyzer.pipeline import run_pipeline
        run_pipeline(
            video_path,
            whisper_model=whisper_model,
            skip_visual=skip_visual,
            status_path=status_path,
            source_metadata=source,
        )
    except Exception as exc:
        log.exception("URL pipeline failed for job %s", job_id)
        _error_status(status_path, str(exc))
