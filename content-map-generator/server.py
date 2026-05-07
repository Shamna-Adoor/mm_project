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

from analyzer._logging import get_logger

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


# ── Local file upload ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    whisper_model: str = "base",
    skip_visual: bool = False,
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    job_id = Path(video.filename or "upload.mp4").stem.lower().replace(" ", "_").replace("-", "_")
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


# ── YouTube download ──────────────────────────────────────────────────────────

@app.post("/api/youtube")
async def download_youtube(youtube_url: str = Form(...)):
    """Download a YouTube video and run the analysis pipeline."""
    # Accept youtu.be/... and youtube.com/watch?v=...
    m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', youtube_url)
    if not m:
        raise HTTPException(400, "Invalid YouTube URL — couldn't extract video ID.")

    vid    = m.group(1)
    job_id = f"yt_{vid}"
    dest   = UPLOAD_DIR / f"{job_id}.mp4"
    status_path = INTERMEDIATE_ROOT / job_id / "status.json"
    _initial_status(status_path, "Downloading from YouTube…")

    threading.Thread(
        target=_youtube_thread,
        args=(youtube_url, dest, job_id, status_path),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "processing"}


def _youtube_thread(url: str, dest: Path, job_id: str, status_path: Path) -> None:
    try:
        import yt_dlp

        def _progress_hook(d: dict) -> None:
            if d.get("status") == "downloading":
                pct_str = re.sub(r'\x1b\[[0-9;]*m', '', d.get("_percent_str", "0%")).strip().rstrip("%")
                try:
                    pct = min(10, int(float(pct_str) * 0.10))
                except ValueError:
                    pct = 0
                with open(status_path, "w") as f:
                    json.dump({"stage_num": 0, "total_stages": 9,
                               "message": f"Downloading from YouTube… {pct_str}",
                               "percent": pct, "status": "processing"}, f)

        ydl_opts = {
            "format":     "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl":    str(dest.with_suffix(".%(ext)s")),
            "merge_output_format": "mp4",
            "quiet":      True,
            "progress_hooks": [_progress_hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            actual = Path(ydl.prepare_filename(info))
            if actual != dest and actual.exists():
                actual.rename(dest)

        log.info("YouTube download complete: %s", dest)
        from analyzer.pipeline import run_pipeline
        run_pipeline(dest, status_path=status_path)

    except Exception as exc:
        log.exception("YouTube pipeline failed for %s", job_id)
        _error_status(status_path, str(exc))


# ── Status + result ───────────────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    p = INTERMEDIATE_ROOT / job_id / "status.json"
    if not p.exists():
        raise HTTPException(404, "Job not found")
    with open(p) as f:
        return json.load(f)


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    p = PREDICTIONS_ROOT / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "Result not ready")
    with open(p) as f:
        return json.load(f)


@app.get("/api/transcript/{job_id}")
def get_transcript(job_id: str):
    p = INTERMEDIATE_ROOT / job_id / "transcript.json"
    if not p.exists():
        raise HTTPException(404, "Transcript not found")
    return FileResponse(str(p), media_type="application/json",
                        filename=f"{job_id}_transcript.json")


# ── Video file serving (for YouTube downloads) ────────────────────────────────

@app.get("/api/video/{job_id}")
def serve_video(job_id: str):
    """Stream a server-side video file (YouTube downloads)."""
    path = _find_video_file(job_id)
    if not path:
        raise HTTPException(404, "Video file not found on server")
    mime = {".mp4": "video/mp4", ".mkv": "video/x-matroska",
            ".webm": "video/webm", ".mov": "video/quicktime"}.get(path.suffix, "video/mp4")
    return FileResponse(str(path), media_type=mime)


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
