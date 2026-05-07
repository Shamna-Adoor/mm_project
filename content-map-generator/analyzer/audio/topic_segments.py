"""Generate topic-based chapter titles from a transcript using a local Ollama model."""

from __future__ import annotations

import json
import urllib.request
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"
WINDOW_SECS   = 120  # group transcript into ~2-minute windows


class Chapter(TypedDict):
    start:   float
    end:     float
    title:   str


def generate_chapters(
    transcript: list[dict],
    video_duration: float,
    *,
    model: str = DEFAULT_MODEL,
    window_seconds: float = WINDOW_SECS,
) -> list[Chapter]:
    """Segment the transcript into topical chapters with LLM-generated titles.

    Falls back to plain 'Part N' titles if Ollama is unreachable.
    """
    if not transcript:
        return []

    windows = _build_windows(transcript, video_duration, window_seconds)
    chapters: list[Chapter] = []

    for i, (start, end, text) in enumerate(windows, 1):
        title = _title_for_window(text, i, model)
        chapters.append({"start": round(start, 3), "end": round(end, 3), "title": title})
        log.info("Chapter %d/%d [%.0fs–%.0fs]: %s", i, len(windows), start, end, title)

    return chapters


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_windows(
    transcript: list[dict],
    video_duration: float,
    window_seconds: float,
) -> list[tuple[float, float, str]]:
    """Bucket transcript segments into fixed-size time windows."""
    windows: list[tuple[float, float, str]] = []
    bucket_start = transcript[0]["start"]
    bucket_texts: list[str] = []

    for seg in transcript:
        bucket_texts.append(seg["text"])
        elapsed = seg["end"] - bucket_start
        if elapsed >= window_seconds:
            windows.append((bucket_start, seg["end"], " ".join(bucket_texts).strip()))
            bucket_start = seg["end"]
            bucket_texts = []

    # remaining text
    if bucket_texts:
        last_end = transcript[-1]["end"]
        windows.append((bucket_start, last_end, " ".join(bucket_texts).strip()))

    return windows


def _title_for_window(text: str, index: int, model: str) -> str:
    """Ask Ollama for a short topic title. Returns 'Part N' on failure."""
    # Truncate to ~800 chars so the prompt stays fast
    snippet = text[:800].strip()
    if not snippet:
        return f"Part {index}"

    prompt = (
        "You are creating YouTube chapter titles. "
        "Given the transcript excerpt below, write a SHORT topic title (4–7 words max). "
        "Output ONLY the title — no punctuation at the end, no quotes, no explanation.\n\n"
        f"Transcript:\n{snippet}"
    )

    try:
        payload = json.dumps({
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 20},
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            title = data.get("response", "").strip().strip('"').strip("'")
            # keep only first line and cap at 60 chars
            title = title.splitlines()[0].strip()[:60]
            return title if title else f"Part {index}"

    except Exception as exc:
        log.warning("Ollama title generation failed (chapter %d): %s", index, exc)
        return f"Part {index}"
