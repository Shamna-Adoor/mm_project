"""Video chatbot — answers questions and returns seek timestamps."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from analyzer._logging import get_logger

log = get_logger(__name__)

INTERMEDIATE_ROOT = Path(__file__).parent.parent / "data" / "intermediate"
PREDICTIONS_ROOT  = Path(__file__).parent.parent / "data" / "predictions"
OLLAMA_CHAT_URL   = "http://localhost:11434/api/chat"
DEFAULT_MODEL     = "llama3.1:8b"


# ── Context loaders ───────────────────────────────────────────────────────────

def load_video_context(job_id: str) -> dict:
    ctx = {"transcript": [], "segments": [], "chapters": [], "duration": 0.0}
    pred = PREDICTIONS_ROOT / f"{job_id}.json"
    if pred.exists():
        data = json.loads(pred.read_text())
        ctx["segments"] = data.get("segments", [])
        ctx["chapters"] = data.get("chapters", [])
        ctx["duration"] = float(data.get("duration_seconds", 0))
    tx = INTERMEDIATE_ROOT / job_id / "transcript.json"
    if tx.exists():
        ctx["transcript"] = json.loads(tx.read_text())
    return ctx


def _fmt_segments(segs: list[dict]) -> str:
    lines = []
    for s in segs:
        m1, s1 = divmod(int(s["start"]), 60)
        m2, s2 = divmod(int(s["end"]),   60)
        skip = "SKIP" if s.get("skip_recommended") else "keep"
        lines.append(f"  [{m1}:{s1:02d}–{m2}:{s2:02d}] {s['label'].upper()} ({skip})")
    return "\n".join(lines) or "  (none)"


def _fmt_chapters(chaps: list[dict]) -> str:
    lines = []
    for c in chaps:
        m, s = divmod(int(c["start"]), 60)
        lines.append(f"  {m}:{s:02d}  {c['title']}")
    return "\n".join(lines) or "  (none)"


def _fmt_transcript(transcript: list[dict], max_chars: int = 7000) -> str:
    lines = [f"[{int(s['start'])}s] {s['text'].strip()}" for s in transcript if s.get("text","").strip()]
    full = "\n".join(lines)
    if len(full) > max_chars:
        half = max_chars // 2
        full = full[:half] + "\n… [middle omitted] …\n" + full[-half // 2:]
    return full


def _parse_time_mention(text: str) -> float | None:
    m = re.search(r'(\d{1,2}):(\d{2})', text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r'(\d+(?:\.\d+)?)\s*min', text, re.I)
    if m:
        return float(m.group(1)) * 60
    m = re.search(r'(\d+(?:\.\d+)?)\s*sec', text, re.I)
    if m:
        return float(m.group(1))
    return None


# ── Main chat function ────────────────────────────────────────────────────────

def chat(
    message: str,
    history: list[dict],
    job_id: str,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Return {reply, seek_to, timestamps} from LLM with full video context."""
    ctx = load_video_context(job_id)
    dur = ctx["duration"]
    dur_str = f"{int(dur//60)}m {int(dur%60)}s"

    system = f"""You are a smart video assistant. Video duration: {dur_str}.

SEGMENTS (label / skip status):
{_fmt_segments(ctx['segments'])}

CHAPTERS:
{_fmt_chapters(ctx['chapters'])}

FULL TRANSCRIPT (time in seconds):
{_fmt_transcript(ctx['transcript'])}

RULES:
- Answer questions about the video based on the transcript.
- If the user wants to navigate (go to, show me, jump to, play from, find), set seek_to to the correct second.
- For "skip to X topic": find the best matching chapter/transcript timestamp.
- For "go to the sponsor/intro/outro": find that segment's start time.
- Timestamps in your reply should be written as MM:SS (e.g. 5:30).
- Be concise but helpful.

Respond ONLY with valid JSON — no markdown, no extra text:
{{"reply": "...", "seek_to": null, "timestamps": [{{"time": 0, "label": "..."}}]}}
seek_to is null or a number of seconds. timestamps is an optional array of references."""

    messages = [{"role": "system", "content": system}]
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        payload = json.dumps({
            "model":   model,
            "messages": messages,
            "stream":  False,
            "options": {"temperature": 0.25, "num_predict": 512},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_CHAT_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())["message"]["content"].strip()

        # Parse JSON
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return {"reply": raw, "seek_to": None, "timestamps": []}

        parsed = json.loads(m.group())
        reply  = str(parsed.get("reply", raw))
        seek_to = parsed.get("seek_to")
        if seek_to is not None:
            try:
                seek_to = float(seek_to)
                if seek_to < 0 or seek_to > dur + 10:
                    seek_to = None
            except (ValueError, TypeError):
                seek_to = None

        timestamps = [
            t for t in parsed.get("timestamps", [])
            if isinstance(t.get("time"), (int, float)) and 0 <= t["time"] <= dur + 10
        ]
        log.info("Chat reply (seek_to=%s): %s…", seek_to, reply[:80])
        return {"reply": reply, "seek_to": seek_to, "timestamps": timestamps}

    except Exception as exc:
        log.warning("Chat failed: %s", exc)
        return {
            "reply": "Ollama isn't responding. Make sure it's running with `ollama serve`.",
            "seek_to": None,
            "timestamps": [],
        }
