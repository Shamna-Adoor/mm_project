"""Use a local Ollama LLM to identify non-content segments from the transcript."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

SKIP_LABELS: frozenset[str] = frozenset({"intro", "sponsor", "outro", "dead_air"})


class LLMSegment(TypedDict):
    start:  float
    end:    float
    label:  str      # intro | outro | sponsor | dead_air | main_content
    reason: str


class Segment(TypedDict):
    start:             float
    end:               float
    label:             str
    confidence:        float
    skip_recommended:  bool
    reason:            str
    signals_used:      list[str]


def classify_transcript(
    transcript: list[dict],
    video_duration: float,
    *,
    model: str = DEFAULT_MODEL,
) -> list[LLMSegment]:
    """Ask the LLM to identify all non-content segments from the transcript.

    Returns a full segment list covering [0, video_duration].
    Falls back to an empty list (rule-based fusion takes over) if Ollama is unreachable.
    """
    if not transcript:
        return []

    transcript_text = _format_transcript(transcript)

    # For very long videos, the LLM context window may not fit the full transcript.
    # Keep the first 3 min + last 3 min (intro/outro) + every 3rd middle segment (sponsors).
    MAX_CHARS = 12_000
    if len(transcript_text) > MAX_CHARS:
        transcript_text = _trim_transcript(transcript, video_duration)

    prompt = f"""You are an expert video content analyst. Your job is to identify NON-CONTENT segments in this video transcript so they can be skipped.

Video duration: {int(video_duration)} seconds ({int(video_duration//60)}m {int(video_duration%60)}s)

DEFINITIONS:
- "intro": Opening sequence before the main topic begins. Includes: channel intro, greeting viewers, theme music, "welcome back", "today we're going to talk about", "I'm [name] and this is [channel]", animated logo sequences, countdown-style openers. Typically the first 30s–3min.
- "outro": Closing sequence after the main content ends. Includes: "thanks for watching", "subscribe and hit the bell", "see you next time", "like and share", end screen promotion, credits, sign-off music. Typically the last 30s–3min.
- "sponsor": Advertisement or paid promotion embedded in the video. Includes: "this video is sponsored by", "I want to thank our sponsor", "use code [X] for [discount]", product demos with purchase links, "check out [product] in the description", affiliate promotions, even subtle ones like "the app I've been using lately is...". Can appear anywhere in the video.
- "dead_air": Silence, blank screen, technical glitches, filler with no informational value, or a noticeable pause/gap between topics where the transcript has no text or only ambient sound.

TRANSCRIPT (format: [start_sec - end_sec] text):
{transcript_text}

STRICT RULES:
1. "intro" ONLY in the FIRST 20% of the video — not mid-video.
2. "outro" ONLY in the LAST 20% of the video — not mid-video transitions.
3. "sponsor" must be at least 8 seconds and contain a clear commercial message.
4. "dead_air": mark any gap where there is no transcript text for 3+ seconds, OR a noticeable silent/filler pause between topics.
5. Transitional FILLER ("um", "uh", "you know", extended silence) is dead_air. Brief connectors ("now let's look at") alone are NOT.
6. Merging: combine adjacent same-type segments into one.
7. For genuine content, keep it as content. For clear non-content, mark it.

Return ONLY a JSON array. Each element:
- "start": seconds (number, from transcript)
- "end": seconds (number, from transcript)
- "label": "intro" | "outro" | "sponsor" | "dead_air"
- "reason": one sentence

Return [] if no clear non-content exists.

JSON array only, no markdown, no other text:"""

    try:
        response = _call_ollama(prompt, model)
        segments = _parse_response(response, video_duration)
        log.info("LLM identified %d non-content segment(s)", len(segments))
        return _fill_main_content(segments, video_duration)
    except Exception as exc:
        log.warning("LLM classification failed: %s — falling back to rule-based", exc)
        return []


def to_output_segments(
    llm_segs: list[LLMSegment],
    silence_intervals: list[dict] | None = None,
    music_intervals:   list[dict] | None = None,
    video_duration:    float = 0.0,
) -> list[Segment]:
    """Convert LLM segments → final Segment list, overlaying audio signals.

    Strategy
    --------
    1. Build "forced" non-content intervals from audio signals:
       - Silence  >= 1.5 s  → dead_air
       - Music    >= 5 s near video start/end → intro / outro
    2. For every forced interval that falls inside a main_content LLM segment,
       *split* that segment and insert the forced label.
       (Non-content LLM labels like sponsor/intro/outro are never overwritten.)
    3. Validate LLM dead_air against silence data; revert to main_content if no
       actual silence detected there.
    4. Convert the final list to the Segment TypedDict format.
    """
    silence_intervals = silence_intervals or []
    music_intervals   = music_intervals   or []

    intro_cutoff = min(video_duration * 0.20, 300) if video_duration else 300
    outro_cutoff = max(video_duration - min(video_duration * 0.20, 300), 0) if video_duration else 0

    # ── Step 1: collect forced intervals from audio ───────────────────────────
    forced: list[dict] = []

    for sil in silence_intervals:
        dur = sil["end"] - sil["start"]
        if dur >= 1.5:
            forced.append({
                "start":  sil["start"], "end": sil["end"],
                "label":  "dead_air",
                "reason": f"Silence detected ({dur:.1f}s)",
                "conf":   0.90,
            })

    for mi in music_intervals:
        dur = mi["end"] - mi["start"]
        if dur < 5.0:
            continue
        if video_duration and mi["start"] < intro_cutoff:
            forced.append({
                "start":  mi["start"], "end": mi["end"],
                "label":  "intro",
                "reason": "Music detected near video start",
                "conf":   0.78,
            })
        elif video_duration and mi["end"] > outro_cutoff:
            forced.append({
                "start":  mi["start"], "end": mi["end"],
                "label":  "outro",
                "reason": "Music detected near video end",
                "conf":   0.78,
            })
        elif dur >= 8.0:
            # Mid-video music break (jingle, transition sting, sponsor music)
            forced.append({
                "start":  mi["start"], "end": mi["end"],
                "label":  "dead_air",
                "reason": f"Music break detected mid-video ({dur:.0f}s)",
                "conf":   0.72,
            })

    # ── Step 2: validate / revert LLM dead_air with silence data ─────────────
    def _silence_ratio(start: float, end: float) -> float:
        dur = end - start
        if dur <= 0 or not silence_intervals:
            return 0.0
        return sum(
            max(0.0, min(end, s["end"]) - max(start, s["start"]))
            for s in silence_intervals
        ) / dur

    # Work on a mutable copy with normalised structure
    working: list[dict] = []
    for seg in llm_segs:
        label = seg["label"]
        seg_dur = seg["end"] - seg["start"]
        if label == "dead_air" and silence_intervals:
            # Only revert if segment is long AND truly no audio silence
            # Short segments (< 15s): trust the LLM — background noise may prevent silence detection
            if seg_dur > 15.0 and _silence_ratio(seg["start"], seg["end"]) < 0.10:
                log.debug("Reverting long LLM dead_air [%.0f-%.0f] — no audio evidence", seg["start"], seg["end"])
                label = "main_content"
        working.append({
            "start":  seg["start"], "end": seg["end"],
            "label":  label,
            "reason": seg["reason"],
            "conf":   0.85 if label != "main_content" else 0.60,
        })

    # ── Step 3: split main_content segments where forced intervals land ───────
    if forced:
        forced_sorted = sorted(forced, key=lambda f: f["start"])
        for f in forced_sorted:
            new_working: list[dict] = []
            for seg in working:
                if seg["label"] != "main_content":
                    new_working.append(seg)
                    continue

                olap_s = max(f["start"], seg["start"])
                olap_e = min(f["end"],   seg["end"])
                if olap_e - olap_s < 1.0:          # no meaningful overlap
                    new_working.append(seg)
                    continue

                # Split: [seg.start … olap_s]  main_content
                #        [olap_s    … olap_e]  forced label
                #        [olap_e    … seg.end] main_content
                if olap_s - seg["start"] > 0.5:
                    new_working.append({**seg, "end": olap_s})
                new_working.append({
                    "start":  olap_s, "end": olap_e,
                    "label":  f["label"],
                    "reason": f["reason"],
                    "conf":   f["conf"],
                })
                if seg["end"] - olap_e > 0.5:
                    new_working.append({**seg, "start": olap_e})
            working = new_working

    working.sort(key=lambda s: s["start"])

    # ── Step 3b: merge adjacent same-label segments (gap ≤ 2 s) ──────────────
    merged: list[dict] = []
    for seg in working:
        if merged and merged[-1]["label"] == seg["label"]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= 2.0:
                merged[-1]["end"]  = seg["end"]
                merged[-1]["conf"] = max(merged[-1]["conf"], seg["conf"])
                continue
        merged.append(dict(seg))
    working = merged

    # ── Step 4: convert to Segment TypedDict ──────────────────────────────────
    result: list[Segment] = []
    for seg in working:
        label = seg["label"]
        signals: list[str] = []
        if label == seg.get("_orig_label", label):  # track signal source
            signals = ["llm_transcript"]
        if any(f["start"] <= seg["start"] and f["end"] >= seg["end"] for f in forced):
            signals = ["audio_signals"]
        if not signals:
            signals = ["llm_transcript"]

        result.append({
            "start":             round(seg["start"], 3),
            "end":               round(seg["end"],   3),
            "label":             label,
            "confidence":        round(min(1.0, seg["conf"]), 3),
            "skip_recommended":  label in SKIP_LABELS,
            "reason":            seg["reason"],
            "signals_used":      signals,
        })

    result = _normalize_output_segments(result, video_duration)
    log.info(
        "to_output_segments: %d total (%d skip) from %d LLM + %d silence + %d music forced",
        len(result),
        sum(1 for s in result if s["skip_recommended"]),
        len(llm_segs),
        sum(1 for f in forced if f["label"] == "dead_air"),
        sum(1 for f in forced if f["label"] in ("intro","outro")),
    )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trim_transcript(transcript: list[dict], video_duration: float) -> str:
    """For long videos, keep intro zone + sampled middle + outro zone."""
    intro_secs  = min(240, video_duration * 0.15)
    outro_secs  = max(video_duration - min(240, video_duration * 0.15), 0)

    intro_segs  = [s for s in transcript if s["start"] < intro_secs]
    outro_segs  = [s for s in transcript if s["start"] >= outro_secs]
    middle_segs = [s for i, s in enumerate(transcript)
                   if intro_secs <= s["start"] < outro_secs and i % 3 == 0]

    combined = intro_segs + middle_segs + outro_segs
    combined.sort(key=lambda s: s["start"])
    note = f"[NOTE: transcript sampled — {len(combined)}/{len(transcript)} segments shown]\n"
    return note + _format_transcript(combined)


def _normalize_output_segments(segments: list[Segment], video_duration: float) -> list[Segment]:
    """Clamp, de-overlap, and fill tiny coverage gaps with main content."""
    if not segments:
        return []

    duration = video_duration or max((s["end"] for s in segments), default=0.0)
    cursor = 0.0
    normalized: list[Segment] = []

    for seg in sorted(segments, key=lambda s: (s["start"], s["end"])):
        start = max(0.0, min(float(seg["start"]), duration))
        end = max(0.0, min(float(seg["end"]), duration))
        if end - start < 0.1:
            continue

        if start > cursor + 0.25:
            normalized.append(_main_content_gap(cursor, start))
        elif start < cursor:
            start = cursor

        if end - start < 0.1:
            continue

        fixed = dict(seg)
        fixed["start"] = round(start, 3)
        fixed["end"] = round(end, 3)
        normalized.append(fixed)  # type: ignore[arg-type]
        cursor = end

    if duration - cursor > 0.25:
        normalized.append(_main_content_gap(cursor, duration))

    merged: list[Segment] = []
    for seg in normalized:
        if merged and merged[-1]["label"] == seg["label"] and seg["start"] - merged[-1]["end"] <= 0.25:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = max(merged[-1]["confidence"], seg["confidence"])
            merged[-1]["signals_used"] = list(dict.fromkeys(merged[-1]["signals_used"] + seg["signals_used"]))
            continue
        merged.append(seg)

    return merged


def _main_content_gap(start: float, end: float) -> Segment:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": "main_content",
        "confidence": 0.55,
        "skip_recommended": False,
        "reason": "Coverage gap filled as content",
        "signals_used": ["coverage_normalization"],
    }


def _format_transcript(transcript: list[dict]) -> str:
    lines: list[str] = []
    prev_end: float = 0.0
    for seg in transcript:
        start = seg["start"]
        end   = seg["end"]
        text  = seg["text"].strip()
        # Emit gap marker when there's silence between segments
        gap = start - prev_end
        if gap >= 2.0 and prev_end > 0:
            lines.append(f"[GAP: {int(prev_end)}s–{int(start)}s — {gap:.0f}s no speech]")
        if text:
            lines.append(f"[{int(start)} - {int(end)}] {text}")
            prev_end = end
        elif gap >= 2.0:
            prev_end = start  # keep gap tracking even if text is empty
    return "\n".join(lines)


def _call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": 2048},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"]


def _parse_response(response: str, video_duration: float) -> list[LLMSegment]:
    """Extract and validate a JSON array from the LLM response."""
    text = re.sub(r"```(?:json)?", "", response).strip()
    text = re.sub(r"```", "", text).strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    raw = json.loads(match.group())

    # Position boundaries: intros only in first 20%, outros only in last 20%
    intro_cutoff = min(video_duration * 0.20, 300)   # max 5 min
    outro_cutoff = max(video_duration * 0.80, video_duration - 300)

    segments: list[LLMSegment] = []
    for item in raw:
        try:
            start = float(item["start"])
            end   = float(item["end"])
            label = str(item.get("label", "")).lower().strip()
            if label not in ("intro", "outro", "sponsor", "dead_air"):
                continue
            if end <= start or start < 0 or end > video_duration + 5:
                continue

            start = round(max(0.0, start), 3)
            end   = round(min(video_duration, end), 3)
            duration = end - start

            # Position filtering: intro must be near the start, outro near the end
            if label == "intro" and start > intro_cutoff:
                log.debug("Dropping mid-video 'intro' at %.0fs (cutoff %.0fs)", start, intro_cutoff)
                continue
            if label == "outro" and end < outro_cutoff:
                log.debug("Dropping early 'outro' ending at %.0fs (cutoff %.0fs)", end, outro_cutoff)
                continue

            # Minimum duration filters to avoid micro false positives
            min_dur = {"intro": 10.0, "outro": 10.0, "sponsor": 8.0, "dead_air": 5.0}
            if duration < min_dur.get(label, 2.0):
                log.debug("Dropping short '%s' (%.1fs) at %.0fs", label, duration, start)
                continue

            segments.append({
                "start":  start,
                "end":    end,
                "label":  label,
                "reason": str(item.get("reason", "")),
            })
        except (KeyError, ValueError, TypeError):
            continue

    return sorted(segments, key=lambda s: s["start"])


def _fill_main_content(
    non_content: list[LLMSegment],
    video_duration: float,
) -> list[LLMSegment]:
    """Insert main_content segments to fill the gaps between non-content."""
    if not non_content:
        return [{"start": 0.0, "end": video_duration, "label": "main_content", "reason": "No non-content detected"}]

    result: list[LLMSegment] = []
    cursor = 0.0

    for seg in non_content:
        if seg["start"] - cursor > 0.5:
            result.append({
                "start":  round(cursor, 3),
                "end":    round(seg["start"], 3),
                "label":  "main_content",
                "reason": "Content between non-content segments",
            })
        result.append(seg)
        cursor = seg["end"]

    if video_duration - cursor > 0.5:
        result.append({
            "start":  round(cursor, 3),
            "end":    round(video_duration, 3),
            "label":  "main_content",
            "reason": "Content between non-content segments",
        })

    return result
