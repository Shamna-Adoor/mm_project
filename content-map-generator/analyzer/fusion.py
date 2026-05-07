"""Rule-based fusion of audio + visual signals into a final segment list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import yaml

from analyzer._logging import get_logger

log = get_logger(__name__)

SegmentLabel = Literal["intro", "main_content", "sponsor", "outro", "dead_air"]
RULES_PATH   = Path(__file__).parent / "fusion_rules.yaml"
SKIP_LABELS: frozenset[str] = frozenset({"intro", "sponsor", "outro", "dead_air"})


class Segment(TypedDict):
    start:          float
    end:            float
    label:          str
    confidence:     float
    skip_recommended: bool
    reason:         str
    signals_used:   list[str]


# ── Public API ────────────────────────────────────────────────────────────────

def classify_segments(
    audio_signals:  dict,
    visual_signals: dict,
    video_duration: float,
    *,
    rules_path: str | Path = RULES_PATH,
) -> list[Segment]:
    """Fuse raw signals into labelled, non-overlapping segments.

    Algorithm
    ---------
    1. Build candidate boundaries from scene changes + silence.
    2. Score each candidate segment against every label using weighted rules.
    3. Merge adjacent same-label segments within a configurable gap.
    4. Ensure the full duration [0, video_duration] is covered.
    """
    rules = _load_rules(Path(rules_path))

    boundaries = _build_boundaries(audio_signals, visual_signals, rules)
    boundaries = sorted({0.0, video_duration} | set(boundaries))

    raw_segments: list[Segment] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 0.5:          # skip slivers
            continue
        label, conf, sigs = _score_segment(
            start, end, audio_signals, visual_signals, video_duration, rules
        )
        raw_segments.append({
            "start":            round(start, 3),
            "end":              round(end,   3),
            "label":            label,
            "confidence":       round(conf,  3),
            "skip_recommended": label in SKIP_LABELS,
            "reason":           _build_reason(label, sigs),
            "signals_used":     sigs,
        })

    merged = _merge_adjacent(raw_segments, rules["merging"]["max_gap_seconds"])
    log.info("Fusion produced %d segment(s)", len(merged))
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rules(rules_path: Path) -> dict:
    with open(rules_path) as f:
        return yaml.safe_load(f)


def _build_boundaries(audio: dict, visual: dict, rules: dict) -> list[float]:
    """Collect candidate cut points from all signal sources."""
    pts: list[float] = []

    sc_weight = rules["boundaries"]["scene_change_weight"]
    for sc in visual.get("scene_changes", []):
        if sc["confidence"] >= sc_weight:
            pts.append(sc["timestamp"])

    min_sil = rules["boundaries"]["min_silence_duration"]
    for sil in audio.get("silence_intervals", []):
        dur = sil["end"] - sil["start"]
        if dur >= min_sil:
            pts.append(sil["start"])
            pts.append(sil["end"])

    for hit in audio.get("sponsor_phrases", []):
        pts.append(hit["start"])

    for bp in audio.get("boilerplate_phrases", []):
        pts.append(bp["start"])

    # Music start/end are strong intro/outro boundaries
    for mi in audio.get("music_intervals", []):
        pts.append(mi["start"])
        pts.append(mi["end"])

    for ocr in visual.get("ocr_detections", []):
        pts.append(ocr["timestamp"])

    return pts


def _score_segment(
    start: float,
    end: float,
    audio: dict,
    visual: dict,
    duration: float,
    rules: dict,
) -> tuple[str, float, list[str]]:
    """Return (label, confidence, signals_used) for a candidate segment."""
    scores: dict[str, float] = {
        "intro":        0.0,
        "sponsor":      0.0,
        "outro":        0.0,
        "dead_air":     0.0,
        "main_content": rules["scoring"]["main_content"]["default_confidence"],
    }
    sigs_fired: dict[str, list[str]] = {k: [] for k in scores}

    r = rules["scoring"]

    # ── intro ────────────────────────────────────────────────────────────────
    if start < r["intro"]["position_window"]:
        scores["intro"] += r["intro"]["position_weight"]
        sigs_fired["intro"].append("position_near_start")

    for mi in audio.get("music_intervals", []):
        if _overlap(start, end, mi["start"], mi["end"]):
            scores["intro"]  += r["intro"]["music_weight"]
            scores["outro"]  += r["outro"]["music_weight"]
            sigs_fired["intro"].append("music_intervals")
            sigs_fired["outro"].append("music_intervals")
            break

    for bp in audio.get("boilerplate_phrases", []):
        if start <= bp["start"] < end:
            if bp["type"] == "intro":
                scores["intro"] += r["intro"]["boilerplate_intro_weight"]
                sigs_fired["intro"].append("boilerplate_phrases")
            elif bp["type"] == "outro":
                scores["outro"] += r["outro"]["boilerplate_outro_weight"]
                sigs_fired["outro"].append("boilerplate_phrases")

    # ── outro ────────────────────────────────────────────────────────────────
    if end > duration - r["outro"]["position_window"]:
        scores["outro"] += r["outro"]["position_weight"]
        sigs_fired["outro"].append("position_near_end")

    for ocr in visual.get("ocr_detections", []):
        if start <= ocr["timestamp"] < end:
            txt = ocr["text"].lower()
            if "subscribe" in txt or "bell" in txt:
                scores["outro"] += r["outro"]["ocr_subscribe_weight"]
                sigs_fired["outro"].append("ocr_detections")

    # ── sponsor ──────────────────────────────────────────────────────────────
    for sp in audio.get("sponsor_phrases", []):
        if start <= sp["start"] < end:
            scores["sponsor"] += r["sponsor"]["phrase_match_weight"]
            sigs_fired["sponsor"].append("sponsor_phrases")
            break

    seg_dur = end - start
    if r["sponsor"]["min_duration"] <= seg_dur <= r["sponsor"]["max_duration"]:
        scores["sponsor"] += r["sponsor"]["duration_in_range_weight"]
        sigs_fired["sponsor"].append("duration_in_range")

    for ocr in visual.get("ocr_detections", []):
        if start <= ocr["timestamp"] < end:
            txt = ocr["text"].lower()
            if any(kw in txt for kw in (".com", ".io", "http", "code", "% off")):
                scores["sponsor"] += r["sponsor"]["ocr_url_weight"]
                sigs_fired["sponsor"].append("ocr_detections")
                break

    # ── dead_air ─────────────────────────────────────────────────────────────
    for sil in audio.get("silence_intervals", []):
        if _overlap(start, end, sil["start"], sil["end"]):
            scores["dead_air"] += r["dead_air"]["silence_weight"]
            sigs_fired["dead_air"].append("silence_intervals")
            break

    for si in visual.get("static_intervals", []):
        if _overlap(start, end, si["start"], si["end"]):
            scores["dead_air"] += r["dead_air"]["static_weight"]
            sigs_fired["dead_air"].append("static_intervals")
            break

    if scores["dead_air"] < r["dead_air"]["min_combined_score"]:
        scores["dead_air"] = 0.0

    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]
    return best_label, min(1.0, best_score), sigs_fired[best_label]


def _overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
    return s1 < e2 and s2 < e1


def _merge_adjacent(segments: list[Segment], max_gap: float) -> list[Segment]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap  = seg["start"] - prev["end"]
        if seg["label"] == prev["label"] and gap <= max_gap:
            prev["end"]          = seg["end"]
            prev["confidence"]   = round((prev["confidence"] + seg["confidence"]) / 2, 3)
            prev["signals_used"] = list(set(prev["signals_used"]) | set(seg["signals_used"]))
            prev["reason"]       = _build_reason(prev["label"], prev["signals_used"])
        else:
            merged.append(dict(seg))
    return merged  # type: ignore[return-value]


def _build_reason(label: str, signals: list[str]) -> str:
    if not signals:
        return "Default classification"
    pretty = {
        "position_near_start":  "near video start",
        "position_near_end":    "near video end",
        "music_intervals":      "music detected",
        "boilerplate_phrases":  "boilerplate speech detected",
        "sponsor_phrases":      "sponsor phrase detected",
        "silence_intervals":    "silence detected",
        "static_intervals":     "static frame detected",
        "ocr_detections":       "overlay text detected",
        "duration_in_range":    "duration matches sponsor window",
    }
    parts = [pretty.get(s, s) for s in dict.fromkeys(signals)]  # dedupe + order
    return f"{label.replace('_', ' ').title()}: " + "; ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse signal JSON into segments.")
    parser.add_argument("signals",   help="Path to signal JSON")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--rules",   default=str(RULES_PATH))
    args = parser.parse_args()

    with open(args.signals) as f:
        data = json.load(f)

    segs = classify_segments(data["audio_signals"], data["visual_signals"], args.duration, rules_path=args.rules)
    print(json.dumps(segs, indent=2))
