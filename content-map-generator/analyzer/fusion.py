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
    novelty_regions: list[dict] | None = None,
) -> list[Segment]:
    """Fuse raw signals into labelled, non-overlapping segments.

    Algorithm
    ---------
    1. Build candidate boundaries from scene changes + silence + novelty regions.
    2. Score each candidate segment against every label using weighted rules.
    3. Merge adjacent same-label segments within a configurable gap.
    4. Ensure the full duration [0, video_duration] is covered.
    """
    rules = _load_rules(Path(rules_path))
    novelty_regions = novelty_regions or []

    boundaries = _build_boundaries(audio_signals, visual_signals, rules)
    # Novelty region boundaries are STRONG cut points — they mark the start
    # and end of a region whose joint signature differs from the rest of the
    # video, so they should always become candidate boundary points.
    for r in novelty_regions:
        try:
            boundaries.append(float(r["start"]))
            boundaries.append(float(r["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    boundaries = sorted({0.0, video_duration} | set(boundaries))

    raw_segments: list[Segment] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 0.5:          # skip slivers
            continue
        label, conf, sigs = _score_segment(
            start, end, audio_signals, visual_signals, video_duration, rules,
            novelty_regions=novelty_regions,
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

    # Priority 2 safety net: if the heuristic detectors over-flagged
    # non-content (e.g., on rapidly edited animation), demote the
    # lowest-confidence non-content segments back to main_content so we
    # never hide more of the video than the cap allows.
    merged = _enforce_non_content_cap(merged, video_duration, rules)

    log.info("Fusion produced %d segment(s)", len(merged))
    return merged


def _enforce_non_content_cap(
    segments: list[Segment],
    video_duration: float,
    rules: dict,
) -> list[Segment]:
    """Demote excess non-content back to main_content based on a global cap.

    Uses the content-type profile to pick the cap percentage. For
    talking-head videos this is generous (40%) and rarely triggers; for
    fast-paced animation it's tighter (25%). When triggered, the
    LOWEST-confidence non-content segments are flipped to main_content first.
    """
    if not segments or video_duration <= 0:
        return segments

    # Lazy-import the helpers to avoid circular imports at module load time.
    from analyzer.audio.llm_classify import classify_content_type, get_profile_for

    visual_signals = {
        "scene_changes":    [],   # not directly available here; rule-based
        "static_intervals": [],   # path doesn't carry these into fusion
    }
    # We don't have raw signals at this layer, so fall back to the segment
    # density itself: if non-content already dominates, default to a tight
    # animation profile; otherwise use the talking-head defaults.
    nc_seconds = sum(s["end"] - s["start"] for s in segments if s["label"] in SKIP_LABELS)
    nc_ratio   = nc_seconds / video_duration
    profile    = get_profile_for("fast_paced" if nc_ratio > 0.5 else "talking_head")
    cap_pct    = float(profile["max_non_content_pct"])

    if nc_ratio <= cap_pct:
        return segments

    # Sort non-content segments by confidence ascending — drop weakest first.
    nc_indices = [i for i, s in enumerate(segments) if s["label"] in SKIP_LABELS]
    nc_indices.sort(key=lambda i: segments[i].get("confidence", 0.0))

    cap_seconds = video_duration * cap_pct
    over = nc_seconds - cap_seconds
    flipped = 0
    for i in nc_indices:
        if over <= 0:
            break
        seg = segments[i]
        dur = seg["end"] - seg["start"]
        log.info(
            "Non-content cap engaged: demoting %s [%.1f-%.1f] (conf=%.2f) → main_content",
            seg["label"], seg["start"], seg["end"], seg.get("confidence", 0.0),
        )
        segments[i] = {
            **seg,
            "label":            "main_content",
            "skip_recommended": False,
            "reason":           "Demoted by non-content cap",
        }
        over -= dur
        flipped += 1

    if flipped:
        log.info("Demoted %d non-content segment(s) to enforce %.0f%% cap", flipped, cap_pct * 100)
    return segments


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

    # Treat scene-burst start/end as strong boundary candidates. Threshold
    # adapts to the video's overall cut density so animated content doesn't
    # generate spurious boundaries everywhere.
    burst_cfg = rules.get("scene_burst", {})
    if burst_cfg and visual.get("scene_changes"):
        from analyzer.audio.llm_classify import detect_scene_bursts
        for b in detect_scene_bursts(
            visual["scene_changes"],
            window_seconds=float(burst_cfg.get("window_seconds", 30.0)),
            min_cuts_in_window=int(burst_cfg.get("min_cuts_in_window", 4)),
            pad_before=float(burst_cfg.get("pad_before", 1.5)),
            pad_after=float(burst_cfg.get("pad_after", 3.0)),
            adaptive_baseline=True,
        ):
            pts.append(b["start"])
            pts.append(b["end"])

    return pts


def _score_segment(
    start: float,
    end: float,
    audio: dict,
    visual: dict,
    duration: float,
    rules: dict,
    *,
    novelty_regions: list[dict] | None = None,
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

    # ── Scene-change burst: strong signal of an inserted video ad ─────────────
    # Threshold and weight both scale with content type. Animation/fast-paced
    # videos use a higher cuts-per-window floor (so normal editing doesn't
    # cross the bar) AND a downscaled weight (so any burst that does fire
    # contributes less, since it's less diagnostic in those contexts).
    burst_cfg = rules.get("scene_burst", {})
    burst_weight_scale = 1.0
    gap_weight_scale   = 1.0
    if burst_cfg and visual.get("scene_changes"):
        from analyzer.audio.llm_classify import (
            classify_content_type,
            detect_scene_bursts,
            get_profile_for,
        )
        ct = classify_content_type(
            visual.get("scene_changes", []),
            visual.get("static_intervals", []),
            duration,
        )
        prof = get_profile_for(ct["type"])
        burst_weight_scale = float(prof.get("scene_burst_weight", 1.0))
        gap_weight_scale   = float(prof.get("long_gap_weight",    1.0))
        bursts = detect_scene_bursts(
            visual["scene_changes"],
            window_seconds=float(burst_cfg.get("window_seconds", 30.0)),
            min_cuts_in_window=int(prof["burst_min_cuts"]),
            video_duration=duration,
            pad_before=float(burst_cfg.get("pad_before", 1.5)),
            pad_after=float(burst_cfg.get("pad_after", 3.0)),
            baseline_multiplier=float(prof["burst_baseline_mult"]),
            floor_min_cuts=int(prof["burst_min_cuts"]),
        )
        for b in bursts:
            if _overlap(start, end, b["start"], b["end"]):
                scores["sponsor"] += (
                    r["sponsor"].get("scene_burst_weight", 1.0) * burst_weight_scale
                )
                sigs_fired["sponsor"].append("scene_change_burst")
                break

    # ── Long mid-video speech gap reinforces sponsor scoring ──────────────────
    # Weight is content-type scaled so animation action scenes (which have
    # natural long speech gaps) don't trip the sponsor classifier.
    gap_cfg = rules.get("speech_gap", {})
    if gap_cfg and audio.get("transcript"):
        from analyzer.audio.llm_classify import detect_long_speech_gaps
        gaps = detect_long_speech_gaps(
            audio["transcript"],
            min_gap_seconds=float(gap_cfg.get("min_gap_seconds", 20.0)),
            video_duration=duration,
            boundary_pct=float(gap_cfg.get("boundary_pct", 0.10)),
        )
        for g in gaps:
            if _overlap(start, end, g["start"], g["end"]):
                scores["sponsor"] += (
                    r["sponsor"].get("long_gap_weight", 0.4) * gap_weight_scale
                )
                sigs_fired["sponsor"].append("long_speech_gap")
                break

    # ── NEW: coherence signals (semantic + stylistic) ──────────────────────────
    # Each fires independently and contributes to sponsor confidence. Weights
    # are deliberately moderate so a single coherence signal alone never
    # dominates — at least two have to agree (or one + a structural signal).
    for cc in audio.get("commercial_clusters", []):
        if _overlap(start, end, cc["start"], cc["end"]):
            scores["sponsor"] += r["sponsor"].get("commercial_cluster_weight", 1.0)
            sigs_fired["sponsor"].append("commercial_cluster")
            break

    for aa in audio.get("audio_anomalies", []):
        if _overlap(start, end, aa["start"], aa["end"]):
            scores["sponsor"] += r["sponsor"].get("audio_anomaly_weight", 0.6)
            sigs_fired["sponsor"].append("audio_style_shift")
            break

    for va in visual.get("visual_anomalies", []):
        if _overlap(start, end, va["start"], va["end"]):
            scores["sponsor"] += r["sponsor"].get("visual_anomaly_weight", 0.6)
            sigs_fired["sponsor"].append("visual_style_shift")
            break

    # ── Joint novelty: STRONG sponsor signal (template-free) ──────────────────
    # A region whose joint audio+visual+structural signature is far from the
    # video's own baseline is, by construction, unlike the rest of the video.
    # That is the universal property of an inserted ad; weight it
    # accordingly. The weight is intentionally high (≥ phrase match) so a
    # confirmed novelty region can outscore main_content on its own.
    novelty_regions = novelty_regions or []
    for nr in novelty_regions:
        try:
            ns = float(nr["start"]); ne = float(nr["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if _overlap(start, end, ns, ne):
            base_w  = r["sponsor"].get("novelty_weight", 1.3)
            score   = float(nr.get("score", 5.0))
            # Higher score → larger contribution, capped to keep the scorer stable.
            scaled  = base_w * (1.0 + min(0.5, max(0.0, (score - 5.0) / 10.0)))
            scores["sponsor"] += scaled
            sigs_fired["sponsor"].append("joint_novelty")
            break

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
