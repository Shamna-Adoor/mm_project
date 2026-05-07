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
- "sponsor": Advertisement or paid promotion embedded in the video. This includes BOTH:
    (a) Host-read sponsorships: "this video is sponsored by", "I want to thank our sponsor",
        "use code [X] for [discount]", "check out [product] in the description", affiliate plugs,
        even subtle ones like "the app I've been using lately is..."
    (b) Inserted video advertisements: a fully spliced-in ad clip whose audio is unrelated to
        the surrounding content. These often appear as a SHORT off-topic line containing a
        BRAND, GAME, APP, SERVICE, or PRODUCT NAME mid-conversation, with no logical
        connection to what came before or after. Examples that MUST be flagged as sponsor
        even when brief: "Clash of Clans", "Raid Shadow Legends", "NordVPN", "Squarespace",
        "Brilliant", "Honey", "Audible", "ExpressVPN", "Skillshare", "BetterHelp",
        "HelloFresh", "MasterClass", "any [game name] download for free / play now / join the army"
        type lines. ALWAYS flag these as sponsor regardless of length.
- "dead_air": Silence, blank screen, technical glitches, filler with no informational value, or a noticeable pause/gap between topics where the transcript has no text or only ambient sound.

TRANSCRIPT (format: [start_sec - end_sec] text):
{transcript_text}

STRICT RULES:
1. "intro" ONLY in the FIRST 20% of the video — not mid-video.
2. "outro" ONLY in the LAST 20% of the video — not mid-video transitions.
3. "sponsor": minimum 5 seconds. Sponsor can appear ANYWHERE in the video.
4. "sponsor" must be flagged whenever a transcript line contains a known commercial brand,
   game, app, service, or product mentioned out of context with the surrounding topic —
   even if the line is just a few seconds long. Bracket the sponsor segment to cover the
   full ad break (use any nearby [GAP] markers as boundaries).
5. "dead_air": mark any gap where there is no transcript text for 3+ seconds, OR a noticeable silent/filler pause between topics, BUT NOT when the gap is sandwiched between off-topic ad-like lines (those gaps are part of a sponsor ad break).
6. Transitional FILLER ("um", "uh", "you know", extended silence) is dead_air. Brief connectors ("now let's look at") alone are NOT.
7. Merging: combine adjacent same-type segments into one.
8. For genuine content, keep it as content. For clear non-content, mark it.

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
    *,
    scene_changes:       list[dict] | None = None,
    transcript:          list[dict] | None = None,
    static_intervals:    list[dict] | None = None,
    commercial_clusters: list[dict] | None = None,
    audio_anomalies:     list[dict] | None = None,
    visual_anomalies:    list[dict] | None = None,
    novelty_regions:     list[dict] | None = None,
) -> list[Segment]:
    """Convert LLM segments → final Segment list, overlaying audio + visual signals.

    Strategy
    --------
    1. Classify the video's content type from existing visual signals (Priority 5).
       Talking-head videos use the original conservative thresholds; rapidly
       edited content (animation, gaming) gets stricter detection thresholds.
    2. Build "forced" non-content intervals from audio + visual signals:
       - Silence  >= 1.5 s  → dead_air
       - Music    >= 5 s near video start/end → intro / outro
       - Scene-change BURSTS (adaptive threshold, Priority 1) → sponsor
       - Long speech gaps (>=20s) co-located with scene bursts → sponsor
    3. Build EXTRA sponsor candidates from coherence signals (semantic +
       stylistic layer). A region wins this round when the combination of
       commercial-language clusters, audio style anomalies and visual
       coherence anomalies meets a confidence threshold. This catches
       calm narrative ads that have no rapid scene cuts.
    4. Apply per-label and total non-content caps (Priority 2 safety net).
    5. For every surviving forced interval that falls inside a main_content
       LLM segment, *split* that segment and insert the forced label.
       (Non-content LLM labels like sponsor/intro/outro are never overwritten.)
    6. Validate LLM dead_air against silence data; revert to main_content if no
       actual silence detected there.
    7. Convert the final list to the Segment TypedDict format.
    """
    silence_intervals   = silence_intervals   or []
    music_intervals     = music_intervals     or []
    scene_changes       = scene_changes       or []
    transcript          = transcript          or []
    static_intervals    = static_intervals    or []
    commercial_clusters = commercial_clusters or []
    audio_anomalies     = audio_anomalies     or []
    visual_anomalies    = visual_anomalies    or []
    novelty_regions     = novelty_regions     or []

    # Priority 5 — content-type classification + threshold profile
    content_info = classify_content_type(scene_changes, static_intervals, video_duration)
    profile      = get_profile_for(content_info["type"])
    log.info(
        "Content type: %s (%.1f cuts/min, %.0f%% static) — using profile %s",
        content_info["type"], content_info["cuts_per_min"],
        content_info["static_ratio"] * 100, profile,
    )

    intro_cutoff = min(video_duration * 0.20, 300) if video_duration else 300
    outro_cutoff = max(video_duration - min(video_duration * 0.20, 300), 0) if video_duration else 0

    # ── Step 1: collect forced intervals from audio + visual ──────────────────
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
            forced.append({
                "start":  mi["start"], "end": mi["end"],
                "label":  "dead_air",
                "reason": f"Music break detected mid-video ({dur:.0f}s)",
                "conf":   0.72,
            })

    # ── Step 1b: combined sponsor detection ───────────────────────────────────
    # Two signals jointly identify inserted video ads:
    #   (1) Scene-change BURST: ≥N rapid cuts in a 30s window — N is now
    #       adaptive based on the video's overall cut density (Priority 1)
    #   (2) Long speech gap + ≥2 scene changes inside that gap
    # Each detected region is merged with overlapping detections from the
    # other signal, then merged with adjacent regions.
    bursts = detect_scene_bursts(
        scene_changes,
        video_duration=video_duration,
        min_cuts_in_window=int(profile["burst_min_cuts"]),
        baseline_multiplier=float(profile["burst_baseline_mult"]),
        floor_min_cuts=int(profile["burst_min_cuts"]),
    )
    gaps   = detect_long_speech_gaps(transcript, video_duration=video_duration)

    candidate_regions: list[dict] = list(bursts)

    # Add gaps that contain enough scene cuts as additional sponsor candidates.
    # The minimum is content-type aware: in animation/fast-paced shows, action
    # sequences naturally produce long speech gaps with multiple cuts, so we
    # require more cuts before treating the gap as ad evidence. For talking-
    # head content (the original behavior) the threshold stays at 2.
    cut_times = [float(sc["timestamp"]) for sc in scene_changes if "timestamp" in sc]
    gap_cut_min = int(profile.get("gap_cut_min", 2))
    for g in gaps:
        cuts_inside = sum(1 for t in cut_times if g["start"] <= t <= g["end"])
        if cuts_inside >= gap_cut_min:
            candidate_regions.append({
                "start": g["start"], "end": g["end"], "cuts": cuts_inside,
            })

    # Also expand any burst to absorb adjacent/overlapping gaps (catches the
    # silent tail of ads where audio drops out before the visual cuts end).
    for c in candidate_regions:
        for g in gaps:
            if _intervals_overlap(c["start"], c["end"], g["start"], g["end"]):
                c["start"] = min(c["start"], g["start"])
                c["end"]   = max(c["end"],   g["end"])

    # Merge overlapping / adjacent candidate regions (within 10s).
    # Ads often have a brief audio resume between visual cut clusters; a 10s
    # merge gap stitches the full ad break back together.
    candidate_regions.sort(key=lambda r: r["start"])
    merged_regions: list[dict] = []
    for c in candidate_regions:
        if merged_regions and c["start"] - merged_regions[-1]["end"] <= 10.0:
            prev = merged_regions[-1]
            prev["end"]  = max(prev["end"], c["end"])
            prev["cuts"] = prev.get("cuts", 0) + c.get("cuts", 0)
        else:
            merged_regions.append(dict(c))

    for r in merged_regions:
        # Don't flag regions that live entirely in the intro window (opening
        # montage is expected for many videos).
        if r["start"] < intro_cutoff and r["end"] < intro_cutoff:
            continue
        forced.append({
            "start":  r["start"], "end": r["end"],
            "label":  "sponsor",
            "reason": f"{r.get('cuts', 0)} scene cuts + speech-gap activity in {r['end']-r['start']:.0f}s — likely inserted ad",
            "conf":   0.88,
        })

    # ── Step 1b-extra: coherence-based sponsor candidates ─────────────────────
    # Three NEW signal sources are collapsed into candidate regions and
    # then voted on. Each region needs >= 2 votes to be promoted to a
    # forced sponsor interval — this avoids false positives from any
    # single noisy detector.
    #
    # Vote sources:
    #   • commercial language cluster   (semantic: brand/url/price density)
    #   • audio style anomaly           (stylistic: spectral shift)
    #   • visual color/style anomaly    (stylistic: HSV histogram shift)
    #
    # A region also wins if (a) a single source is highly confident, or
    # (b) the region sits inside a zone where Whisper is unreliable AND a
    # stylistic signal alone is strong. The second case prevents missing
    # ads in genres where transcripts can't be trusted (animation, music,
    # noise-heavy content) without inflating false positives elsewhere.
    low_conf_zones = _low_confidence_zones(transcript, video_duration)
    if low_conf_zones:
        log.info(
            "Low-confidence transcript zones: %d region(s) (%.0fs total)",
            len(low_conf_zones),
            sum(e - s for s, e in low_conf_zones),
        )
    coherence_regions = _collect_coherence_candidates(
        commercial_clusters=commercial_clusters,
        audio_anomalies=audio_anomalies,
        visual_anomalies=visual_anomalies,
        intro_cutoff=intro_cutoff,
        outro_cutoff=outro_cutoff,
        existing_forced=forced,
        low_confidence_zones=low_conf_zones,
        coherence_solo_score=float(profile.get("coherence_solo_score", 2.5)),
    )
    for r in coherence_regions:
        forced.append({
            "start":  r["start"], "end": r["end"],
            "label":  "sponsor",
            "reason": r["reason"],
            "conf":   r["conf"],
        })

    # ── Step 1b-novelty: joint-novelty regions (template-free detector) ───────
    # These are regions whose joint audio+visual+structural signature differs
    # from the per-video baseline. They are the PRIMARY ad signal because
    # they are template-free — they don't depend on commercial keywords,
    # specific brand names, or hand-tuned thresholds. A region that is
    # JOINTLY anomalous (audio AND visual both deviate) is the universal
    # property of every inserted ad.
    #
    # Because joint novelty is principled and corroborated across modalities,
    # we give it confidence ≥ 0.92 — enough to beat dead_air (conf 0.90) in
    # the cap layer's confidence sort, ensuring true ads aren't dropped in
    # favour of speculative silence intervals when budgets are tight.
    for nr in novelty_regions:
        try:
            n_start = float(nr["start"]); n_end = float(nr["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if n_end <= n_start:
            continue
        # Map score → confidence in [0.92, 0.98]. A score at the threshold
        # (~2.0 for animation) gives 0.92; very strong novelty (score ≥ 4)
        # saturates near 0.98. Stays above dead_air's 0.90 by construction.
        score = float(nr.get("score", 2.5))
        conf  = max(0.92, min(0.98, 0.88 + 0.02 * score))
        axes  = nr.get("dominant_axes", []) or []
        axes_str = ", ".join(axes[:3]) if axes else "joint"
        # Absorb novelty into any existing forced sponsor that already covers
        # this region (e.g., a scene-burst burst caught the same ad). This
        # avoids emitting duplicate overlapping sponsor blocks.
        absorbed = False
        for f in forced:
            if f.get("label") == "sponsor" and _intervals_overlap(
                n_start, n_end, f["start"], f["end"],
            ):
                f["start"]  = min(f["start"], n_start)
                f["end"]    = max(f["end"],   n_end)
                f["conf"]   = max(f.get("conf", 0.0), conf)
                f["reason"] = (
                    f.get("reason", "")
                    + f" | reinforced by novelty (score={score:.2f}; {axes_str})"
                )
                absorbed = True
                break
        if absorbed:
            continue
        forced.append({
            "start":  n_start, "end": n_end,
            "label":  "sponsor",
            "reason": f"joint-novelty region (score={score:.2f}; {axes_str})",
            "conf":   conf,
        })

    # ── Step 1c: cap non-content (Priority 2 safety net) ──────────────────────
    # If the heuristic detectors over-fire on rapidly edited content, we
    # never let total forced non-content exceed a configurable fraction of
    # the video. Per-label caps protect against any single signal dominating.
    forced = cap_non_content_intervals(
        forced,
        video_duration,
        max_total_pct=float(profile["max_non_content_pct"]),
        max_per_label_pct={
            "sponsor":  float(profile["max_sponsor_pct"]),
            "dead_air": 0.50,   # silence is reliable, allow a generous cap
            "intro":    0.10,
            "outro":    0.10,
        },
    )

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

    log.info(
        "to_output_segments: %d total (%d skip) from %d LLM + %d silence + %d music forced",
        len(result),
        sum(1 for s in result if s["skip_recommended"]),
        len(llm_segs),
        sum(1 for f in forced if f["label"] == "dead_air"),
        sum(1 for f in forced if f["label"] in ("intro","outro")),
    )
    return result


# ── Content-type classifier (rule-based) ──────────────────────────────────────

# Coarse buckets used to scale detection thresholds. Choosing the right bucket
# lets a single pipeline behave correctly for both calm interviews and rapid
# animated shows without over- or under-detecting.
CONTENT_TYPES = ("talking_head", "mixed", "animation", "fast_paced")


def classify_content_type(
    scene_changes: list[dict] | None,
    static_intervals: list[dict] | None,
    video_duration: float,
) -> dict:
    """Bucket the video by visual style using existing signals.

    Returns
    -------
    dict with keys:
      - "type":           one of CONTENT_TYPES
      - "cuts_per_min":   scene cuts per minute across the whole video
      - "static_ratio":   fraction of video covered by static-frame intervals

    The result is purely informational + drives threshold scaling. It never
    triggers a label by itself, so misclassification is non-fatal.
    """
    scene_changes    = scene_changes or []
    static_intervals = static_intervals or []
    if video_duration <= 30.0:
        return {"type": "mixed", "cuts_per_min": 0.0, "static_ratio": 0.0}

    cuts_per_min = len(scene_changes) / (video_duration / 60.0)
    static_secs  = sum(max(0.0, s["end"] - s["start"]) for s in static_intervals)
    static_ratio = static_secs / video_duration if video_duration > 0 else 0.0

    if cuts_per_min < 4 and static_ratio > 0.30:
        kind = "talking_head"
    elif cuts_per_min >= 25:
        kind = "fast_paced"   # very rapid editing (animation, music videos, gaming highlights)
    elif cuts_per_min >= 12:
        kind = "animation"
    elif cuts_per_min >= 6:
        kind = "mixed"
    else:
        kind = "talking_head"

    return {
        "type":         kind,
        "cuts_per_min": round(cuts_per_min, 2),
        "static_ratio": round(static_ratio, 3),
    }


# Threshold + cap multipliers per content type. Each entry is conservative —
# values were chosen so the talking_head defaults exactly match prior behavior.
#
# Profile keys
# ------------
# burst_min_cuts        Minimum scene cuts inside ``window_seconds`` for a
#                       region to count as a scene-burst. Higher for
#                       intrinsically high-cut content (animation, music
#                       video) so normal editing isn't read as an ad.
# burst_baseline_mult   Adaptive threshold = baseline_density × this.
# max_non_content_pct   Hard cap on total non-content. Protects against
#                       runaway over-detection.
# max_sponsor_pct       Hard cap on sponsor specifically.
# gap_cut_min           A long mid-video speech gap is only treated as
#                       sponsor evidence if it ALSO contains at least this
#                       many scene cuts. Prevents action sequences in
#                       animation from being misread as ad breaks.
# scene_burst_weight    Scoring weight for the scene-burst signal in the
#                       rule-based fusion path (downscaled for content where
#                       bursts are normal).
# long_gap_weight       Same idea for long-speech-gap evidence.
# coherence_solo_score  Audio-only or visual-only coherence score required
#                       to promote a region in low-confidence transcript
#                       zones (one strong stylistic signal alone is enough
#                       when Whisper can't tell us what's happening).
_TYPE_PROFILES: dict[str, dict] = {
    "talking_head": {
        "burst_min_cuts":      4,        # absolute floor (matches legacy)
        "burst_baseline_mult": 3.0,      # adaptive: median + 3*std
        "max_non_content_pct": 0.40,
        "max_sponsor_pct":     0.30,
        "gap_cut_min":         2,
        "scene_burst_weight":  1.0,
        "long_gap_weight":     1.0,
        "coherence_solo_score": 2.5,
        # Joint-novelty threshold k. Higher = stricter. Talking-head
        # content has stable audio so even moderate visual+audio shifts
        # are noisy. Use a stricter threshold to reduce false positives
        # from chapter transitions, slide changes, demo segments.
        "novelty_sigmas":       3.0,
    },
    "mixed": {
        "burst_min_cuts":      6,
        "burst_baseline_mult": 2.5,
        "max_non_content_pct": 0.35,
        "max_sponsor_pct":     0.25,
        "gap_cut_min":         3,
        "scene_burst_weight":  0.9,
        "long_gap_weight":     0.85,
        "coherence_solo_score": 2.5,
        "novelty_sigmas":       2.75,
    },
    "animation": {
        "burst_min_cuts":      10,
        "burst_baseline_mult": 2.0,
        "max_non_content_pct": 0.30,
        "max_sponsor_pct":     0.20,
        "gap_cut_min":         4,
        "scene_burst_weight":  0.8,
        "long_gap_weight":     0.55,
        "coherence_solo_score": 2.2,    # transcripts unreliable → trust style
        # Animation has high natural variance; ads stand out clearly so
        # a moderate threshold catches them with few false positives.
        "novelty_sigmas":       2.5,
    },
    "fast_paced": {
        "burst_min_cuts":      18,
        "burst_baseline_mult": 1.8,
        "max_non_content_pct": 0.25,
        "max_sponsor_pct":     0.15,
        "gap_cut_min":         5,
        "scene_burst_weight":  0.75,
        "long_gap_weight":     0.45,
        "coherence_solo_score": 2.2,
        "novelty_sigmas":       2.5,
    },
}


def get_profile_for(content_type: str) -> dict:
    """Look up the threshold profile for a content type (with safe fallback)."""
    return dict(_TYPE_PROFILES.get(content_type, _TYPE_PROFILES["talking_head"]))


# ── Hard cap on non-content (Priority 2 safety net) ───────────────────────────

def cap_non_content_intervals(
    forced_intervals: list[dict],
    video_duration:   float,
    *,
    max_total_pct:    float = 0.40,
    max_per_label_pct: dict[str, float] | None = None,
) -> list[dict]:
    """Drop the lowest-confidence forced intervals when caps are exceeded.

    This is a safety net: if the heuristic detectors over-fire (e.g., on
    rapidly edited animation), we never let the predicted non-content exceed
    a configurable fraction of the video. The user always sees at least
    ``1 - max_total_pct`` of the runtime as content.

    The function never drops or modifies anything when the cap isn't reached;
    it is a no-op in the common case.
    """
    if not forced_intervals or video_duration <= 0:
        return list(forced_intervals)

    max_per_label_pct = max_per_label_pct or {}

    # Per-label caps first: prune the lowest-confidence intervals of each
    # label whose label budget is exceeded.
    by_label: dict[str, list[dict]] = {}
    for f in forced_intervals:
        by_label.setdefault(f["label"], []).append(f)

    kept_after_label_caps: list[dict] = []
    for label, intervals in by_label.items():
        cap_pct = max_per_label_pct.get(label)
        if cap_pct is None or cap_pct >= 1.0:
            kept_after_label_caps.extend(intervals)
            continue
        cap_seconds = video_duration * cap_pct
        sorted_iv = sorted(intervals, key=lambda f: f.get("conf", 0.5), reverse=True)
        accumulated = 0.0
        for f in sorted_iv:
            dur = max(0.0, f["end"] - f["start"])
            if accumulated + dur <= cap_seconds + 1e-6:
                kept_after_label_caps.append(f)
                accumulated += dur
            else:
                log.info(
                    "cap_non_content_intervals: dropping %s [%.1f-%.1f] "
                    "(label cap %.0f%% reached)",
                    label, f["start"], f["end"], cap_pct * 100,
                )

    # Global cap: across all labels, total non-content cannot exceed max_total_pct.
    if max_total_pct >= 1.0:
        return kept_after_label_caps

    cap_seconds = video_duration * max_total_pct
    sorted_iv = sorted(kept_after_label_caps, key=lambda f: f.get("conf", 0.5), reverse=True)
    final: list[dict] = []
    accumulated = 0.0
    for f in sorted_iv:
        dur = max(0.0, f["end"] - f["start"])
        if accumulated + dur <= cap_seconds + 1e-6:
            final.append(f)
            accumulated += dur
        else:
            log.info(
                "cap_non_content_intervals: dropping %s [%.1f-%.1f] "
                "(total cap %.0f%% reached)",
                f["label"], f["start"], f["end"], max_total_pct * 100,
            )

    if len(final) < len(forced_intervals):
        log.info(
            "Non-content cap engaged: kept %d/%d forced intervals (%.0fs of %.0fs duration)",
            len(final), len(forced_intervals), accumulated, video_duration,
        )
    return final


# ── Signal helpers (scene bursts, speech gaps) ────────────────────────────────

def detect_scene_bursts(
    scene_changes: list[dict],
    *,
    window_seconds: float = 30.0,
    min_cuts_in_window: int = 4,
    video_duration: float = 0.0,
    pad_before: float = 1.5,
    pad_after:  float = 3.0,
    adaptive_baseline:  bool = True,
    baseline_multiplier: float = 3.0,
    floor_min_cuts:     int  = 4,
) -> list[dict]:
    """Find regions with high scene-change density — strong signal of inserted ads.

    Talking-head interviews produce ~0–2 cuts per minute. Inserted video ads
    typically produce 5+ cuts in a 30s window because real ads have rapid
    montage editing.

    When ``adaptive_baseline`` is True (default), the threshold is auto-raised
    on high-cut-density videos so an animated cartoon's normal editing isn't
    misread as one giant ad. The threshold becomes::

        threshold = max(floor_min_cuts, baseline_multiplier × baseline_density)

    where ``baseline_density`` is the average number of cuts per
    ``window_seconds`` window across the whole video. For a calm interview
    this averages to ≈ 0–2, so the floor (``min_cuts_in_window``) wins and
    behavior is identical to the previous fixed-threshold detector.

    Returns a list of {"start", "end", "cuts"} bursts (already padded).
    """
    if not scene_changes or len(scene_changes) < min_cuts_in_window:
        return []

    cuts = sorted(float(sc["timestamp"]) for sc in scene_changes if "timestamp" in sc)

    # ── Adaptive threshold: raise the bar on high-cut-density content ──────
    effective_threshold = min_cuts_in_window
    if adaptive_baseline and video_duration > window_seconds and len(cuts) >= 8:
        baseline_density = len(cuts) / max(1.0, video_duration / window_seconds)
        adaptive = baseline_multiplier * baseline_density
        effective_threshold = max(floor_min_cuts, int(round(adaptive)))
        if effective_threshold > min_cuts_in_window:
            log.info(
                "Adaptive scene-burst threshold: %d cuts/window (baseline=%.2f, k=%.1f)",
                effective_threshold, baseline_density, baseline_multiplier,
            )

    # For every cut, count how many other cuts fall within the window centered
    # at that cut. If the count crosses the threshold, mark this cut as "hot".
    hot: list[float] = []
    half = window_seconds / 2.0
    for i, t in enumerate(cuts):
        lo = t - half
        hi = t + half
        nearby = sum(1 for c in cuts if lo <= c <= hi)
        if nearby >= effective_threshold:
            hot.append(t)

    if not hot:
        return []

    # Group hot cuts into contiguous bursts: gaps > window mean a new burst.
    bursts: list[dict] = []
    burst_start = hot[0]
    burst_end   = hot[0]
    burst_count = 1
    for t in hot[1:]:
        if t - burst_end <= window_seconds:
            burst_end = t
            burst_count += 1
        else:
            bursts.append({"start": burst_start, "end": burst_end, "cuts": burst_count})
            burst_start = t
            burst_end   = t
            burst_count = 1
    bursts.append({"start": burst_start, "end": burst_end, "cuts": burst_count})

    # Pad lightly to capture the actual ad boundaries (scenedetect tends to
    # land on first/last frame of the burst, which clips the edges).
    upper = video_duration if video_duration > 0 else float("inf")
    padded: list[dict] = []
    for b in bursts:
        padded.append({
            "start": max(0.0, b["start"] - pad_before),
            "end":   min(upper, b["end"] + pad_after),
            "cuts":  b["cuts"],
        })
    return padded


def detect_long_speech_gaps(
    transcript: list[dict],
    *,
    min_gap_seconds: float = 20.0,
    video_duration:  float = 0.0,
    boundary_pct:    float = 0.10,
) -> list[dict]:
    """Find long stretches with no transcribed speech in the *middle* of a video.

    A 20+ second gap in the middle of an active conversation is suspicious —
    real interview pauses (thinking time) rarely exceed ~10s without filler.
    Boundary regions are excluded since intro/outro/credit gaps are normal.
    """
    if not transcript or video_duration < 60:
        return []

    intro_cutoff = min(video_duration * boundary_pct, 60.0)
    outro_cutoff = max(video_duration - min(video_duration * boundary_pct, 60.0), 0.0)

    segments = sorted(
        (s for s in transcript if "start" in s and "end" in s),
        key=lambda s: float(s["start"]),
    )

    gaps: list[dict] = []
    for i in range(len(segments) - 1):
        gap_start = float(segments[i]["end"])
        gap_end   = float(segments[i + 1]["start"])
        gap_dur   = gap_end - gap_start
        if gap_dur < min_gap_seconds:
            continue
        # Skip gaps that fall entirely in the intro or outro zone
        if gap_end <= intro_cutoff:
            continue
        if gap_start >= outro_cutoff:
            continue
        gaps.append({"start": gap_start, "end": gap_end, "duration": gap_dur})
    return gaps


def _intervals_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
    return s1 < e2 and s2 < e1


# ── Low-confidence transcript zones ───────────────────────────────────────────
# When the transcriber can't reliably hear what's being said (typical for
# inserted ads with unfamiliar voices / stock music / heavy effects),
# Whisper either drops out entirely or hallucinates plausible-sounding text
# at low confidence. We don't want to penalise these regions — instead we
# want to TRUST the style-based detectors more there.

def _low_confidence_zones(
    transcript: list[dict],
    video_duration: float,
    *,
    avg_logprob_max:    float = -0.85,
    no_speech_prob_min: float = 0.50,
    silence_pad_seconds: float = 4.0,
    merge_gap_seconds:   float = 6.0,
) -> list[tuple[float, float]]:
    """Return time intervals where the transcript is unreliable.

    A region is "unreliable" if:
      • there is NO transcript segment covering it for at least
        ``silence_pad_seconds`` (Whisper missed it entirely), OR
      • a covering transcript segment has avg_logprob below
        ``avg_logprob_max`` (low recogniser confidence), OR
      • a covering transcript segment has no_speech_prob above
        ``no_speech_prob_min`` (Whisper itself thinks it's not speech).

    These thresholds are conservative — talking-head videos with clean
    audio essentially never produce zones, so existing behaviour is
    unchanged for that content type. The function is purely informational;
    callers decide what (if anything) to do with the zones.
    """
    if video_duration <= 0:
        return []

    zones: list[tuple[float, float]] = []

    # 1. Per-segment confidence flags.
    if transcript:
        for seg in transcript:
            try:
                s = float(seg["start"]); e = float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if e <= s:
                continue
            flagged = False
            if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                try:
                    if float(seg["avg_logprob"]) < avg_logprob_max:
                        flagged = True
                except (TypeError, ValueError):
                    pass
            if not flagged and "no_speech_prob" in seg and seg["no_speech_prob"] is not None:
                try:
                    if float(seg["no_speech_prob"]) > no_speech_prob_min:
                        flagged = True
                except (TypeError, ValueError):
                    pass
            if flagged:
                zones.append((s, e))

    # 2. Coverage gaps in the transcript timeline (Whisper produced nothing).
    if transcript:
        sorted_segs = sorted(
            ((float(s["start"]), float(s["end"])) for s in transcript
             if "start" in s and "end" in s),
            key=lambda x: x[0],
        )
        cursor = 0.0
        for s, e in sorted_segs:
            if s - cursor >= silence_pad_seconds:
                zones.append((cursor, s))
            cursor = max(cursor, e)
        if video_duration - cursor >= silence_pad_seconds:
            zones.append((cursor, video_duration))
    else:
        zones.append((0.0, video_duration))

    if not zones:
        return []

    # 3. Merge overlapping / nearby zones so they form clean intervals.
    zones.sort(key=lambda z: z[0])
    merged: list[list[float]] = [list(zones[0])]
    for s, e in zones[1:]:
        if s - merged[-1][1] <= merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def _zone_overlap_ratio(start: float, end: float, zones: list[tuple[float, float]]) -> float:
    """Return the fraction of [start, end] covered by any low-confidence zone."""
    if end <= start or not zones:
        return 0.0
    covered = 0.0
    for zs, ze in zones:
        ovl = max(0.0, min(end, ze) - max(start, zs))
        covered += ovl
    return min(1.0, covered / (end - start))


def _collect_coherence_candidates(
    *,
    commercial_clusters: list[dict],
    audio_anomalies:     list[dict],
    visual_anomalies:    list[dict],
    intro_cutoff:        float,
    outro_cutoff:        float,
    existing_forced:     list[dict],
    min_votes:           int   = 2,
    min_duration:        float = 5.0,
    merge_gap_seconds:   float = 8.0,
    base_conf_two:       float = 0.78,
    base_conf_three:     float = 0.86,
    high_score_solo:     float = 0.85,
    low_confidence_zones: list[tuple[float, float]] | None = None,
    coherence_solo_score: float = 2.5,
    low_conf_overlap_min: float = 0.5,
) -> list[dict]:
    """Cross-reference commercial/audio/visual anomalies into sponsor candidates.

    Returns a list of merged regions [{start, end, conf, reason}] that the
    caller can append to ``forced`` as ``sponsor`` intervals.

    Voting rule:
      • >= ``min_votes`` distinct evidence types overlapping → strong candidate
      • OR a single commercial cluster with score >= ``high_score_solo`` AND
        at least one of (audio | visual) anomaly co-located → strong candidate
      • OR a strong stylistic-only signal (audio score ≥ ``coherence_solo_score``
        OR visual score ≥ ``coherence_solo_score``) inside a region where the
        transcript is unreliable. This handles inserted ads where Whisper
        either drops out or hallucinates content — there's nothing for the
        commercial cluster detector to grab onto, so the style-shift
        detectors carry the load. Outside low-confidence zones the bar
        remains at 2 votes, so editorial cutaways aren't promoted.
      • Otherwise the region is dropped.

    Regions overlapping any existing forced interval (from scene-burst path)
    are absorbed into that interval rather than re-emitted, so we don't
    create duplicate sponsor blocks.
    """
    low_confidence_zones = low_confidence_zones or []
    if not (commercial_clusters or audio_anomalies or visual_anomalies):
        return []

    # Build a flat list of (start, end, source, score) tuples.
    events: list[tuple[float, float, str, float]] = []
    for c in commercial_clusters:
        events.append((float(c["start"]), float(c["end"]), "commercial", float(c.get("score", 0.5))))
    for a in audio_anomalies:
        events.append((float(a["start"]), float(a["end"]), "audio",      float(a.get("score", 1.0))))
    for v in visual_anomalies:
        events.append((float(v["start"]), float(v["end"]), "visual",     float(v.get("score", 1.0))))

    if not events:
        return []

    # Group events into merged time-windows.
    events.sort(key=lambda x: x[0])
    groups: list[dict] = []
    cur = {
        "start":   events[0][0],
        "end":     events[0][1],
        "sources": {events[0][2]},
        "scores":  {events[0][2]: events[0][3]},
    }
    for s, e, src, sc in events[1:]:
        if s - cur["end"] <= merge_gap_seconds:
            cur["end"]            = max(cur["end"], e)
            cur["sources"].add(src)
            cur["scores"][src]    = max(cur["scores"].get(src, 0.0), sc)
        else:
            groups.append(cur)
            cur = {"start": s, "end": e, "sources": {src}, "scores": {src: sc}}
    groups.append(cur)

    # Score each group and decide whether it deserves to be flagged.
    out: list[dict] = []
    for g in groups:
        if g["end"] - g["start"] < min_duration:
            continue
        # Don't double-flag intro/outro positional zones.
        if g["start"] < intro_cutoff and g["end"] < intro_cutoff:
            continue
        if outro_cutoff and g["start"] > outro_cutoff and g["end"] > outro_cutoff:
            continue

        n_votes = len(g["sources"])
        comm_score   = g["scores"].get("commercial", 0.0)
        audio_score  = g["scores"].get("audio",  0.0)
        visual_score = g["scores"].get("visual", 0.0)

        # Fraction of this region that sits inside a low-confidence
        # transcript zone (Whisper unreliable). 0 means transcripts are
        # reliable, 1 means we should heavily trust style detectors.
        low_conf_ratio = _zone_overlap_ratio(g["start"], g["end"], low_confidence_zones)
        in_low_conf    = low_conf_ratio >= low_conf_overlap_min

        promote = False
        promotion_path = ""
        if n_votes >= min_votes:
            promote = True
            promotion_path = "votes"
        elif comm_score >= high_score_solo and (
            "audio" in g["sources"] or "visual" in g["sources"]
        ):
            promote = True
            promotion_path = "commercial+style"
        elif in_low_conf and (
            audio_score >= coherence_solo_score or visual_score >= coherence_solo_score
        ):
            # Single strong stylistic signal inside a region where the
            # transcript is unreliable — trust the style detector since
            # Whisper has nothing useful to contribute here.
            promote = True
            promotion_path = "low_confidence_solo"

        if not promote:
            continue

        # Skip regions already covered by an existing forced sponsor interval.
        # (Prevents emitting an overlapping duplicate after scene-burst already
        # caught the same ad.)
        absorbed = False
        for f in existing_forced:
            if f.get("label") == "sponsor" and _intervals_overlap(
                g["start"], g["end"], f["start"], f["end"],
            ):
                f["start"]  = min(f["start"], g["start"])
                f["end"]    = max(f["end"],   g["end"])
                f["reason"] = (
                    f.get("reason", "")
                    + f" | reinforced by {sorted(g['sources'])}"
                )
                absorbed = True
                break
        if absorbed:
            continue

        if n_votes >= 3:
            conf = base_conf_three
        elif promotion_path == "low_confidence_solo":
            # One signal alone — credible only because transcript is
            # unreliable. Use a moderate confidence so the cap layer can
            # demote it first if needed.
            conf = 0.72
        else:
            conf = base_conf_two

        sources_str = "+".join(sorted(g["sources"]))
        reason = f"coherence sponsor ({sources_str}; score {comm_score:.2f})"
        if promotion_path == "low_confidence_solo":
            reason += f"; low-confidence transcript ratio={low_conf_ratio:.2f}"
        out.append({
            "start":  g["start"],
            "end":    g["end"],
            "conf":   conf,
            "reason": reason,
        })

    if out:
        log.info(
            "Coherence-based sponsor candidates: %d region(s) %s",
            len(out),
            [(round(o["start"], 1), round(o["end"], 1), o["reason"]) for o in out],
        )
    return out


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
