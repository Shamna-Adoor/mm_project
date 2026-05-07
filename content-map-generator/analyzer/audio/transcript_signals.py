"""Extract sponsor and boilerplate phrase signals from a transcript."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal, TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

# Strong patterns — very specific, low false-positive rate
SPONSOR_PATTERNS_STRONG = [
    "sponsored by",
    "brought to you by",
    "this video is sponsored",
    "this episode is sponsored",
    "this episode is supported by",
    "paid partnership",
    "in partnership with",
    "partnered with",
    "use code",
    "use my code",
    "use promo code",
    "discount code",
    "promo code",
    "coupon code",
    "percent off",
    "% off",
    "first month free",
    "click the link below",
    "link in the description",
    "this is an ad",
    "this is a paid",
    "ad supported",
    "affiliate link",
    "thanks to our sponsor",
    "today's sponsor",
    "today's video is sponsored",
]

# Weak patterns — only fire when combined with context
SPONSOR_PATTERNS_WEAK = [
    "sign up for free",
    "sign up at",
    "head over to",
    "check them out",
    "free trial",
    "get it for free",
    "i want to thank",
    "not sponsored but",
]

SPONSOR_PATTERNS = SPONSOR_PATTERNS_STRONG + SPONSOR_PATTERNS_WEAK

INTRO_PATTERNS = [
    "welcome back",
    "what's up everybody",
    "hey guys",
    "hey everyone",
    "welcome to",
    "hello everyone",
    "good morning",
    "good evening",
    "today we're",
    "in today's video",
    "in this video",
    "today i'm going to",
    "what is going on",
]

OUTRO_PATTERNS = [
    "thanks for watching",
    "see you next time",
    "see you in the next",
    "don't forget to subscribe",
    "hit that bell",
    "smash that like",
    "like and subscribe",
    "until next time",
    "peace out",
    "that's all for today",
    "that's it for today",
    "if you enjoyed this",
    "leave a comment",
]

BoilerplateType = Literal["intro", "outro", "cta"]


class SponsorPhrase(TypedDict):
    start: float
    end: float
    phrase: str
    context: str


class BoilerplatePhrase(TypedDict):
    start: float
    end: float
    phrase: str
    type: BoilerplateType


def find_sponsor_phrases(
    transcript: list[dict],
    *,
    similarity_threshold: int = 88,
) -> list[SponsorPhrase]:
    """Scan transcript segments for sponsor-related language via fuzzy matching."""
    from rapidfuzz import fuzz

    hits: list[SponsorPhrase] = []
    for seg in transcript:
        text_lower = seg["text"].lower()

        # Strong patterns: high confidence, standard threshold
        for pattern in SPONSOR_PATTERNS_STRONG:
            if fuzz.partial_ratio(pattern, text_lower) >= similarity_threshold:
                hits.append({"start": float(seg["start"]), "end": float(seg["end"]),
                              "phrase": pattern, "context": seg["text"]})
                break
        else:
            # Weak patterns: require higher threshold to reduce false positives
            for pattern in SPONSOR_PATTERNS_WEAK:
                if fuzz.partial_ratio(pattern, text_lower) >= 92:
                    hits.append({"start": float(seg["start"]), "end": float(seg["end"]),
                                  "phrase": pattern, "context": seg["text"]})
                    break

    log.info("Found %d sponsor phrase hit(s)", len(hits))
    return hits


def find_boilerplate(
    transcript: list[dict],
    *,
    similarity_threshold: int = 85,
) -> list[BoilerplatePhrase]:
    """Scan transcript for intro / outro / CTA boilerplate phrases."""
    from rapidfuzz import fuzz

    hits: list[BoilerplatePhrase] = []
    for seg in transcript:
        text_lower = seg["text"].lower()

        for pattern in INTRO_PATTERNS:
            if fuzz.partial_ratio(pattern, text_lower) >= similarity_threshold:
                hits.append({
                    "start":  float(seg["start"]),
                    "end":    float(seg["end"]),
                    "phrase": pattern,
                    "type":   "intro",
                })
                break

        for pattern in OUTRO_PATTERNS:
            if fuzz.partial_ratio(pattern, text_lower) >= similarity_threshold:
                hits.append({
                    "start":  float(seg["start"]),
                    "end":    float(seg["end"]),
                    "phrase": pattern,
                    "type":   "outro",
                })
                break

    log.info("Found %d boilerplate hit(s)", len(hits))
    return hits


# ── Commercial-language cluster detection ──────────────────────────────────────
# Beyond simple phrase matching, this catches ad regions by looking at the
# DENSITY of commercial signals across consecutive transcript segments. A
# single phrase (e.g., "head over to") might be a false positive — but a
# dense cluster of brand mentions, URLs, prices, and CTAs in a 30-second
# window is almost always an ad break.

# URL / domain signals — strong commercial cue, especially with a CTA.
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+|"
    r"\b[a-z0-9][-a-z0-9]{1,30}\.(?:com|io|net|co|app|tv|gg|ly|me)\b",
    re.IGNORECASE,
)

# Pricing and percentage-off language (very strong commercial signal).
_PRICE_RE = re.compile(
    r"\$\s*\d+(?:\.\d+)?|"
    r"\b\d+(?:\.\d+)?\s*(?:dollars?|bucks?)\b|"
    r"\b\d+\s*%\s*(?:off|discount)\b|"
    r"\b(?:starting\s+at|just|only)\s*\$\d+",
    re.IGNORECASE,
)

# Time-limited / urgency cues common in ad copy.
_URGENCY_PATTERNS = [
    "limited time", "today only", "for a limited", "while supplies last",
    "exclusive offer", "act now", "don't miss", "deal expires",
    "this week only", "ends soon",
]

# Strong call-to-action verbs that frequently bracket ad breaks.
_CTA_PATTERNS = [
    "head to", "visit", "go to", "sign up at", "download now",
    "get started", "try it free", "free trial", "first month free",
    "click the link", "tap the link", "link in bio", "link in description",
    "swipe up", "use my link", "use the link",
]

# Common sponsor brand names — only flagged when their density spikes;
# a single mention won't flip a region. This is intentionally conservative.
_SPONSOR_BRANDS = [
    "nordvpn", "expressvpn", "surfshark", "proton vpn", "atlas vpn",
    "raid shadow legends", "world of tanks", "world of warships",
    "clash of clans", "clash royale", "rise of kingdoms",
    "squarespace", "shopify", "wix",
    "skillshare", "masterclass", "brilliant", "audible",
    "betterhelp", "talkspace",
    "hellofresh", "factor", "blue apron",
    "honey", "rocket money", "rakuten",
    "manscaped", "athletic greens", "ag1",
    "manscaped", "keeps", "hims", "ridge wallet",
    "established titles", "displate", "dollar shave club",
]

# Generic broadcast / streaming ad phrasings. These cover a HUGE fraction of
# televised consumer ads (food, appliances, automotive, retail, finance,
# pharma, lifestyle) without naming any specific brand. They appear in real
# ad copy with very high frequency and almost never in editorial content.
# Kept short and generic on purpose: missing one is fine, but each pattern
# we DO have should fire only on commercial language.
_BROADCAST_AD_PATTERNS = [
    # Product-launch language
    "introducing", "the all-new", "all-new", "all new",
    "from the makers of", "from the maker of",
    "now in stores", "now available", "available now",
    "available at", "available wherever", "find it at", "find us at",
    "coming soon to", "coming soon",
    # Provenance / authority claims (extremely common in mature-brand ads)
    "trusted by", "trusted for",
    "for over",                # "for over 100 years", "for over 50 years"
    "since 18", "since 19",    # "since 1869", "since 1923" — historic-brand cliché
    # Ad-copy verbs / lifestyle framings
    "experience the", "discover the", "discover a",
    "made with", "made for", "made from",
    "engineered to", "engineered for",
    "designed to", "designed for",
    "for the way you live", "for the way you",
    "the future of", "the next generation of",
    "see what's new", "see what's possible",
    # Direct-response cues (sale / promotion / availability)
    "in stores now", "shop now", "order now",
    "save more", "save big", "save up to",
    # Pharma / supplement boilerplate disclaimers
    "ask your doctor", "talk to your doctor",
    "side effects may include",
    "do not take", "consult your physician",
    # Insurance / finance ad cliches
    "switch and save", "fifteen minutes could save you",
    "we'll be there", "we've got you covered",
]


class CommercialCluster(TypedDict):
    start: float
    end: float
    score: float            # aggregated density / 0–1
    signals: list[str]      # which kinds of cues fired
    snippets: list[str]     # short excerpts for explainability


def _score_segment_commercial(text: str) -> tuple[float, list[str]]:
    """Score a single transcript segment for commercial-ness.

    Returns (score, signals_fired). Score is roughly the count of distinct
    cue types present, scaled into [0, 1] via a soft cap.
    """
    text_lower = text.lower()
    fired: list[str] = []
    score = 0.0

    if _URL_RE.search(text):
        fired.append("url"); score += 1.5
    if _PRICE_RE.search(text):
        fired.append("price"); score += 1.5

    for p in _URGENCY_PATTERNS:
        if p in text_lower:
            fired.append("urgency"); score += 0.8
            break

    for p in _CTA_PATTERNS:
        if p in text_lower:
            fired.append("cta"); score += 0.7
            break

    brand_hits = [b for b in _SPONSOR_BRANDS if b in text_lower]
    if brand_hits:
        fired.append("brand")
        score += min(2.0, 0.9 * len(brand_hits))

    # Strong-sponsor phrases (already-defined list) count too — but we don't
    # double-count if find_sponsor_phrases already caught them; instead they
    # boost density confidence.
    for p in SPONSOR_PATTERNS_STRONG:
        if p in text_lower:
            fired.append("sponsor_phrase"); score += 1.2
            break

    # Generic broadcast/streaming ad phrasings. Each match adds a moderate
    # weight; the cluster window aggregator decides whether the density
    # crosses the threshold for the region. A single match alone is never
    # enough to promote.
    broadcast_hits = sum(1 for p in _BROADCAST_AD_PATTERNS if p in text_lower)
    if broadcast_hits:
        fired.append("broadcast_ad")
        score += min(1.6, 0.7 * broadcast_hits)

    return min(1.0, score / 3.0), list(dict.fromkeys(fired))   # dedupe, preserve order


def find_commercial_clusters(
    transcript: list[dict],
    *,
    window_seconds: float = 30.0,
    min_window_score: float = 0.65,
    pad_before: float = 1.5,
    pad_after:  float = 3.0,
    merge_gap_seconds: float = 8.0,
) -> list[CommercialCluster]:
    """Detect REGIONS with high density of commercial language.

    Slides a window over the transcript and flags windows where the sum of
    per-segment commercial scores crosses ``min_window_score``. Adjacent
    flagged windows are merged into a single cluster.

    This is purely additive — it does not replace ``find_sponsor_phrases``.
    Both can co-fire and reinforce each other.
    """
    if not transcript:
        return []

    # Step 1 — score each transcript segment.
    scored: list[tuple[float, float, float, list[str], str]] = []
    for seg in transcript:
        s, e = float(seg["start"]), float(seg["end"])
        text = seg.get("text", "") or ""
        score, fired = _score_segment_commercial(text)
        if score > 0:
            scored.append((s, e, score, fired, text.strip()))

    if not scored:
        log.info("Found 0 commercial cluster(s)")
        return []

    # Step 2 — slide a window over time, accumulating scores.
    half = window_seconds / 2.0
    flagged_windows: list[tuple[float, float, float, list[str], list[str]]] = []
    for i, (s, e, sc, fired, text) in enumerate(scored):
        center = (s + e) / 2.0
        lo, hi = center - half, center + half
        nearby = [(ss, ee, ssc, ff, tt) for ss, ee, ssc, ff, tt in scored if lo <= (ss+ee)/2.0 <= hi]
        window_score = sum(x[2] for x in nearby)
        if window_score >= min_window_score:
            sigs   = []
            snips  = []
            for _, _, _, f, t in nearby:
                sigs.extend(f)
                snips.append(t)
            flagged_windows.append((
                min(x[0] for x in nearby),
                max(x[1] for x in nearby),
                window_score,
                list(dict.fromkeys(sigs)),
                snips[:5],          # cap snippets for log readability
            ))

    if not flagged_windows:
        log.info("Found 0 commercial cluster(s)")
        return []

    # Step 3 — merge adjacent / overlapping flagged windows.
    flagged_windows.sort(key=lambda x: x[0])
    clusters: list[CommercialCluster] = []
    cur_s, cur_e, cur_sc, cur_sigs, cur_snips = flagged_windows[0]
    for s, e, sc, sigs, snips in flagged_windows[1:]:
        if s - cur_e <= merge_gap_seconds:
            cur_e   = max(cur_e, e)
            cur_sc  = max(cur_sc, sc)
            cur_sigs = list(dict.fromkeys(cur_sigs + sigs))
            cur_snips = (cur_snips + snips)[:5]
        else:
            clusters.append({
                "start":    max(0.0, cur_s - pad_before),
                "end":      cur_e + pad_after,
                "score":    round(min(1.0, cur_sc / 3.0), 3),
                "signals":  cur_sigs,
                "snippets": cur_snips,
            })
            cur_s, cur_e, cur_sc, cur_sigs, cur_snips = s, e, sc, sigs, snips
    clusters.append({
        "start":    max(0.0, cur_s - pad_before),
        "end":      cur_e + pad_after,
        "score":    round(min(1.0, cur_sc / 3.0), 3),
        "signals":  cur_sigs,
        "snippets": cur_snips,
    })

    # Step 4 — drop clusters that are too short (likely noise).
    clusters = [c for c in clusters if c["end"] - c["start"] >= 5.0]

    log.info(
        "Found %d commercial cluster(s): %s",
        len(clusters),
        [(round(c["start"], 1), round(c["end"], 1), c["signals"]) for c in clusters],
    )
    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find sponsor/boilerplate phrases in a transcript JSON.")
    parser.add_argument("transcript", help="Path to transcript JSON from transcribe.py")
    parser.add_argument("--threshold", type=int, default=85)
    args = parser.parse_args()

    with open(args.transcript) as f:
        tx = json.load(f)

    out = {
        "sponsor_phrases":     find_sponsor_phrases(tx, similarity_threshold=args.threshold),
        "boilerplate_phrases": find_boilerplate(tx,   similarity_threshold=args.threshold),
        "commercial_clusters": find_commercial_clusters(tx),
    }
    print(json.dumps(out, indent=2))
