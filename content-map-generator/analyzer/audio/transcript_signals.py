"""Extract sponsor and boilerplate phrase signals from a transcript."""

from __future__ import annotations

import argparse
import json
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find sponsor/boilerplate phrases in a transcript JSON.")
    parser.add_argument("transcript", help="Path to transcript JSON from transcribe.py")
    parser.add_argument("--threshold", type=int, default=85)
    args = parser.parse_args()

    with open(args.transcript) as f:
        tx = json.load(f)

    out = {
        "sponsor_phrases":    find_sponsor_phrases(tx, similarity_threshold=args.threshold),
        "boilerplate_phrases": find_boilerplate(tx,   similarity_threshold=args.threshold),
    }
    print(json.dumps(out, indent=2))
