"""Run OCR on sampled frames and flag sponsor/ad keywords."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

FLAG_KEYWORDS = [
    "sponsor", "ad", "advertisement", "subscribe", "starting soon",
    "promo", "discount", "code", "% off", ".com", ".io", ".net",
    "http", "www", "sign up", "use code", "link below",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|io|net|org)\b", re.I)


class OcrDetection(TypedDict):
    timestamp: float
    text: str
    bbox: list[float]


def ocr_frames(
    frames_dir: str | Path,
    *,
    sample_every_n_seconds: int = 5,
    min_confidence: int = 60,
    min_text_length: int = 2,
) -> list[OcrDetection]:
    """Run pytesseract on every Nth frame and return flagged detections."""
    import pytesseract
    from PIL import Image

    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not all_frames:
        log.warning("No frames found in %s", frames_dir)
        return []

    log.info("Running OCR on frames in %s (every %ds)…", frames_dir, sample_every_n_seconds)

    last_sampled_sec = -sample_every_n_seconds
    detections: list[OcrDetection] = []

    for frame_path in all_frames:
        ts = _timestamp_from_filename(frame_path)
        if ts - last_sampled_sec < sample_every_n_seconds:
            continue
        last_sampled_sec = ts

        try:
            img  = Image.open(frame_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            log.debug("OCR failed on %s: %s", frame_path.name, e)
            continue

        # Collect words that pass confidence + length filter
        words_seen: list[tuple[str, list[float]]] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = int(data["conf"][i])
            if conf < min_confidence or len(word) < min_text_length:
                continue
            bbox = [
                float(data["left"][i]),
                float(data["top"][i]),
                float(data["width"][i]),
                float(data["height"][i]),
            ]
            words_seen.append((word, bbox))

        if not words_seen:
            continue

        full_text = " ".join(w for w, _ in words_seen)
        text_lower = full_text.lower()

        flagged = (
            any(kw in text_lower for kw in FLAG_KEYWORDS)
            or bool(_URL_RE.search(full_text))
        )

        if flagged:
            # Use bbox of first matching token as the representative location
            best_word, best_bbox = words_seen[0]
            detections.append({
                "timestamp": ts,
                "text":      full_text,
                "bbox":      best_bbox,
            })
            log.debug("OCR hit at %.1fs: %s", ts, full_text[:80])

    log.info("OCR produced %d flagged detection(s)", len(detections))
    return detections


def _timestamp_from_filename(path: Path) -> float:
    """Parse millisecond timestamp from ``frame_{ms}.jpg`` filename."""
    stem = path.stem  # e.g. "frame_0000005000"
    ms   = int(stem.split("_", 1)[1])
    return ms / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR frames and flag sponsor keywords.")
    parser.add_argument("frames_dir")
    parser.add_argument("--sample-every", type=int, default=5, dest="sample_every_n_seconds")
    parser.add_argument("--min-confidence", type=int, default=60)
    args = parser.parse_args()

    print(json.dumps(ocr_frames(args.frames_dir, sample_every_n_seconds=args.sample_every_n_seconds, min_confidence=args.min_confidence), indent=2))
