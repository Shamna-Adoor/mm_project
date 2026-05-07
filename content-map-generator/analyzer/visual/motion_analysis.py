"""Detect low-motion (static) intervals via frame-to-frame pixel differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)


class StaticInterval(TypedDict):
    start: float
    end: float
    motion_score: float


def detect_static_intervals(
    frames_dir: str | Path,
    *,
    threshold: float = 5.0,
    min_duration: float = 5.0,
    resize_to: tuple[int, int] = (320, 180),
) -> list[StaticInterval]:
    """Find video intervals with near-zero motion.

    Computes mean absolute pixel difference between consecutive frames
    downsampled to *resize_to* for speed. Runs where the score stays
    below *threshold* for ≥ *min_duration* seconds are returned.
    """
    import cv2

    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 2:
        log.info("Not enough frames for motion analysis")
        return []

    log.info("Analysing motion across %d frames…", len(frame_paths))

    def load_gray(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        img = cv2.resize(img, resize_to)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    timestamps: list[float] = []
    scores: list[float] = []

    prev = load_gray(frame_paths[0])
    prev_ts = _ts(frame_paths[0])

    for path in frame_paths[1:]:
        curr = load_gray(path)
        curr_ts = _ts(path)
        diff = float(np.mean(np.abs(curr - prev)))
        mid_ts = (prev_ts + curr_ts) / 2.0
        timestamps.append(mid_ts)
        scores.append(diff)
        prev, prev_ts = curr, curr_ts

    # Merge consecutive low-motion frames into intervals
    intervals: list[StaticInterval] = []
    in_static  = False
    start_ts   = 0.0
    buf_scores: list[float] = []

    for ts, sc in zip(timestamps, scores):
        if sc < threshold:
            if not in_static:
                in_static = True
                start_ts  = ts
                buf_scores = []
            buf_scores.append(sc)
        else:
            if in_static:
                in_static = False
                duration  = ts - start_ts
                if duration >= min_duration:
                    intervals.append({
                        "start":        round(start_ts, 3),
                        "end":          round(ts, 3),
                        "motion_score": round(float(np.mean(buf_scores)), 4),
                    })

    if in_static and timestamps:
        duration = timestamps[-1] - start_ts
        if duration >= min_duration:
            intervals.append({
                "start":        round(start_ts, 3),
                "end":          round(timestamps[-1], 3),
                "motion_score": round(float(np.mean(buf_scores)), 4),
            })

    log.info("Found %d static interval(s)", len(intervals))
    return intervals


def _ts(path: Path) -> float:
    return int(path.stem.split("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect static intervals.")
    parser.add_argument("frames_dir")
    parser.add_argument("--threshold",    type=float, default=5.0)
    parser.add_argument("--min-duration", type=float, default=5.0)
    args = parser.parse_args()

    print(json.dumps(detect_static_intervals(args.frames_dir, threshold=args.threshold, min_duration=args.min_duration), indent=2))
