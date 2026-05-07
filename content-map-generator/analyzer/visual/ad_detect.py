"""Detect ad intervals by finding paired abrupt visual transitions between frames.

This module provides both:
- detect_ad_intervals_visual_only(): the original visual-only spike-pair method
- detect_ad_intervals(): wrapper that attempts multimodal detection first,
  falling back to visual-only when audio is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)


class AdInterval(TypedDict):
    start:      float
    end:        float
    confidence: float


def detect_ad_intervals(
    frames_dir: str | Path,
    *,
    audio_path: str | Path | None = None,
    video_duration: float = 0.0,
    spike_factor: float = 2.5,
    min_ad_duration: float = 15.0,
    max_ad_duration: float = 360.0,
    resize_to: tuple[int, int] = (320, 180),
) -> list[AdInterval]:
    """Detect ad intervals, preferring multimodal when audio is available.

    If audio_path is provided, attempts multimodal 2-sigma detection first.
    Falls back to visual-only spike-pair method.
    """
    if audio_path is not None:
        audio_p = Path(audio_path)
        if audio_p.exists():
            try:
                from analyzer.multimodal_detect import detect_multimodal_ads
                results = detect_multimodal_ads(
                    audio_p, frames_dir, video_duration=video_duration,
                )
                if results:
                    log.info("Multimodal detection succeeded: %d ad(s)", len(results))
                    return results
            except Exception as exc:
                log.warning("Multimodal detection failed, using visual fallback: %s", exc)

    return detect_ad_intervals_visual_only(
        frames_dir,
        spike_factor=spike_factor,
        min_ad_duration=min_ad_duration,
        max_ad_duration=max_ad_duration,
        resize_to=resize_to,
    )


def detect_ad_intervals_visual_only(
    frames_dir: str | Path,
    *,
    spike_factor: float = 2.5,
    min_ad_duration: float = 15.0,
    max_ad_duration: float = 360.0,
    resize_to: tuple[int, int] = (320, 180),
) -> list[AdInterval]:
    """Find ad intervals from abrupt frame-to-frame visual transitions (visual-only fallback).

    Algorithm
    ---------
    1. Compute mean-absolute-pixel-difference between consecutive frames.
    2. Identify "spike" timestamps where diff > median + spike_factor * std.
    3. Merge spikes within 2 s of each other into a single boundary.
    4. Any consecutive boundary pair whose gap is in [min_ad, max_ad] seconds
       is returned as a candidate ad interval.
    """
    import cv2

    frames_dir  = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 4:
        log.info("Not enough frames for ad detection")
        return []

    log.info("Detecting ad intervals from %d frames…", len(frame_paths))

    def load_gray(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        img = cv2.resize(img, resize_to)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    timestamps: list[float] = []
    diffs:      list[float] = []

    prev    = load_gray(frame_paths[0])
    prev_ts = _ts(frame_paths[0])

    for path in frame_paths[1:]:
        curr    = load_gray(path)
        curr_ts = _ts(path)
        diff    = float(np.mean(np.abs(curr - prev)))
        timestamps.append((prev_ts + curr_ts) / 2.0)
        diffs.append(diff)
        prev, prev_ts = curr, curr_ts

    arr       = np.array(diffs)
    threshold = float(np.median(arr) + spike_factor * np.std(arr))
    log.info("Frame-diff spike threshold: %.2f (median=%.2f std=%.2f)",
             threshold, float(np.median(arr)), float(np.std(arr)))

    spike_times: list[float] = [
        ts for ts, d in zip(timestamps, diffs) if d >= threshold
    ]
    log.info("Found %d raw spike(s)", len(spike_times))

    # Merge spikes within 2 s into a single boundary (take average timestamp)
    merged: list[float] = []
    for t in spike_times:
        if merged and t - merged[-1] < 2.0:
            merged[-1] = (merged[-1] + t) / 2.0
        else:
            merged.append(t)

    # Pair consecutive boundaries whose gap matches a typical ad duration
    intervals: list[AdInterval] = []
    for i in range(len(merged) - 1):
        start = merged[i]
        end   = merged[i + 1]
        gap   = end - start
        if min_ad_duration <= gap <= max_ad_duration:
            # Confidence: how far above threshold the two boundary spikes were
            si = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - start))
            ei = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - end))
            peak = max(diffs[si], diffs[ei])
            conf = min(1.0, round((peak / threshold - 1.0) * 0.5 + 0.6, 3))
            intervals.append({"start": round(start, 3), "end": round(end, 3), "confidence": conf})

    log.info("Found %d candidate ad interval(s)", len(intervals))
    return intervals


def _ts(path: Path) -> float:
    return int(path.stem.split("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ad intervals from frame transitions.")
    parser.add_argument("frames_dir")
    parser.add_argument("--audio",            default=None, help="Path to audio WAV for multimodal detection")
    parser.add_argument("--duration",         type=float, default=0.0)
    parser.add_argument("--spike-factor",     type=float, default=2.5)
    parser.add_argument("--min-ad-duration",  type=float, default=15.0)
    parser.add_argument("--max-ad-duration",  type=float, default=360.0)
    args = parser.parse_args()

    print(json.dumps(
        detect_ad_intervals(
            args.frames_dir,
            audio_path=args.audio,
            video_duration=args.duration,
            spike_factor=args.spike_factor,
            min_ad_duration=args.min_ad_duration,
            max_ad_duration=args.max_ad_duration,
        ),
        indent=2,
    ))
