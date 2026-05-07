"""Detect visual STYLE shifts via HSV color-histogram comparison.

Inserted ads are typically color-graded very differently from the host
content: more saturated, brighter, more contrasty, or simply set in a
different environment (CGI scenes, product shots, lifestyle stock footage).
This module compares each sampled frame's HSV histogram against the
video's robust median histogram and flags continuous regions of unusually
distant frames.

It complements the existing scene-change detector (PySceneDetect) which
counts hard cuts. Cuts alone don't indicate ads — many videos cut
constantly. What matters is whether a *cluster* of cuts also shifts the
overall look-and-feel of the video.

Runtime cost: ~5–15s on a 30-min video — works on the same JPEG frames
that scene_detect / motion_analysis already produced (no extra ffmpeg
seek). Uses OpenCV (already a project dependency).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

_TS_RE = re.compile(r"frame_(\d{10})\.jpg$")


class VisualAnomaly(TypedDict):
    start:  float
    end:    float
    score:  float          # max distance from baseline in this region
    reason: str


def detect_visual_anomalies(
    frames_dir: str | Path,
    *,
    deviation_sigmas: float = 2.5,
    min_duration:     float = 6.0,
    pad_before:       float = 1.0,
    pad_after:        float = 2.0,
    hue_bins:         int   = 16,
    sat_bins:         int   = 8,
    val_bins:         int   = 8,
    resize_to:        tuple[int, int] = (320, 180),
) -> list[VisualAnomaly]:
    """Find continuous frame regions whose color signature differs from the baseline.

    Algorithm
    ---------
    1. For every sampled frame in ``frames_dir``, compute a normalized HSV
       3-D histogram. HSV (rather than RGB) because hue/saturation are far
       more diagnostic of color grading than raw RGB channels.
    2. Compute the per-bin median across all frames — this is the "baseline
       look" of the video.
    3. For each frame, compute the Bhattacharyya distance to the baseline.
    4. Mark frames whose distance is more than ``deviation_sigmas`` MADs
       from the median distance. Group consecutive marked frames.

    The detector is purely additive — it returns CANDIDATE intervals only.
    Frames must already be extracted at a fixed sampling rate (the existing
    pipeline does this via ``extract_frames``).
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        log.warning("Visual coherence skipped — OpenCV unavailable: %s", exc)
        return []

    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 8:
        log.info("Visual coherence skipped — need >= 8 frames (got %d)", len(frame_paths))
        return []

    log.info("Computing visual coherence histograms across %d frames…", len(frame_paths))

    histograms: list = []
    timestamps: list[float] = []
    for fp in frame_paths:
        ts = _frame_ts(fp.name)
        if ts is None:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img  = cv2.resize(img, resize_to)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [hue_bins, sat_bins, val_bins],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        histograms.append(hist.flatten().astype(np.float32))
        timestamps.append(ts)

    if len(histograms) < 8:
        return []

    H = np.stack(histograms, axis=0)               # (n_frames, n_bins)
    baseline = np.median(H, axis=0).astype(np.float32)
    if float(np.sum(baseline)) == 0.0:
        return []
    baseline = baseline / float(np.sum(baseline))

    distances = np.array([
        cv2.compareHist(h, baseline, cv2.HISTCMP_BHATTACHARYYA)
        for h in H
    ], dtype=np.float32)

    med = float(np.median(distances))
    mad = float(np.median(np.abs(distances - med)))
    if mad < 1e-9:
        # Effectively uniform video — nothing to flag.
        return []
    scaled_mad = mad * 1.4826
    threshold  = med + deviation_sigmas * scaled_mad

    # Group consecutive frames whose distance > threshold into regions.
    anomalies: list[VisualAnomaly] = []
    in_anom   = False
    start_ts  = 0.0
    region_max = 0.0
    last_ts    = timestamps[0]
    for ts, d in zip(timestamps, distances):
        if d > threshold:
            if not in_anom:
                in_anom    = True
                start_ts   = ts
                region_max = float(d)
            else:
                region_max = max(region_max, float(d))
            last_ts = ts
        else:
            if in_anom:
                in_anom = False
                anomalies.append({
                    "start":  round(max(0.0, start_ts - pad_before), 3),
                    "end":    round(last_ts + pad_after, 3),
                    "score":  round(float(region_max), 3),
                    "reason": f"visual color/style shift (dist={region_max:.2f}, baseline≈{med:.2f})",
                })
    if in_anom:
        anomalies.append({
            "start":  round(max(0.0, start_ts - pad_before), 3),
            "end":    round(last_ts + pad_after, 3),
            "score":  round(float(region_max), 3),
            "reason": f"visual color/style shift (dist={region_max:.2f}, baseline≈{med:.2f})",
        })

    # Drop short blips
    anomalies = [a for a in anomalies if a["end"] - a["start"] >= min_duration]

    log.info(
        "Found %d visual coherence anomaly region(s): %s",
        len(anomalies),
        [(round(a["start"], 1), round(a["end"], 1)) for a in anomalies],
    )
    return anomalies


def _frame_ts(name: str) -> float | None:
    """Parse a frame filename of form ``frame_{ms:010d}.jpg`` into seconds."""
    m = _TS_RE.search(name)
    if not m:
        return None
    return int(m.group(1)) / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect visual coherence anomalies.")
    parser.add_argument("frames_dir")
    parser.add_argument("--deviation-sigmas", type=float, default=2.5)
    args = parser.parse_args()

    out = detect_visual_anomalies(
        args.frames_dir,
        deviation_sigmas=args.deviation_sigmas,
    )
    print(json.dumps(out, indent=2))
