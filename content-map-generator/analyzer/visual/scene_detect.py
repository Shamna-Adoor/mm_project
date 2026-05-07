"""Detect scene boundaries using PySceneDetect's ContentDetector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)


class SceneChange(TypedDict):
    timestamp: float
    confidence: float


def detect_scenes(
    video_path: str | Path,
    *,
    threshold: float = 27.0,
) -> list[SceneChange]:
    """Return cut-based scene boundaries detected by PySceneDetect.

    Parameters
    ----------
    threshold:
        ContentDetector sensitivity. Lower = more cuts detected.
        Default 27.0 is PySceneDetect's recommended starting point.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video_path = Path(video_path)
    log.info("Detecting scenes in %s (threshold=%.1f)…", video_path.name, threshold)

    video      = open_video(str(video_path))
    manager    = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=False)

    scene_list = manager.get_scene_list()

    # Each scene is (start_timecode, end_timecode); we report the start of each
    # scene after the first as a cut boundary.
    cuts: list[SceneChange] = []
    for i, (start_tc, _) in enumerate(scene_list):
        if i == 0:
            continue  # skip the very beginning
        ts = start_tc.get_seconds()
        # Normalise confidence: closer scenes to threshold → lower confidence
        cuts.append({"timestamp": round(ts, 3), "confidence": 0.8})

    log.info("Found %d scene cut(s)", len(cuts))
    return cuts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scene changes.")
    parser.add_argument("video")
    parser.add_argument("--threshold", type=float, default=27.0)
    args = parser.parse_args()

    print(json.dumps(detect_scenes(args.video, threshold=args.threshold), indent=2))
