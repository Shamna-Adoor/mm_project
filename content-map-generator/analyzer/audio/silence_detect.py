"""Detect silence intervals in an audio file using pydub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)


class SilenceInterval(TypedDict):
    start: float
    end: float
    energy: float


def detect_silence(
    audio_path: str | Path,
    *,
    min_duration: float = 2.0,
    threshold_db: float = -40.0,
) -> list[SilenceInterval]:
    """Find intervals where audio energy falls below *threshold_db* dBFS.

    Parameters
    ----------
    min_duration:
        Minimum gap length in seconds to report.
    threshold_db:
        dBFS ceiling; audio below this level is considered silent.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_silence as _pydub_detect

    log.info("Detecting silence in %s (threshold=%.0f dBFS, min=%.1fs)…", Path(audio_path).name, threshold_db, min_duration)

    audio = AudioSegment.from_file(str(audio_path))
    min_silence_ms = int(min_duration * 1000)

    # Returns [[start_ms, end_ms], …]
    raw = _pydub_detect(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=threshold_db,
        seek_step=50,
    )

    intervals: list[SilenceInterval] = []
    for start_ms, end_ms in raw:
        chunk = audio[start_ms:end_ms]
        db = chunk.dBFS
        intervals.append({
            "start": start_ms / 1000.0,
            "end":   end_ms   / 1000.0,
            "energy": float(db) if db != float("-inf") else -100.0,
        })

    log.info("Found %d silence interval(s)", len(intervals))
    return intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect silence in audio.")
    parser.add_argument("audio")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--threshold-db", type=float, default=-40.0)
    args = parser.parse_args()

    print(json.dumps(detect_silence(args.audio, min_duration=args.min_duration, threshold_db=args.threshold_db), indent=2))
