"""Detect music intervals using librosa spectral features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)


class MusicInterval(TypedDict):
    start: float
    end: float
    confidence: float


def detect_music(
    audio_path: str | Path,
    *,
    frame_duration: float = 1.0,
    hop_duration: float = 0.5,
) -> list[MusicInterval]:
    """Classify 1-second frames as music or speech and return music intervals.

    Heuristic uses three features per frame:
    - Zero-crossing rate (ZCR): low ZCR → periodic → music
    - Spectral flatness: low flatness → tonal → music
    - MFCC variance: low variance → stable timbre → music
    """
    import librosa

    log.info("Detecting music in %s…", Path(audio_path).name)

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    frame_len = int(frame_duration * sr)
    hop_len   = int(hop_duration   * sr)

    zcr      = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_len, hop_length=hop_len)[0]
    mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len)
    mfcc_var = np.var(mfccs, axis=0)  # shape: (n_frames,)

    n_frames = min(len(zcr), len(flatness), len(mfcc_var))

    music_flags  = []
    confidences  = []
    for i in range(n_frames):
        is_m, conf = _is_music_frame(float(zcr[i]), float(mfcc_var[i]), float(flatness[i]))
        music_flags.append(is_m)
        confidences.append(conf)

    # Merge consecutive music frames into intervals
    intervals: list[MusicInterval] = []
    in_music    = False
    start_frame = 0

    for i, is_m in enumerate(music_flags):
        if is_m and not in_music:
            in_music    = True
            start_frame = i
        elif not is_m and in_music:
            in_music = False
            intervals.append({
                "start":      round(start_frame * hop_duration, 3),
                "end":        round(i            * hop_duration, 3),
                "confidence": round(float(np.mean(confidences[start_frame:i])), 3),
            })

    if in_music:
        intervals.append({
            "start":      round(start_frame  * hop_duration, 3),
            "end":        round(n_frames      * hop_duration, 3),
            "confidence": round(float(np.mean(confidences[start_frame:])), 3),
        })

    # Drop short blips — real music runs for at least 4 seconds
    intervals = [iv for iv in intervals if iv["end"] - iv["start"] >= 4.0]

    log.info("Found %d music interval(s)", len(intervals))
    return intervals


def _is_music_frame(zcr_val: float, mfcc_var_val: float, flatness_val: float) -> tuple[bool, float]:
    """Score a single frame and return (is_music, confidence ∈ [0,1]).

    All three features must agree — avoids false positives on speech/silence.
    """
    score = 0.0

    # ZCR must be quite low (pure speech sits around 0.08–0.15)
    if zcr_val < 0.04:
        score += 0.40
    elif zcr_val < 0.07:
        score += 0.15

    # Spectral flatness: music is tonal (low flatness); speech is noisy (high)
    if flatness_val < 0.02:
        score += 0.40
    elif flatness_val < 0.06:
        score += 0.15

    # MFCC variance: music has stable timbre
    if mfcc_var_val < 30:
        score += 0.20
    elif mfcc_var_val < 70:
        score += 0.08

    # Require all three to partially agree (threshold raised from 0.50 → 0.70)
    return score >= 0.70, min(1.0, score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect music in audio.")
    parser.add_argument("audio")
    parser.add_argument("--frame-duration", type=float, default=1.0)
    parser.add_argument("--hop-duration",   type=float, default=0.5)
    args = parser.parse_args()

    print(json.dumps(detect_music(args.audio, frame_duration=args.frame_duration, hop_duration=args.hop_duration), indent=2))
