"""Detect audio STYLE shifts that look unlike the rest of the video.

Inserted ads are almost always recorded in a different studio with a
different microphone, mix, and ambient noise floor than the surrounding
content. This module finds short windows whose audio signature deviates
significantly from the video's robust median signature, regardless of
*what* is being said.

It complements the existing structural detectors (silence, scene-burst,
speech-gap) which rely on *how* the video is edited. A polar bear ad with
calm narrative and slow cuts is structurally invisible — but its audio
spectrum is dramatically different from a podcast interview.

Runtime cost: ~3–8s on a 30-min video. Uses librosa (already a project
dependency, no new installs needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)


class AudioAnomaly(TypedDict):
    start:  float
    end:    float
    score:  float          # average deviation count (0-4)
    reason: str            # human-readable description


def detect_audio_anomalies(
    audio_path: str | Path,
    *,
    window_seconds:    float = 5.0,
    deviation_sigmas:  float = 2.5,
    min_features_dev:  int   = 2,
    min_duration:      float = 6.0,
    pad_before:        float = 1.0,
    pad_after:         float = 2.0,
    sample_rate:       int   = 16000,
) -> list[AudioAnomaly]:
    """Find continuous regions where audio deviates from the video's baseline.

    Algorithm
    ---------
    1. Slice audio into ``window_seconds`` non-overlapping windows.
    2. Compute four spectral features per window: RMS energy, spectral
       centroid, spectral rolloff, and spectral flatness.
    3. Compute a robust baseline (median + MAD) for each feature across all
       windows. Robust statistics resist getting biased by the very
       anomalies we're trying to detect.
    4. A window is "anomalous" when at least ``min_features_dev`` features
       are more than ``deviation_sigmas`` MADs from the baseline.
    5. Group consecutive anomalous windows into intervals and pad slightly.

    The detector never modifies anything by itself — it only emits CANDIDATE
    anomalies. The fusion layer combines this with other signals.
    """
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        log.warning("Audio coherence skipped — librosa unavailable: %s", exc)
        return []

    audio_path = Path(audio_path)
    if not audio_path.exists():
        log.warning("Audio coherence skipped — audio file not found: %s", audio_path)
        return []

    log.info("Computing audio coherence features for %s…", audio_path.name)

    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as exc:
        log.warning("Audio coherence skipped — failed to load audio: %s", exc)
        return []

    if len(y) < window_seconds * sr:
        log.info("Audio too short for coherence analysis (need >= %.0fs)", window_seconds)
        return []

    win_len = int(window_seconds * sr)

    rms      = librosa.feature.rms(y=y, frame_length=win_len, hop_length=win_len)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=win_len, hop_length=win_len)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=win_len, hop_length=win_len)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=win_len, hop_length=win_len)[0]

    n = min(len(rms), len(centroid), len(rolloff), len(flatness))
    if n < 6:
        log.info("Audio too short for coherence analysis (need >= 6 windows)")
        return []

    rms      = np.asarray(rms[:n],      dtype=np.float32)
    centroid = np.asarray(centroid[:n], dtype=np.float32)
    rolloff  = np.asarray(rolloff[:n],  dtype=np.float32)
    flatness = np.asarray(flatness[:n], dtype=np.float32)

    feats = [rms, centroid, rolloff, flatness]
    feat_names = ["rms", "centroid", "rolloff", "flatness"]

    # Robust baseline: median + median absolute deviation
    deviation_counts = np.zeros(n, dtype=np.int32)
    deviating_per_window: list[list[str]] = [[] for _ in range(n)]
    for f, name in zip(feats, feat_names):
        med = float(np.median(f))
        mad = float(np.median(np.abs(f - med)))
        if mad < 1e-9:
            continue   # constant feature — no information
        # 1.4826 scales MAD to std-equivalent for normal data; we use raw MAD
        # for robustness on non-normal distributions.
        scaled_mad = mad * 1.4826
        threshold  = deviation_sigmas * scaled_mad
        for i in range(n):
            if abs(f[i] - med) > threshold:
                deviation_counts[i] += 1
                deviating_per_window[i].append(name)

    # Group consecutive windows whose deviation count meets the threshold.
    anomalies: list[AudioAnomaly] = []
    in_anom   = False
    start_idx = 0
    deviating_buf: list[str] = []
    for i in range(n):
        if deviation_counts[i] >= min_features_dev:
            if not in_anom:
                in_anom   = True
                start_idx = i
                deviating_buf = []
            deviating_buf.extend(deviating_per_window[i])
        else:
            if in_anom:
                in_anom = False
                _emit_anomaly(
                    anomalies, start_idx, i, window_seconds,
                    pad_before, pad_after,
                    deviation_counts[start_idx:i],
                    deviating_buf,
                )
    if in_anom:
        _emit_anomaly(
            anomalies, start_idx, n, window_seconds,
            pad_before, pad_after,
            deviation_counts[start_idx:n],
            deviating_buf,
        )

    # Drop short blips
    anomalies = [a for a in anomalies if a["end"] - a["start"] >= min_duration]

    log.info(
        "Found %d audio coherence anomaly region(s): %s",
        len(anomalies),
        [(round(a["start"], 1), round(a["end"], 1)) for a in anomalies],
    )
    return anomalies


def _emit_anomaly(out, i_start, i_end, window_seconds, pad_before, pad_after, counts, deviating):
    import numpy as np
    s = max(0.0, i_start * window_seconds - pad_before)
    e = i_end * window_seconds + pad_after
    avg_count = float(np.mean(counts)) if len(counts) else 0.0
    feats_seen = list(dict.fromkeys(deviating))[:4]
    out.append({
        "start":  round(s, 3),
        "end":    round(e, 3),
        "score":  round(avg_count, 2),
        "reason": f"audio style shift ({', '.join(feats_seen) or 'multi-feature'})",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect audio coherence anomalies.")
    parser.add_argument("audio")
    parser.add_argument("--window-seconds",   type=float, default=5.0)
    parser.add_argument("--deviation-sigmas", type=float, default=2.5)
    parser.add_argument("--min-features-dev", type=int,   default=2)
    args = parser.parse_args()

    out = detect_audio_anomalies(
        args.audio,
        window_seconds=args.window_seconds,
        deviation_sigmas=args.deviation_sigmas,
        min_features_dev=args.min_features_dev,
    )
    print(json.dumps(out, indent=2))
