"""Detect video intro in the first 10-15 seconds via multimodal feature comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)

INTRO_WINDOW_SEC = 15.0
BASELINE_START_SEC = 20.0
BASELINE_END_SEC = 50.0
MIN_INTRO_DURATION = 3.0
FEATURE_DISTANCE_THRESHOLD = 1.8


class IntroInterval(TypedDict):
    start: float
    end: float
    confidence: float
    reason: str


def detect_intro(
    audio_path: str | Path,
    frames_dir: str | Path,
    *,
    video_duration: float = 0.0,
    intro_window: float = INTRO_WINDOW_SEC,
    baseline_start: float = BASELINE_START_SEC,
    baseline_end: float = BASELINE_END_SEC,
    distance_threshold: float = FEATURE_DISTANCE_THRESHOLD,
) -> IntroInterval | None:
    """Detect an intro segment in the first 10-15 seconds of a video.

    Compares audio+video features in [0, intro_window] against a baseline
    window [baseline_start, baseline_end] that represents early main content.

    Returns an IntroInterval if the intro region is sufficiently distinct
    from the main content that follows, None otherwise.
    """
    audio_path = Path(audio_path)
    frames_dir = Path(frames_dir)

    if not audio_path.exists():
        log.warning("Audio not found for intro detection: %s", audio_path)
        return None

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 4:
        log.info("Not enough frames for intro detection")
        return None

    # Need enough video for both intro and baseline windows
    max_frame_ts = _frame_timestamp(frame_paths[-1])
    if max_frame_ts < baseline_start:
        log.info("Video too short for intro baseline comparison (%.1fs)", max_frame_ts)
        return None

    audio_features = _extract_audio_segment_features(audio_path, 0, intro_window)
    audio_baseline = _extract_audio_segment_features(audio_path, baseline_start, baseline_end)

    if audio_features is None or audio_baseline is None:
        return None

    video_features = _extract_video_segment_features(frame_paths, 0, intro_window)
    video_baseline = _extract_video_segment_features(frame_paths, baseline_start, baseline_end)

    if video_features is None or video_baseline is None:
        return None

    # Compute feature distances between intro and baseline
    audio_distance = _feature_distance(audio_features, audio_baseline)
    video_distance = _feature_distance(video_features, video_baseline)

    # Both modalities should show the intro is different from main content
    combined_distance = (audio_distance + video_distance) / 2.0

    log.info(
        "Intro detection: audio_dist=%.2f video_dist=%.2f combined=%.2f (threshold=%.2f)",
        audio_distance, video_distance, combined_distance, distance_threshold,
    )

    if combined_distance < distance_threshold:
        log.info("Intro not detected — features too similar to main content")
        return None

    # Find the transition point: where features start resembling the baseline
    intro_end = _find_transition_point(
        audio_path, frames_dir, frame_paths, intro_window, baseline_start
    )

    if intro_end < MIN_INTRO_DURATION:
        log.info("Detected intro too short (%.1fs), skipping", intro_end)
        return None

    confidence = min(1.0, round(0.5 + (combined_distance - distance_threshold) * 0.25, 3))

    reasons: list[str] = []
    if audio_distance > distance_threshold:
        reasons.append("distinct audio signature")
    if video_distance > distance_threshold:
        reasons.append("distinct visual style")
    reason = "Intro detected: " + " + ".join(reasons) if reasons else "Multimodal intro detection"

    result: IntroInterval = {
        "start": 0.0,
        "end": round(intro_end, 3),
        "confidence": max(0.6, confidence),
        "reason": reason,
    }
    log.info("Intro detected: 0.0–%.1fs (conf=%.2f)", intro_end, result["confidence"])
    return result


# ── Feature extraction helpers ────────────────────────────────────────────────


def _extract_audio_segment_features(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
) -> np.ndarray | None:
    """Extract mean audio features for a time range. Returns shape (n_features,)."""
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True,
                             offset=start_sec, duration=end_sec - start_sec)
    except Exception as e:
        log.warning("Failed to load audio segment [%.0f–%.0fs]: %s", start_sec, end_sec, e)
        return None

    if len(y) < sr * 0.5:
        return None

    hop_len = sr // 2
    frame_len = sr

    rms = np.mean(librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_len, hop_length=hop_len))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=frame_len, hop_length=hop_len))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len), axis=1)

    return np.concatenate([[rms, centroid, zcr, flatness], mfccs])


def _extract_video_segment_features(
    frame_paths: list[Path],
    start_sec: float,
    end_sec: float,
    resize_to: tuple[int, int] = (320, 180),
) -> np.ndarray | None:
    """Extract mean video features for frames in [start_sec, end_sec]. Returns shape (n_features,)."""
    import cv2

    segment_frames = [
        p for p in frame_paths
        if start_sec <= _frame_timestamp(p) < end_sec
    ]

    if not segment_frames:
        return None

    features_list: list[np.ndarray] = []
    for path in segment_frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, resize_to)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        brightness = np.mean(gray)
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        color_std = np.mean([np.std(img[:, :, c]) for c in range(3)])
        pixel_variance = np.var(gray)

        # Color histogram features (mean hue, saturation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])

        features_list.append(np.array([
            brightness, edge_density, color_std, pixel_variance, mean_hue, mean_sat
        ]))

    if not features_list:
        return None

    return np.mean(features_list, axis=0)


def _feature_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compute normalized Euclidean distance between two feature vectors.

    Normalizes each dimension by the max of the two values to put features
    on comparable scales, then computes RMS of per-dimension differences.
    """
    scales = np.maximum(np.abs(features_a), np.abs(features_b))
    scales[scales < 1e-10] = 1.0
    normalized_diff = (features_a - features_b) / scales
    return float(np.sqrt(np.mean(normalized_diff ** 2)))


def _find_transition_point(
    audio_path: Path,
    frames_dir: Path,
    frame_paths: list[Path],
    intro_window: float,
    baseline_start: float,
) -> float:
    """Find the second at which features transition from intro to main content.

    Scans second-by-second within [0, intro_window] and finds where the
    features start resembling the baseline more than the intro.
    """
    import librosa

    baseline_audio = _extract_audio_segment_features(audio_path, baseline_start, baseline_start + 10)
    baseline_video = _extract_video_segment_features(frame_paths, baseline_start, baseline_start + 10)

    if baseline_audio is None or baseline_video is None:
        return intro_window

    # Scan at 1-second resolution from the end of the intro window backward
    best_end = intro_window
    min_distance = float("inf")

    for t in range(int(intro_window), int(MIN_INTRO_DURATION) - 1, -1):
        # Check if [t, t+3] resembles baseline
        check_start = float(t)
        check_end = min(check_start + 3.0, baseline_start)

        audio_check = _extract_audio_segment_features(audio_path, check_start, check_end)
        video_check = _extract_video_segment_features(frame_paths, check_start, check_end)

        if audio_check is None or video_check is None:
            continue

        audio_dist = _feature_distance(audio_check, baseline_audio)
        video_dist = _feature_distance(video_check, baseline_video)
        dist = (audio_dist + video_dist) / 2.0

        if dist < min_distance:
            min_distance = dist
            best_end = check_start

    # If we couldn't find a clear transition, use the full intro window
    if min_distance > FEATURE_DISTANCE_THRESHOLD * 0.7:
        return intro_window

    return best_end


def _frame_timestamp(path: Path) -> float:
    return int(path.stem.split("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect video intro segment.")
    parser.add_argument("audio", help="Path to audio WAV")
    parser.add_argument("frames_dir", help="Path to frames directory")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    result = detect_intro(args.audio, args.frames_dir, video_duration=args.duration)
    print(json.dumps(result, indent=2))
