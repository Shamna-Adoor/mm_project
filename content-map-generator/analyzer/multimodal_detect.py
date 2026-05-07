"""Multimodal ad detection using PELT change point segmentation + island test.

An embedded ad is a foreign segment spliced into a video such that the content
BEFORE and AFTER the ad is drawn from the same production (same show, same style)
while the ad itself is drawn from a different production. This creates a
characteristic "island" pattern in feature space.

Algorithm:
1. Extract audio + video features at 1 sample/second → F ∈ R^(T×20)
2. MAD-normalize each dimension so Euclidean distances are meaningful
3. Run PELT (globally-optimal change point detection) to partition the video
4. For each segment bounded by change points, apply the island test:
   does the content return to baseline after this segment?
5. Score islands by anomaly contrast; boost (never gate) with transcript keywords
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_AD_DURATION = 15.0
MAX_AD_DURATION = 180.0
MAX_ADS_PER_VIDEO = 8
PELT_PENALTY_FACTOR = 1.0
ISLAND_RATIO_THRESHOLD = 2.0
MIN_CONFIDENCE = 0.4
CONTEXT_WINDOW_SEC = 30
DEAD_AIR_MIN_SEC = 2.0

SPONSOR_KEYWORDS = [
    "sponsor", "sponsored", "brought to you", "partnership",
    "use code", "promo code", "discount", "coupon",
    "percent off", "% off", "free trial",
    "link in the description", "link below", "click the link",
    "check out", "sign up", "head over to",
    ".com", ".io", ".net", "http",
    "this video is sponsored", "today's sponsor",
    "thanks to", "shout out to",
]


class AdInterval(TypedDict):
    start: float
    end: float
    confidence: float


class DeadAirInterval(TypedDict):
    start: float
    end: float
    confidence: float


# ── Public API ────────────────────────────────────────────────────────────────


def detect_multimodal_ads(
    audio_path: str | Path,
    frames_dir: str | Path,
    *,
    video_duration: float = 0.0,
    transcript: list[dict] | None = None,
    min_ad_duration: float = MIN_AD_DURATION,
    max_ad_duration: float = MAX_AD_DURATION,
) -> list[AdInterval]:
    """Detect embedded ads using PELT change point detection + island test.

    Steps:
    1. Extract audio+video features resampled to a 1-second grid.
    2. MAD-normalize all dimensions for scale-invariant distances.
    3. Run PELT to find globally-optimal change points.
    4. Apply island test: segments where before ≈ after ≠ island.
    5. Score by anomaly contrast, boost with transcript keywords.
    """
    audio_path = Path(audio_path)
    frames_dir = Path(frames_dir)

    if not audio_path.exists():
        log.warning("Audio not found for multimodal detection: %s", audio_path)
        return []

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 10:
        log.info("Not enough frames for multimodal ad detection (%d)", len(frame_paths))
        return []

    log.info(
        "PELT ad detection: audio=%s, %d frames, transcript=%s",
        audio_path.name,
        len(frame_paths),
        f"{len(transcript)} segs" if transcript else "none",
    )

    audio_features, audio_ts = _extract_audio_features(audio_path)
    video_features, video_ts = _extract_video_features(frame_paths)

    if len(audio_features) < 20 or len(video_features) < 10:
        log.info("Insufficient features for detection")
        return []

    duration = video_duration or max(audio_ts[-1], video_ts[-1])
    n_seconds = int(duration) + 1

    audio_grid = _resample_to_grid(audio_features, audio_ts, n_seconds)
    video_grid = _resample_to_grid(video_features, video_ts, n_seconds)
    combined = np.hstack([audio_grid, video_grid])

    # Step 2: MAD normalization
    F_norm = _normalize_features(combined)

    # Step 3: PELT change point detection
    change_points = _pelt_change_points(F_norm)
    log.info("PELT found %d change point(s)", len(change_points))

    if len(change_points) < 2:
        log.info("Too few change points for island detection")
        return []

    # Step 4: Two-pass island detection
    # Pass 1: find islands with uncontaminated context
    pass1_candidates = _find_islands(F_norm, change_points, min_ad_duration, max_ad_duration)
    log.info("Island test pass 1: %d candidate(s)", len(pass1_candidates))

    # Pass 2: exclude pass-1 islands from context windows, re-run to find
    # ads that were hidden by adjacent-ad contamination
    all_candidates = list(pass1_candidates)
    if pass1_candidates:
        exclude_ranges = [(int(s), int(e)) for s, e, _ in pass1_candidates]
        pass2_candidates = _find_islands(
            F_norm, change_points, min_ad_duration, max_ad_duration,
            exclude_ranges=exclude_ranges,
        )
        # Add only genuinely new candidates from pass 2
        for c in pass2_candidates:
            is_new = not any(
                c[0] < k[1] and c[1] > k[0] for k in pass1_candidates
            )
            if is_new:
                all_candidates.append(c)
                log.info("Pass 2 new island: %.0f–%.0fs (ratio=%.2f)", c[0], c[1], c[2])

    # Deduplicate across both passes
    all_candidates = _deduplicate_islands(all_candidates)
    log.info("Island test total: %d candidate(s) after 2 passes", len(all_candidates))

    if not all_candidates:
        return []

    # Step 5: Boundary refinement — sharpen each island start/end
    refined_candidates: list[tuple[float, float, float]] = []
    for start, end, ratio in all_candidates:
        refined_start = _refine_boundary(F_norm, int(start))
        refined_end = _refine_boundary(F_norm, int(end))
        # Ensure start < end and duration constraints still hold
        if refined_end <= refined_start:
            refined_start, refined_end = int(start), int(end)
        dur = refined_end - refined_start
        if dur < min_ad_duration or dur > max_ad_duration:
            refined_start, refined_end = int(start), int(end)
        refined_candidates.append((float(refined_start), float(refined_end), ratio))
        if refined_start != int(start) or refined_end != int(end):
            log.info(
                "Refined %.0f–%.0fs → %.0f–%.0fs",
                start, end, refined_start, refined_end,
            )

    # Step 6: Confidence scoring from island ratio
    scored = _score_candidates(refined_candidates)

    # Step 7: Transcript boost (additive, never a gate)
    confirmed: list[AdInterval] = []
    for start, end, structural_conf in scored:
        if structural_conf < MIN_CONFIDENCE:
            continue

        linguistic_score = 0.0
        if transcript:
            linguistic_score = _transcript_sponsor_score(transcript, start, end)

        final_conf = min(1.0, structural_conf * (1.0 + 0.3 * linguistic_score))

        confirmed.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "confidence": round(final_conf, 3),
        })
        log.info(
            "Ad confirmed: %.0f–%.0fs (structural=%.2f, linguistic=%.2f, final=%.2f)",
            start, end, structural_conf, linguistic_score, final_conf,
        )

    # Cap at MAX_ADS_PER_VIDEO, keep highest confidence
    confirmed.sort(key=lambda x: x["confidence"], reverse=True)
    confirmed = confirmed[:MAX_ADS_PER_VIDEO]
    confirmed.sort(key=lambda x: x["start"])

    log.info("PELT ad detection: %d interval(s) confirmed", len(confirmed))
    return confirmed


def detect_multimodal_dead_air(
    audio_path: str | Path,
    frames_dir: str | Path,
    *,
    video_duration: float = 0.0,
    min_duration: float = DEAD_AIR_MIN_SEC,
    energy_threshold: float = 0.02,
    motion_threshold: float = 3.0,
) -> list[DeadAirInterval]:
    """Detect dead air: simultaneous audio silence + video static.

    Conservative: requires BOTH audio near-zero AND video near-static
    for at least min_duration seconds.
    """
    audio_path = Path(audio_path)
    frames_dir = Path(frames_dir)

    if not audio_path.exists():
        return []

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 4:
        return []

    audio_features, audio_ts = _extract_audio_features(audio_path)
    video_features, video_ts = _extract_video_features(frame_paths)

    if len(audio_features) == 0 or len(video_features) == 0:
        return []

    duration = video_duration or max(audio_ts[-1], video_ts[-1])
    n_seconds = int(duration) + 1

    audio_grid = _resample_to_grid(audio_features, audio_ts, n_seconds)
    video_grid = _resample_to_grid(video_features, video_ts, n_seconds)

    audio_rms = audio_grid[:, 0]
    rms_max = np.max(audio_rms) if np.max(audio_rms) > 0 else 1.0
    audio_silent = (audio_rms / rms_max) < energy_threshold

    video_variance = video_grid[:, -1]
    video_static = video_variance < motion_threshold

    dead_mask = audio_silent & video_static

    intervals: list[DeadAirInterval] = []
    in_dead = False
    start_sec = 0

    for i in range(len(dead_mask)):
        if dead_mask[i] and not in_dead:
            in_dead = True
            start_sec = i
        elif not dead_mask[i] and in_dead:
            in_dead = False
            dur = i - start_sec
            if dur >= min_duration:
                intervals.append({
                    "start": float(start_sec),
                    "end": float(i),
                    "confidence": min(1.0, round(0.7 + 0.05 * dur, 3)),
                })

    if in_dead:
        dur = len(dead_mask) - start_sec
        if dur >= min_duration:
            intervals.append({
                "start": float(start_sec),
                "end": float(len(dead_mask)),
                "confidence": min(1.0, round(0.7 + 0.05 * dur, 3)),
            })

    log.info("Multimodal dead-air: %d interval(s)", len(intervals))
    return intervals


# ── PELT Change Point Detection ───────────────────────────────────────────────


def _normalize_features(F: np.ndarray) -> np.ndarray:
    """MAD-normalize each feature dimension for scale-invariant distances.

    For each dimension j:
        F_norm[:, j] = (F[:, j] - median_j) / (1.4826 * MAD_j + eps)

    The constant 1.4826 makes MAD consistent with σ under Gaussian assumptions.
    """
    eps = 1e-10
    medians = np.median(F, axis=0)
    mad = np.median(np.abs(F - medians), axis=0)
    scale = 1.4826 * mad + eps
    return (F - medians) / scale


def _pelt_change_points(F_norm: np.ndarray) -> list[int]:
    """Run PELT on the normalized feature matrix to find globally-optimal change points."""
    import ruptures as rpt

    T, d = F_norm.shape
    penalty = np.log(T) * d * PELT_PENALTY_FACTOR

    algo = rpt.Pelt(model="l2", min_size=15, jump=1).fit(F_norm)
    bkps = algo.predict(pen=penalty)

    # ruptures returns breakpoints including the final index T; remove it
    change_points = [bp for bp in bkps if bp < T]

    return change_points


def _find_islands(
    F_norm: np.ndarray,
    change_points: list[int],
    min_duration: float,
    max_duration: float,
    exclude_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[float, float, float]]:
    """Find island segments: bounded regions where before ≈ after ≠ island.

    Tests single segments AND multi-segment spans (up to 3 consecutive PELT
    segments as one island). Deduplicates overlapping candidates by keeping
    the one with the highest ratio.

    If exclude_ranges is provided, those time ranges are masked out of context
    windows (used in the second pass to avoid contamination by already-detected ads).

    Returns (start, end, ratio) tuples where ratio is the island contrast ratio.
    """
    T = len(F_norm)
    eps = 1e-10

    boundaries = sorted(set([0] + change_points + [T]))
    raw_candidates: list[tuple[float, float, float]] = []

    # Test spans of 1, 2, and 3 consecutive PELT segments as single island candidates
    for span in range(1, 4):
        for idx in range(1, len(boundaries) - span):
            island_start = boundaries[idx]
            island_end = boundaries[idx + span]

            duration = island_end - island_start
            if duration < min_duration or duration > max_duration:
                continue

            # Build context, excluding known islands if provided
            before_ctx = _get_clean_context(
                F_norm, island_start - CONTEXT_WINDOW_SEC, island_start,
                exclude_ranges,
            )
            after_ctx = _get_clean_context(
                F_norm, island_end, island_end + CONTEXT_WINDOW_SEC,
                exclude_ranges,
            )
            island_ctx = F_norm[island_start:island_end]

            if len(before_ctx) < 5 or len(island_ctx) < 5 or len(after_ctx) < 5:
                continue

            mu_before = np.mean(before_ctx, axis=0)
            mu_island = np.mean(island_ctx, axis=0)
            mu_after = np.mean(after_ctx, axis=0)

            d_before_after = np.linalg.norm(mu_before - mu_after)
            d_before_island = np.linalg.norm(mu_before - mu_island)
            d_after_island = np.linalg.norm(mu_after - mu_island)

            avg_island_distance = (d_before_island + d_after_island) / 2.0
            ratio = avg_island_distance / (d_before_after + eps)

            if ratio >= ISLAND_RATIO_THRESHOLD:
                raw_candidates.append((float(island_start), float(island_end), ratio))

    # Deduplicate overlapping candidates: keep highest ratio for each time region
    candidates = _deduplicate_islands(raw_candidates)

    for start, end, ratio in candidates:
        log.info(
            "Island candidate: %.0f–%.0fs (ratio=%.2f)",
            start, end, ratio,
        )

    return candidates


def _get_clean_context(
    F_norm: np.ndarray,
    start: float,
    end: float,
    exclude_ranges: list[tuple[int, int]] | None,
) -> np.ndarray:
    """Get feature rows in [start, end), excluding any ranges in exclude_ranges."""
    T = len(F_norm)
    lo = max(0, int(start))
    hi = min(T, int(end))

    if lo >= hi:
        return np.empty((0, F_norm.shape[1]))

    if not exclude_ranges:
        return F_norm[lo:hi]

    mask = np.ones(hi - lo, dtype=bool)
    for ex_start, ex_end in exclude_ranges:
        # Overlap with [lo, hi)
        overlap_lo = max(ex_start, lo) - lo
        overlap_hi = min(ex_end, hi) - lo
        if overlap_lo < overlap_hi:
            mask[overlap_lo:overlap_hi] = False

    return F_norm[lo:hi][mask]


def _deduplicate_islands(
    candidates: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Remove overlapping island candidates, keeping the highest-ratio one."""
    if not candidates:
        return []

    # Sort by ratio descending — greedily keep non-overlapping
    sorted_cands = sorted(candidates, key=lambda x: x[2], reverse=True)
    kept: list[tuple[float, float, float]] = []

    for start, end, ratio in sorted_cands:
        overlaps = False
        for k_start, k_end, _ in kept:
            if start < k_end and end > k_start:
                overlaps = True
                break
        if not overlaps:
            kept.append((start, end, ratio))

    return sorted(kept, key=lambda x: x[0])


def _score_candidates(
    candidates: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Score each island candidate by its contrast ratio.

    The island ratio directly encodes structural confidence: passing the
    threshold (2.0) means it IS an island; the ratio magnitude tells us
    how clearly separated it is. Confidence maps ratio to [0, 1]:
      ratio=2.0 → conf=0.5 (minimum passing island)
      ratio=5.0+ → conf=1.0 (extremely clear island)
    """
    scored: list[tuple[float, float, float]] = []
    for start, end, ratio in candidates:
        structural_conf = float(np.clip(
            0.5 + 0.5 * (ratio - ISLAND_RATIO_THRESHOLD) / 3.0,
            0.0, 1.0,
        ))
        scored.append((start, end, structural_conf))
        log.info("Scoring %.0f–%.0fs: ratio=%.2f → conf=%.2f", start, end, ratio, structural_conf)

    return scored


def _refine_boundary(F_norm: np.ndarray, coarse_t: int, search_radius: int = 5) -> int:
    """Find the exact transition point near coarse_t by maximum feature derivative.

    Searches +/-search_radius seconds around the coarse boundary and returns
    the time index where the largest single-second feature jump occurs.
    """
    T = len(F_norm)
    lo = max(0, coarse_t - search_radius)
    hi = min(T - 1, coarse_t + search_radius)

    if hi - lo < 2:
        return coarse_t

    segment = F_norm[lo:hi + 1]
    diffs = np.linalg.norm(np.diff(segment, axis=0), axis=1)

    best_offset = int(np.argmax(diffs))
    return lo + best_offset


# ── Transcript Analysis ───────────────────────────────────────────────────────


def _transcript_sponsor_score(
    transcript: list[dict],
    start: float,
    end: float,
) -> float:
    """Score how likely the transcript in [start, end] contains sponsor language.

    Returns 0.0–1.0 where higher means more sponsor keywords detected.
    """
    texts: list[str] = []
    for seg in transcript:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        if seg_start < end and seg_end > start:
            text = seg.get("text", "").strip()
            if text:
                texts.append(text.lower())

    if not texts:
        return 0.0

    combined_text = " ".join(texts)
    hits = sum(1 for kw in SPONSOR_KEYWORDS if kw in combined_text)

    if hits == 0:
        return 0.0
    elif hits == 1:
        return 0.3
    elif hits == 2:
        return 0.5
    elif hits <= 4:
        return 0.7
    else:
        return min(1.0, 0.7 + hits * 0.05)


# ── Feature Extraction ────────────────────────────────────────────────────────


def _extract_audio_features(
    audio_path: Path,
    hop_duration: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-second audio features: [RMS, spectral_centroid, zcr, mfcc_mean x13]."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop_len = int(hop_duration * sr)
    frame_len = hop_len * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_len, hop_length=hop_len
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_len, hop_length=hop_len
    )[0]
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len
    )

    n_frames = min(len(rms), len(centroid), len(zcr), mfccs.shape[1])
    features = np.column_stack([
        rms[:n_frames],
        centroid[:n_frames],
        zcr[:n_frames],
        mfccs[:, :n_frames].T,
    ])

    timestamps = np.arange(n_frames) * hop_duration
    log.info("Audio features: %d frames (%.1fs hop)", n_frames, hop_duration)
    return features, timestamps


def _extract_video_features(
    frame_paths: list[Path],
    resize_to: tuple[int, int] = (160, 90),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-frame video features: [brightness, edge_density, color_std, pixel_variance]."""
    import cv2

    features_list: list[list[float]] = []
    timestamps: list[float] = []

    for path in frame_paths:
        ts = _frame_timestamp(path)
        timestamps.append(ts)

        img = cv2.imread(str(path))
        if img is None:
            features_list.append([0.0, 0.0, 0.0, 0.0])
            continue

        img = cv2.resize(img, resize_to)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        brightness = float(np.mean(gray))
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        color_std = float(np.mean([np.std(img[:, :, c]) for c in range(3)]))
        pixel_variance = float(np.var(gray))

        features_list.append([brightness, edge_density, color_std, pixel_variance])

    return np.array(features_list, dtype=np.float64), np.array(timestamps, dtype=np.float64)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resample_to_grid(
    features: np.ndarray,
    timestamps: np.ndarray,
    n_seconds: int,
) -> np.ndarray:
    """Resample features onto a regular 1-second grid via nearest-neighbor."""
    n_features = features.shape[1]
    grid = np.zeros((n_seconds, n_features), dtype=np.float64)

    for sec in range(n_seconds):
        idx = np.argmin(np.abs(timestamps - sec))
        grid[sec] = features[idx]

    return grid


def _frame_timestamp(path: Path) -> float:
    return int(path.stem.split("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PELT-based multimodal ad detection.")
    parser.add_argument("audio", help="Path to audio WAV")
    parser.add_argument("frames_dir", help="Path to frames directory")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--transcript", type=str, default=None, help="Path to transcript JSON")
    args = parser.parse_args()

    transcript_data = None
    if args.transcript:
        with open(args.transcript) as f:
            transcript_data = json.load(f)

    ads = detect_multimodal_ads(
        args.audio, args.frames_dir,
        video_duration=args.duration,
        transcript=transcript_data,
    )
    print(json.dumps(ads, indent=2))
