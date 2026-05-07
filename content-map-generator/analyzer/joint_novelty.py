"""Joint-novelty detection: find regions whose joint signature differs from the rest of the video.

This module asks the universal question every other detector dodges:

    "Does this region look unlike the rest of THIS video?"

That is the only property genuinely shared by every inserted ad regardless of
medium (podcast, animation, broadcast, gaming, music video). All other
detectors in this codebase are *positive matchers* — "does this look like a
rapid-cut burst?", "does this contain commercial keywords?", "does the HSV
histogram shift?". Each is brittle on content that doesn't fit its template.

Joint novelty is template-free. It builds ONE feature vector per N-second
window covering audio spectral shape, color distribution, editing structure
and transcript characteristics, robust-standardises each feature against
the whole-video median, and emits a single L2 distance per window. Regions
whose distance is far above the per-video median become candidate
non-content intervals. Because the threshold is computed FROM this video,
a chatty podcast and a high-action animated film both calibrate themselves.

Design notes
------------
• Defensive: any missing dependency (librosa, opencv, numpy) downgrades to
  empty output rather than crashing the pipeline.
• Cheap: reuses the audio file and frame JPEGs the pipeline already
  produces. Adds ~10–20 s on a 30-min video on commodity hardware.
• Additive: the existing detectors are untouched. Novelty is just one more
  signal source with high confidence; the cap layer handles overflow.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)

_FRAME_TS_RE = re.compile(r"frame_(\d{10})\.jpg$")


class NoveltyRegion(TypedDict):
    start:         float
    end:           float
    score:         float           # mean standardised distance inside the region
    n_windows:     int             # number of contiguous flagged windows
    dominant_axes: list[str]       # which features deviated most (debug aid)


def detect_novelty_regions(
    audio_path:        str | Path,
    frames_dir:        str | Path | None,
    scene_changes:     list[dict],
    transcript:        list[dict],
    video_duration:    float,
    *,
    window_seconds:    float = 5.0,
    novelty_sigmas:    float = 2.5,
    min_duration:      float = 8.0,
    pad_before:        float = 1.0,
    pad_after:         float = 2.0,
    smooth_windows:    int   = 3,
    sample_rate:       int   = 16000,
    intro_cutoff:      float = 0.0,
    outro_cutoff:      float = 0.0,
) -> list[NoveltyRegion]:
    """Return regions whose joint signature differs from the per-video baseline.

    Steps
    -----
    1. Slice the video into ``window_seconds`` windows (~360 windows for 30 min).
    2. Per window, extract:
         audio:      RMS, spectral centroid, rolloff, flatness, ZCR, MFCC 1-5
         visual:     mean H/S/V, std H/S/V, 4×4×4 HSV histogram bins (64)
         editing:    scene-cut count, words spoken, mean Whisper avg_logprob
    3. Robust-standardise each feature across the whole video using
       median + 1.4826·MAD. Resists outlier influence and brings every
       feature to the same scale.
    4. Per window, novelty = L2 norm of the standardised feature vector.
       This is a Mahalanobis-style distance under the assumption that
       feature correlations are small after standardisation. (Empirically
       sufficient; full covariance inversion is unnecessary at this scale
       and adds numerical fragility.)
    5. Smooth distances with a 3-window rolling median to suppress single-
       window spikes.
    6. Threshold = video median + k · MAD of the distances. Flag windows
       above threshold; merge consecutive flagged windows (allowing 1-window
       gaps) into regions; drop regions shorter than ``min_duration``.

    The detector does NOT label regions (intro vs sponsor vs outro). It only
    reports "this is unlike the rest". The caller is expected to combine
    the result with positional rules and the LLM transcript classifier
    (which already handle intro/outro/sponsor distinction).
    """
    # ── Defensive imports — never let a missing dep break the pipeline ────
    try:
        import numpy as np
    except ImportError as exc:
        log.warning("Joint novelty skipped — numpy unavailable: %s", exc)
        return []

    if video_duration < window_seconds * 6:
        log.info("Video too short for novelty analysis (need ≥ %.0fs)", window_seconds * 6)
        return []

    n_windows = int(video_duration // window_seconds)
    log.info(
        "Computing joint-novelty features over %d windows of %.0fs each…",
        n_windows, window_seconds,
    )

    # ── 1) Audio features ───────────────────────────────────────────────────
    audio_feats, audio_names = _audio_features(audio_path, n_windows, window_seconds, sample_rate)
    if audio_feats is None:
        log.warning("Joint novelty skipped — audio feature extraction failed")
        return []

    # ── 2) Visual features (optional — pipeline can run with --skip-visual) ─
    visual_feats, visual_names = _visual_features(frames_dir, n_windows, window_seconds)

    # ── 3) Structural features (cheap, always available) ────────────────────
    cut_count = np.zeros(n_windows, dtype=np.float32)
    for sc in scene_changes:
        try:
            t = float(sc["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        idx = int(t // window_seconds)
        if 0 <= idx < n_windows:
            cut_count[idx] += 1.0

    speech_rate  = np.zeros(n_windows, dtype=np.float32)
    avg_logprob  = np.full(n_windows, np.nan, dtype=np.float32)
    for seg in transcript:
        try:
            s = float(seg["start"]); e = float(seg["end"])
            text = seg.get("text", "") or ""
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        n_words = len(text.split())
        seg_dur = max(1e-6, e - s)
        for idx in range(int(s // window_seconds), int(e // window_seconds) + 1):
            if 0 <= idx < n_windows:
                w_start = idx * window_seconds
                w_end   = w_start + window_seconds
                overlap = max(0.0, min(e, w_end) - max(s, w_start))
                speech_rate[idx] += n_words * (overlap / seg_dur) / window_seconds
        # Whisper per-segment avg_logprob: anchor to segment midpoint window
        lp_raw = seg.get("avg_logprob")
        if lp_raw is not None:
            try:
                lp = float(lp_raw)
                idx = int(((s + e) / 2.0) // window_seconds)
                if 0 <= idx < n_windows:
                    if np.isnan(avg_logprob[idx]):
                        avg_logprob[idx] = lp
                    else:
                        avg_logprob[idx] = (avg_logprob[idx] + lp) / 2.0
            except (TypeError, ValueError):
                pass

    if np.any(np.isnan(avg_logprob)):
        # Fill missing logprobs with the median observed value (treat
        # unknown segments as "average confidence", neutral signal).
        if np.any(~np.isnan(avg_logprob)):
            fill_val = float(np.nanmedian(avg_logprob))
        else:
            fill_val = 0.0
        avg_logprob = np.where(np.isnan(avg_logprob), fill_val, avg_logprob)

    structural = np.stack([cut_count, speech_rate, avg_logprob], axis=1)
    structural_names = ["x_cuts", "x_words_per_s", "x_logprob"]

    # ── 4) Concatenate full feature matrix ─────────────────────────────────
    parts: list = [audio_feats]
    feat_names: list[str] = list(audio_names)
    if visual_feats is not None:
        parts.append(visual_feats)
        feat_names += list(visual_names)
    parts.append(structural)
    feat_names += structural_names
    F = np.concatenate(parts, axis=1).astype(np.float32)
    log.info("Joint-novelty feature matrix: %d windows × %d features", F.shape[0], F.shape[1])

    # ── 5) Robust per-feature standardisation ──────────────────────────────
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0)
    scaled_mad = mad * 1.4826
    scaled_mad = np.where(scaled_mad < 1e-9, 1.0, scaled_mad)   # guard div-by-zero
    Z = (F - med) / scaled_mad

    # ── 6) Per-modality norms, harmonic-mean joint score ──────────────────
    # The principle: an inserted ad differs from the host along multiple
    # modalities simultaneously. We compute one L2 norm per modality
    # group, then combine with a harmonic mean. The harmonic mean is
    # SMALL whenever ANY of its inputs is small — exactly the behaviour
    # we want for joint detection. Single-modality outliers (slide
    # changes, action scene cuts) get suppressed; regions where audio
    # AND visual both shift get amplified.
    audio_dim     = audio_feats.shape[1]
    visual_dim    = visual_feats.shape[1] if visual_feats is not None else 0
    structural_dim = structural.shape[1]

    audio_idx_hi  = audio_dim
    visual_idx_hi = audio_dim + visual_dim
    struct_idx_hi = audio_dim + visual_dim + structural_dim

    audio_norm = np.linalg.norm(Z[:, 0:audio_idx_hi], axis=1) / np.sqrt(max(1, audio_dim))
    if visual_dim > 0:
        visual_norm = (
            np.linalg.norm(Z[:, audio_idx_hi:visual_idx_hi], axis=1) / np.sqrt(visual_dim)
        )
    else:
        visual_norm = np.zeros(n_windows, dtype=np.float32)
    struct_norm = (
        np.linalg.norm(Z[:, visual_idx_hi:struct_idx_hi], axis=1)
        / np.sqrt(max(1, structural_dim))
    )

    eps = 1e-3
    if visual_dim > 0:
        # Harmonic mean of audio and visual: tends towards the smaller of
        # the two, suppressing single-modality novelty.
        joint = 2.0 * audio_norm * visual_norm / (audio_norm + visual_norm + eps)
    else:
        joint = audio_norm
    dist = (joint + 0.25 * struct_norm).astype(np.float32)

    # ── 7) Smooth to suppress single-window spikes ─────────────────────────
    if smooth_windows > 1 and n_windows >= smooth_windows:
        smoothed = np.copy(dist)
        half = smooth_windows // 2
        for i in range(n_windows):
            lo = max(0, i - half)
            hi = min(n_windows, i + half + 1)
            smoothed[i] = float(np.median(dist[lo:hi]))
        dist = smoothed

    # ── 8) Video-relative threshold ────────────────────────────────────────
    d_med = float(np.median(dist))
    d_mad = float(np.median(np.abs(dist - d_med))) * 1.4826
    d_mad = max(d_mad, 1e-3)
    threshold = d_med + novelty_sigmas * d_mad
    log.info(
        "Joint novelty threshold: %.3f (median=%.3f, MAD=%.3f, k=%.1f) "
        "[audio_dim=%d visual_dim=%d struct_dim=%d]",
        threshold, d_med, d_mad, novelty_sigmas,
        audio_dim, visual_dim, structural_dim,
    )
    joint_flagged = dist > threshold

    # ── 9) Mark, merge, emit ────────────────────────────────────────────────
    flagged = joint_flagged
    regions: list[NoveltyRegion] = []
    i = 0
    while i < n_windows:
        if not flagged[i]:
            i += 1
            continue
        j = i
        # Allow 1-window holes inside a region for short transitional frames.
        while j < n_windows and (flagged[j] or (j + 1 < n_windows and flagged[j + 1])):
            j += 1

        window_scores = dist[i:j]
        z_window      = Z[i:j]
        if len(z_window):
            axis_means = np.mean(np.abs(z_window), axis=0)
            top_idx    = np.argsort(axis_means)[-3:][::-1]
            dominant   = [feat_names[k] for k in top_idx if axis_means[k] > 1.0]
        else:
            dominant = []

        start_t = max(0.0, i * window_seconds - pad_before)
        end_t   = min(video_duration, j * window_seconds + pad_after)
        dur     = end_t - start_t

        # Note: we deliberately do NOT positional-filter here. The
        # downstream LLM segment overlay preserves correctly-detected
        # intro/outro labels because forced intervals can only split
        # main_content, not other LLM labels. So a novelty region landing
        # inside the LLM intro window is safely absorbed; one landing
        # outside is correctly promoted to sponsor.
        if dur >= min_duration:
            regions.append({
                "start":         round(start_t, 3),
                "end":           round(end_t,   3),
                "score":         round(float(np.mean(window_scores)), 3),
                "n_windows":     int(j - i),
                "dominant_axes": dominant,
            })
        i = j + 1

    log.info(
        "Joint-novelty: %d region(s): %s",
        len(regions),
        [(round(r["start"], 1), round(r["end"], 1), r["score"]) for r in regions],
    )
    return regions


# ── Feature extractors ────────────────────────────────────────────────────────

def _audio_features(audio_path, n_windows, window_seconds, sample_rate):
    """Return ((n_windows, 10) ndarray, list[str]) of audio features, or (None, [])."""
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        log.warning("Audio features skipped — librosa unavailable: %s", exc)
        return None, []

    audio_path = Path(audio_path)
    if not audio_path.exists():
        log.warning("Audio path does not exist: %s", audio_path)
        return None, []

    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as exc:
        log.warning("Audio load failed: %s", exc)
        return None, []

    win_len = int(window_seconds * sr)
    feats = np.zeros((n_windows, 10), dtype=np.float32)
    names = ["a_rms", "a_centroid", "a_rolloff", "a_flatness", "a_zcr",
             "a_mfcc1", "a_mfcc2", "a_mfcc3", "a_mfcc4", "a_mfcc5"]
    for i in range(n_windows):
        s = i * win_len
        e = min(len(y), (i + 1) * win_len)
        chunk = y[s:e]
        if len(chunk) < win_len // 4:
            continue   # too few samples — leave row at zeros, standardisation handles
        try:
            feats[i, 0] = float(np.sqrt(np.mean(chunk ** 2)))
            feats[i, 1] = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
            feats[i, 2] = float(np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr)))
            feats[i, 3] = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
            feats[i, 4] = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=5)
            feats[i, 5:10] = np.mean(mfcc, axis=1)
        except Exception as exc:
            log.debug("Audio feature extraction skipped at window %d: %s", i, exc)
            continue
    return feats, names


def _visual_features(frames_dir, n_windows, window_seconds):
    """Return ((n_windows, 7) ndarray, list[str]) of visual features, or (None, []).

    Features per window
    -------------------
    - v_h_mean, v_s_mean, v_v_mean   mean H/S/V across frames in window
    - v_h_std,  v_s_std,  v_v_std    std H/S/V across frames in window
    - v_hist_dist                    Bhattacharyya distance from this window's
                                     mean HSV histogram to the per-video median
                                     histogram (scalar, captures distribution
                                     shape without flooding the feature vector
                                     with 64 correlated bins)
    """
    if frames_dir is None:
        return None, []

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        log.warning("Visual features skipped — opencv unavailable: %s", exc)
        return None, []

    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(paths) < 8:
        log.info("Visual features skipped — fewer than 8 extracted frames")
        return None, []

    # Per-window summary stats (6) + histograms accumulated for distance pass.
    summary = np.zeros((n_windows, 6), dtype=np.float32)
    n_hist_bins = 4 * 4 * 4
    histograms  = np.zeros((n_windows, n_hist_bins), dtype=np.float32)
    counts = np.zeros(n_windows, dtype=np.int32)

    for fp in paths:
        ts = _parse_frame_ts(fp.name)
        if ts is None:
            continue
        idx = int(ts // window_seconds)
        if not (0 <= idx < n_windows):
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        summary[idx, 0] += float(np.mean(hsv[..., 0]))
        summary[idx, 1] += float(np.mean(hsv[..., 1]))
        summary[idx, 2] += float(np.mean(hsv[..., 2]))
        summary[idx, 3] += float(np.std(hsv[..., 0]))
        summary[idx, 4] += float(np.std(hsv[..., 1]))
        summary[idx, 5] += float(np.std(hsv[..., 2]))
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        histograms[idx] += hist.flatten()
        counts[idx] += 1

    counts_safe = np.maximum(counts, 1).reshape(-1, 1)
    summary    /= counts_safe
    histograms /= counts_safe
    # Re-normalise per-window histograms to a probability distribution.
    h_sum = histograms.sum(axis=1, keepdims=True)
    h_sum = np.where(h_sum < 1e-9, 1.0, h_sum)
    histograms /= h_sum

    # Compute per-video baseline (median histogram) and per-window
    # Bhattacharyya distance to it. One scalar per window — scales like every
    # other feature and avoids the 64-correlated-dimensions explosion.
    baseline = np.median(histograms, axis=0).astype(np.float32)
    if float(np.sum(baseline)) > 0:
        baseline = baseline / float(np.sum(baseline))
    hist_dist = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        # Bhattacharyya: 1 - Σ √(p·q). Bounded in [0, 1].
        bc = float(np.sum(np.sqrt(np.maximum(0.0, histograms[i] * baseline))))
        hist_dist[i] = 1.0 - bc

    feats = np.concatenate([summary, hist_dist.reshape(-1, 1)], axis=1)
    names = ["v_h_mean", "v_s_mean", "v_v_mean",
             "v_h_std",  "v_s_std",  "v_v_std",
             "v_hist_dist"]
    return feats, names


def _parse_frame_ts(name: str) -> float | None:
    m = _FRAME_TS_RE.search(name)
    if not m:
        return None
    return int(m.group(1)) / 1000.0


# ── CLI for ad-hoc testing on cached intermediate data ────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint-novelty detection.")
    parser.add_argument("audio_path", help="Path to extracted .wav")
    parser.add_argument("frames_dir", help="Directory of frame_*.jpg files")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--sigmas", type=float, default=2.5)
    args = parser.parse_args()

    out = detect_novelty_regions(
        args.audio_path, args.frames_dir, [], [], args.duration,
        window_seconds=args.window_seconds, novelty_sigmas=args.sigmas,
    )
    print(json.dumps(out, indent=2))
