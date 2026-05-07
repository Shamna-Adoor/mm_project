"""Majority-voting ad splice detector.

Detects embedded ads by finding boundary pairs (splice-in + splice-out) where
multiple independent signals agree that a production change occurred.

Signals (8 total, 3 audio + 3 video + 2 transcript):
  Audio:  LUFS jump, tone change, speaker change
  Video:  color temperature shift, pixel diff spike, text/brand OCR
  Transcript: topic discontinuity, speech gap + sponsor keywords

Architecture:
  Phase 1 (cheap): LUFS, pixel diff, color shift, tone change, speech gap, keywords
  Phase 2 (targeted): speaker change, OCR, topic discontinuity — only at ambiguous boundaries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from analyzer._logging import get_logger

log = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MIN_AD_DURATION = 15.0
MAX_AD_DURATION = 180.0
MAX_ADS_PER_VIDEO = 6

# Voting thresholds
BOUNDARY_CONFIRM_THRESHOLD = 1.0  # weighted sum for a boundary to be a candidate
PAIR_CONFIRM_THRESHOLD = 2.8      # combined boundary scores for a valid pair

# Signal weights (per-signal, used within modality scoring)
WEIGHTS = {
    "lufs_jump": 1.0,
    "tone_change": 0.7,
    "speaker_change": 1.0,
    "color_shift": 0.8,
    "pixel_diff": 1.2,
    "text_brand": 1.5,
    "topic_jump": 0.8,
    "speech_gap": 0.6,
    "sponsor_keywords": 1.2,
}

# Modality groupings — signals within a modality are correlated,
# so we use max() within and sum() across modalities (LLR framework)
MODALITY_GROUPS = {
    "audio": ["lufs_jump", "tone_change", "speaker_change"],
    "video": ["pixel_diff", "color_shift", "text_brand"],
    "transcript": ["topic_jump", "speech_gap", "sponsor_keywords"],
}

# Cross-modal bonus: reward when multiple independent modalities fire simultaneously
CROSS_MODAL_BONUS = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}

# Cross-modal pair corroboration bonus (at both boundaries)
CROSS_MODAL_PAIR_BONUS = {0: 0.0, 1: 0.0, 2: 0.8, 3: 1.5}

MODALITY_FIRE_THRESHOLD = 0.3

SPONSOR_KEYWORDS = [
    "sponsor", "sponsored", "brought to you", "partnership",
    "use code", "promo code", "discount", "coupon",
    "percent off", "% off", "free trial",
    "link in the description", "link below", "click the link",
    "check out", "sign up", "head over to",
    ".com", ".io", ".net", "http",
    "this video is sponsored", "today's sponsor",
    "thanks to", "shout out to",
    "download the app", "available at", "order now",
    "subscribe", "click here", "limited time",
    "instacart", "instacard", "bosch",
]

# Boundary detection: a signal "fires" when its score exceeds this
SIGNAL_FIRE_THRESHOLD = 0.5


class AdInterval(TypedDict):
    start: float
    end: float
    confidence: float
    signals: dict[str, float]


# ── Public API ────────────────────────────────────────────────────────────────


def detect_ads_voting(
    audio_path: str | Path,
    frames_dir: str | Path,
    *,
    video_duration: float = 0.0,
    transcript: list[dict] | None = None,
    scene_changes: list[dict] | None = None,
) -> list[AdInterval]:
    """Detect embedded ads using majority-voting across multiple signals.

    Phase 1: Run cheap signals, confirm boundaries where 3+ agree.
    Phase 2: For ambiguous boundaries, run expensive targeted analysis.

    Parameters
    ----------
    scene_changes : list of {"timestamp": float, "confidence": float}
        Pre-computed hard cuts from PySceneDetect (full-framerate, precise).
        When provided, used as the primary pixel_diff signal instead of
        sparse frame comparison.
    """
    audio_path = Path(audio_path)
    frames_dir = Path(frames_dir)

    if not audio_path.exists():
        log.warning("Audio not found: %s", audio_path)
        return []

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) < 4:
        log.info("Not enough frames for splice detection (%d)", len(frame_paths))
        return []

    # Determine video duration
    duration = video_duration
    if duration <= 0:
        duration = _frame_timestamp(frame_paths[-1]) + 2.0
    n_seconds = int(duration) + 1

    log.info(
        "Splice detection: audio=%s, %d frames, duration=%.0fs, transcript=%s",
        audio_path.name, len(frame_paths), duration,
        f"{len(transcript)} segs" if transcript else "none",
    )

    # ── Phase 1: Cheap signals ────────────────────────────────────────────────
    signals: dict[str, np.ndarray] = {}

    # Audio signals
    signals["lufs_jump"] = _detect_lufs_jumps(audio_path, n_seconds)
    signals["tone_change"] = _detect_tone_changes(audio_path, n_seconds)

    # Video signals (also build frame histogram index for visual return check)
    frame_histograms = _build_frame_histogram_index(frame_paths, n_seconds)

    # Use PySceneDetect results (full-framerate) for pixel_diff when available
    if scene_changes:
        signals["pixel_diff"] = _scene_changes_to_signal(scene_changes, n_seconds)
    else:
        signals["pixel_diff"] = _detect_pixel_diff_spikes(frame_paths, n_seconds)

    signals["color_shift"] = _detect_color_shifts(frame_paths, n_seconds)

    # Transcript signals
    if transcript:
        signals["speech_gap"] = _detect_speech_gaps(transcript, n_seconds)
        signals["sponsor_keywords"] = _detect_sponsor_keywords(transcript, n_seconds)
    else:
        signals["speech_gap"] = np.zeros(n_seconds)
        signals["sponsor_keywords"] = np.zeros(n_seconds)

    # ── Find boundary candidates ─────────────────────────────────────────────
    # Strategy: use hard cuts (pixel_diff) as primary boundary candidates,
    # then score each candidate by how many other signals corroborate it.
    # Also consider non-hard-cut boundaries where 3+ other signals fire.
    boundary_scores = _compute_weighted_boundary_scores(signals, WEIGHTS)
    boundaries = _find_boundary_peaks(boundary_scores, min_distance=8)

    log.info("Phase 1: %d boundary candidates found", len(boundaries))

    if len(boundaries) < 2:
        log.info("Too few boundaries for ad detection")
        return []

    # ── Phase 2: Expensive signals (targeted at top candidates) ───────────────
    top_boundary_times = [t for t, score in boundaries[:30]]

    if top_boundary_times and transcript:
        topic_signal = _detect_topic_discontinuity(transcript, n_seconds)
        signals["topic_jump"] = topic_signal
        boundary_scores = _compute_weighted_boundary_scores(signals, WEIGHTS)
        boundaries = _find_boundary_peaks(boundary_scores, min_distance=8)
        log.info("Phase 2 (topic): updated to %d boundary candidates", len(boundaries))

    # Speaker change — only run on targeted windows
    ambiguous_times = [t for t, score in boundaries if 1.5 <= score < BOUNDARY_CONFIRM_THRESHOLD]
    if ambiguous_times:
        speaker_signal = _detect_speaker_changes_targeted(
            audio_path, ambiguous_times[:20], n_seconds,
        )
        if speaker_signal is not None:
            signals["speaker_change"] = speaker_signal
            boundary_scores = _compute_weighted_boundary_scores(signals, WEIGHTS)
            boundaries = _find_boundary_peaks(boundary_scores, min_distance=8)
            log.info("Phase 2 (speaker): updated to %d boundaries", len(boundaries))

    # Text/Brand OCR — only on frames near hard cuts
    hard_cut_times = [t for t, _ in boundaries[:20] if signals["pixel_diff"][int(t)] > 0.3]
    if hard_cut_times and frame_paths:
        text_signal = _detect_text_brands_targeted(
            frame_paths, hard_cut_times, n_seconds,
        )
        if text_signal is not None:
            signals["text_brand"] = text_signal
            boundary_scores = _compute_weighted_boundary_scores(signals, WEIGHTS)
            boundaries = _find_boundary_peaks(boundary_scores, min_distance=8)

    # ── Pair boundaries into ad intervals ─────────────────────────────────────
    # Use ALL boundary candidates (not just confirmed) for pairing.
    # The pair scoring will determine which are real ads.
    log.info("Pairing from %d boundary candidates", len(boundaries))
    ads = _pair_boundaries_to_ads(boundaries, signals, n_seconds, frame_histograms)

    # Filter low-confidence detections
    ads = [a for a in ads if a["confidence"] >= 0.55]

    # Merge overlapping or adjacent detections
    ads = _merge_adjacent_ads(ads)

    # Sort by confidence and cap
    ads.sort(key=lambda x: x["confidence"], reverse=True)
    ads = ads[:MAX_ADS_PER_VIDEO]
    ads.sort(key=lambda x: x["start"])

    log.info("Voting detector: %d ad(s) confirmed", len(ads))
    return ads


# ── Phase 1: Cheap Signal Detectors ──────────────────────────────────────────


def _detect_lufs_jumps(audio_path: Path, n_seconds: int) -> np.ndarray:
    """Detect abrupt loudness (LUFS) step changes in the audio.

    Computes a smoothed RMS energy curve (5-second windows), then detects
    abrupt step changes (> 6 dB) that indicate a production change.
    """
    import soundfile as sf

    try:
        data, rate = sf.read(str(audio_path))
    except Exception as e:
        log.warning("Failed to read audio for LUFS: %s", e)
        return np.zeros(n_seconds)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Compute RMS energy per second
    window_samples = int(rate * 1.0)
    rms_per_sec = np.zeros(n_seconds)
    for sec in range(n_seconds):
        start = sec * window_samples
        end = min(start + window_samples, len(data))
        if end - start < rate // 4:
            continue
        block = data[start:end]
        rms = float(np.sqrt(np.mean(block ** 2)))
        rms_per_sec[sec] = max(rms, 1e-10)

    # Convert to dB
    rms_db = 20.0 * np.log10(rms_per_sec + 1e-10)

    # Smooth with 5-second running average to remove per-second jitter
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(rms_db, kernel, mode="same")

    # Detect step changes: compare 5s window before vs 5s window after each point
    signal = np.zeros(n_seconds)
    half_win = 5
    for t in range(half_win, n_seconds - half_win):
        before = np.mean(smoothed[t - half_win:t])
        after = np.mean(smoothed[t:t + half_win])
        jump_db = abs(after - before)
        # Only fire for jumps > 4 dB (perceptually significant)
        if jump_db > 4.0:
            signal[t] = min(1.0, (jump_db - 4.0) / 6.0)

    # Keep only the top peaks (max ~20 strongest step changes)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(signal, distance=15, height=0.3, prominence=0.2)
    filtered = np.zeros(n_seconds)
    if len(peaks) > 0:
        # Keep only top 20 by height
        heights = properties["peak_heights"]
        top_idx = np.argsort(heights)[-20:]
        for idx in top_idx:
            filtered[peaks[idx]] = signal[peaks[idx]]

    # Slight smear ±2s
    filtered = _smear_signal(filtered, radius=2)

    n_spikes = int(np.sum(filtered > SIGNAL_FIRE_THRESHOLD))
    log.info("LUFS: %d step change(s) detected", n_spikes)
    return filtered


def _detect_tone_changes(audio_path: Path, n_seconds: int) -> np.ndarray:
    """Detect spectral envelope changes (room tone / production style shifts).

    Computes MFCCs per second and measures L2 distance between adjacent windows.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception as e:
        log.warning("Failed to load audio for tone detection: %s", e)
        return np.zeros(n_seconds)

    hop_len = sr  # 1 second
    frame_len = sr * 2

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len)
    n_frames = min(n_seconds, mfccs.shape[1])

    # L2 distance between adjacent MFCC vectors
    diffs = np.zeros(n_seconds)
    for i in range(1, n_frames):
        diffs[i] = float(np.linalg.norm(mfccs[:, i] - mfccs[:, i - 1]))

    # Percentile-based normalization: only top 5% of changes are interesting
    nonzero_diffs = diffs[diffs > 0]
    if len(nonzero_diffs) > 10:
        p95 = np.percentile(nonzero_diffs, 95)
        p50 = np.median(nonzero_diffs)
        signal = np.clip((diffs - p95) / (p95 - p50 + 1e-10), 0.0, 1.0)
    else:
        signal = np.zeros(n_seconds)

    n_spikes = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Tone change: %d spike(s) detected", n_spikes)
    return signal


def _scene_changes_to_signal(scene_changes: list[dict], n_seconds: int) -> np.ndarray:
    """Convert PySceneDetect scene change timestamps into a pixel_diff signal.

    Each scene change becomes a 1.0 at its timestamp, then smeared ±1s to
    account for minor alignment differences with other signals.
    """
    signal = np.zeros(n_seconds, dtype=np.float64)
    for sc in scene_changes:
        ts = sc.get("timestamp", 0.0)
        idx = int(round(ts))
        if 0 <= idx < n_seconds:
            signal[idx] = 1.0
    signal = _smear_signal(signal, radius=1)
    log.info("Scene changes → pixel_diff: %d hard cut(s)", len(scene_changes))
    return signal


def _detect_pixel_diff_spikes(
    frame_paths: list[Path], n_seconds: int,
) -> np.ndarray:
    """Detect hard cuts by measuring frame-to-frame pixel differences in HSV.

    Since frames may be sparsely sampled (0.2-1 fps), the boundary is placed
    at the midpoint between frames and smeared ±2s to cover timing uncertainty.
    """
    import cv2

    signal = np.zeros(n_seconds)
    prev_frame = None
    prev_ts = 0.0

    for path in frame_paths:
        ts = _frame_timestamp(path)
        if ts >= n_seconds:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        if prev_frame is not None:
            diff = float(np.mean(np.abs(hsv - prev_frame)))
            # Place the cut at the midpoint between frames
            midpoint = int((prev_ts + ts) / 2.0)
            midpoint = min(midpoint, n_seconds - 1)
            signal[midpoint] = max(signal[midpoint], diff)

        prev_frame = hsv
        prev_ts = ts

    # Normalize: only the top 5% of differences are "hard cuts"
    nonzero = signal[signal > 0]
    if len(nonzero) > 10:
        p95 = np.percentile(nonzero, 95)
        p50 = np.median(nonzero)
        signal = np.clip((signal - p95) / (p95 - p50 + 1e-10), 0.0, 1.0)
    else:
        signal = np.zeros(n_seconds)

    # Smear: spread each detection ±3s to account for sparse frame sampling
    signal = _smear_signal(signal, radius=3)

    n_spikes = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Pixel diff: %d hard cut(s) detected", n_spikes)
    return signal


def _detect_color_shifts(
    frame_paths: list[Path], n_seconds: int,
) -> np.ndarray:
    """Detect color temperature / grading shifts between consecutive frames.

    Uses hue and saturation histograms with chi-squared distance.
    Places boundary at midpoint between frames and smears ±2s.
    """
    import cv2

    signal = np.zeros(n_seconds)
    prev_hist_h = None
    prev_hist_s = None
    prev_ts = 0.0

    for path in frame_paths:
        ts = _frame_timestamp(path)
        if ts >= n_seconds:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)

        if prev_hist_h is not None:
            d_h = cv2.compareHist(prev_hist_h, hist_h, cv2.HISTCMP_CHISQR)
            d_s = cv2.compareHist(prev_hist_s, hist_s, cv2.HISTCMP_CHISQR)
            midpoint = int((prev_ts + ts) / 2.0)
            midpoint = min(midpoint, n_seconds - 1)
            signal[midpoint] = max(signal[midpoint], d_h + d_s)

        prev_hist_h = hist_h
        prev_hist_s = hist_s
        prev_ts = ts

    # Normalize: top 15% of color changes fire (p85 threshold)
    nonzero = signal[signal > 0]
    if len(nonzero) > 10:
        p85 = np.percentile(nonzero, 85)
        p50 = np.median(nonzero)
        signal = np.clip((signal - p85) / (p85 - p50 + 1e-10), 0.0, 1.0)
    else:
        signal = np.zeros(n_seconds)

    # Smear to cover frame timing uncertainty
    signal = _smear_signal(signal, radius=3)

    n_spikes = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Color shift: %d spike(s) detected", n_spikes)
    return signal


def _detect_speech_gaps(
    transcript: list[dict], n_seconds: int,
) -> np.ndarray:
    """Mark regions with no speech (potential ad with music/different voice).

    Returns per-second signal: 1.0 where there's a significant gap in speech.
    """
    speech_mask = np.zeros(n_seconds)

    for seg in transcript:
        start = int(seg.get("start", 0))
        end = int(seg.get("end", 0))
        start = max(0, min(start, n_seconds - 1))
        end = max(0, min(end, n_seconds))
        speech_mask[start:end] = 1.0

    # Find gaps longer than 5 seconds
    signal = np.zeros(n_seconds)
    in_gap = False
    gap_start = 0

    for i in range(n_seconds):
        if speech_mask[i] == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif speech_mask[i] == 1 and in_gap:
            in_gap = False
            gap_len = i - gap_start
            if gap_len >= 5:
                # Mark the boundaries of the gap (where speech stops and resumes)
                signal[gap_start] = min(1.0, gap_len / 20.0)
                signal[min(i, n_seconds - 1)] = min(1.0, gap_len / 20.0)

    if in_gap:
        gap_len = n_seconds - gap_start
        if gap_len >= 5:
            signal[gap_start] = min(1.0, gap_len / 20.0)

    n_gaps = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Speech gaps: %d boundary point(s)", n_gaps)
    return signal


def _detect_sponsor_keywords(
    transcript: list[dict], n_seconds: int,
) -> np.ndarray:
    """Mark regions containing sponsor/ad keywords in the transcript.

    Uses a density approach: aggregate keyword hits within sliding windows,
    then mark the boundaries of high-density regions.
    """
    # Per-second keyword score
    raw_score = np.zeros(n_seconds)

    for seg in transcript:
        text = seg.get("text", "").lower()
        if not text:
            continue

        hits = sum(1 for kw in SPONSOR_KEYWORDS if kw in text)
        if hits == 0:
            continue

        start = int(seg.get("start", 0))
        end = int(seg.get("end", 0))
        start = max(0, min(start, n_seconds - 1))
        end = max(start + 1, min(end, n_seconds))

        score = min(1.0, hits * 0.4)
        raw_score[start:end] = np.maximum(raw_score[start:end], score)

    # Smooth with a sliding window to find keyword-dense regions
    window = 30  # 30-second window
    density = np.convolve(raw_score, np.ones(window) / window, mode="same")

    # Mark boundaries where keyword density rises or falls
    signal = np.zeros(n_seconds)
    for i in range(1, n_seconds):
        # Rising edge: keyword region starts
        if density[i] > 0.05 and density[i - 1] < 0.02:
            signal[i] = min(1.0, density[i] * 10)
        # Falling edge: keyword region ends
        elif density[i] < 0.02 and density[i - 1] > 0.05:
            signal[i] = min(1.0, density[i - 1] * 10)

    # Also mark the raw presence with moderate weight
    signal = np.maximum(signal, raw_score * 0.7)

    # Smear slightly
    signal = _smear_signal(signal, radius=2)

    n_regions = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Sponsor keywords: %d region(s) detected", n_regions)
    return signal


# ── Phase 2: Expensive Signal Detectors (Targeted) ────────────────────────────


def _detect_topic_discontinuity(
    transcript: list[dict], n_seconds: int,
) -> np.ndarray:
    """Detect semantic topic jumps using sentence embeddings.

    Groups transcript into 10-second windows, embeds each window,
    and measures cosine distance between adjacent windows.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("sentence-transformers not installed, skipping topic detection")
        return np.zeros(n_seconds)

    # Group transcript text into 10-second windows
    window_size = 10
    n_windows = n_seconds // window_size + 1
    window_texts: list[str] = [""] * n_windows

    for seg in transcript:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if text:
            w_idx = min(int(start) // window_size, n_windows - 1)
            window_texts[w_idx] += " " + text

    # Filter out empty windows
    non_empty = [(i, t.strip()) for i, t in enumerate(window_texts) if t.strip()]
    if len(non_empty) < 3:
        return np.zeros(n_seconds)

    indices, texts = zip(*non_empty)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(list(texts), show_progress_bar=False)

    # Two approaches combined:
    # 1. Adjacent discontinuity: cosine distance between consecutive windows
    adj_disc = np.zeros(n_seconds)
    for i in range(1, len(embeddings)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]) /
                    (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-10))
        discontinuity = max(0.0, 1.0 - sim)
        t = indices[i] * window_size
        if t < n_seconds:
            adj_disc[t] = discontinuity

    # 2. Global foreignness: distance from the video's median topic
    median_emb = np.median(embeddings, axis=0)
    median_emb = median_emb / (np.linalg.norm(median_emb) + 1e-10)
    foreignness = np.zeros(n_seconds)
    for i, emb in enumerate(embeddings):
        sim = float(np.dot(emb, median_emb) /
                    (np.linalg.norm(emb) + 1e-10))
        dist = max(0.0, 1.0 - sim)
        t = indices[i] * window_size
        t_end = min(t + window_size, n_seconds)
        foreignness[t:t_end] = dist

    # Normalize adjacency: keep top 15%
    nonzero_adj = adj_disc[adj_disc > 0]
    if len(nonzero_adj) > 5:
        p85 = np.percentile(nonzero_adj, 85)
        p50 = np.median(nonzero_adj)
        adj_norm = np.clip((adj_disc - p85) / (p85 - p50 + 1e-10), 0.0, 1.0)
    else:
        adj_norm = np.zeros(n_seconds)

    # Normalize foreignness: keep top 20%
    nonzero_for = foreignness[foreignness > 0]
    if len(nonzero_for) > 5:
        p80 = np.percentile(nonzero_for, 80)
        p50 = np.median(nonzero_for)
        for_norm = np.clip((foreignness - p80) / (p80 - p50 + 1e-10), 0.0, 1.0)
    else:
        for_norm = np.zeros(n_seconds)

    # Combined signal: adjacency fires at boundaries, foreignness fires inside ad
    # For boundary detection, mark where foreignness transitions in/out
    foreign_boundary = np.zeros(n_seconds)
    for i in range(1, n_seconds):
        if for_norm[i] > 0.3 and for_norm[i - 1] < 0.1:
            foreign_boundary[i] = for_norm[i]
        elif for_norm[i] < 0.1 and for_norm[i - 1] > 0.3:
            foreign_boundary[i] = for_norm[i - 1]

    signal = np.maximum(adj_norm, foreign_boundary)
    signal = _smear_signal(signal, radius=2)

    n_jumps = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Topic discontinuity: %d jump(s) detected", n_jumps)
    return signal


def _detect_speaker_changes_targeted(
    audio_path: Path,
    target_times: list[float],
    n_seconds: int,
) -> np.ndarray | None:
    """Detect speaker changes at specific time points (targeted, not full-video).

    Uses speechbrain speaker embeddings on 3-second windows around each target time.
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        log.info("speechbrain not installed, skipping speaker change detection")
        return None

    import torchaudio

    signal = np.zeros(n_seconds)

    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    except Exception as e:
        log.warning("Failed to load speaker model: %s", e)
        return None

    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    except Exception as e:
        log.warning("Failed to load audio for speaker detection: %s", e)
        return None

    window_sec = 3.0
    window_samples = int(sr * window_sec)

    for t in target_times:
        t_sample = int(t * sr)
        # Get embeddings before and after the boundary
        before_start = max(0, t_sample - window_samples)
        before_end = t_sample
        after_start = t_sample
        after_end = min(waveform.shape[1], t_sample + window_samples)

        if before_end - before_start < sr or after_end - after_start < sr:
            continue

        before_wav = waveform[:, before_start:before_end]
        after_wav = waveform[:, after_start:after_end]

        try:
            emb_before = classifier.encode_batch(before_wav).squeeze()
            emb_after = classifier.encode_batch(after_wav).squeeze()
            # Cosine distance
            cos_sim = float(
                np.dot(emb_before.numpy(), emb_after.numpy()) /
                (np.linalg.norm(emb_before.numpy()) * np.linalg.norm(emb_after.numpy()) + 1e-10)
            )
            # Low similarity = different speaker
            score = max(0.0, 1.0 - cos_sim)
            sec = int(t)
            if sec < n_seconds:
                signal[sec] = score
        except Exception:
            continue

    n_changes = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Speaker change (targeted): %d change(s) at %d points", n_changes, len(target_times))
    return signal


def _detect_text_brands_targeted(
    frame_paths: list[Path],
    target_times: list[float],
    n_seconds: int,
) -> np.ndarray | None:
    """Detect brand/company text in frames near boundary candidates.

    Only runs OCR on frames close to detected hard cuts (not all frames).
    """
    try:
        import easyocr
    except ImportError:
        log.info("easyocr not installed, skipping text/brand detection")
        return None

    signal = np.zeros(n_seconds)

    # Only process frames within 3 seconds of a target time
    target_set = set()
    for t in target_times:
        for offset in range(-3, 4):
            target_set.add(int(t) + offset)

    frames_to_ocr = [
        p for p in frame_paths
        if int(_frame_timestamp(p)) in target_set
    ]

    if not frames_to_ocr:
        return signal

    log.info("OCR: processing %d frames near %d boundaries", len(frames_to_ocr), len(target_times))

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as e:
        log.warning("Failed to initialize OCR: %s", e)
        return None

    brand_keywords = [
        "pepsi", "coca-cola", "coke", "nike", "adidas", "samsung",
        "apple", "google", "amazon", "microsoft", "netflix", "spotify",
        "instacart", "bosch", "frank", "redhot", "red hot",
        "visit", "download", "subscribe", "order now", "free trial",
        "promo", "discount", "code", ".com", ".io",
    ]

    for path in frames_to_ocr:
        ts = _frame_timestamp(path)
        sec = int(ts)
        if sec >= n_seconds:
            continue

        try:
            results = reader.readtext(str(path), detail=0)
            all_text = " ".join(results).lower()
            hits = sum(1 for kw in brand_keywords if kw in all_text)
            if hits > 0:
                signal[sec] = min(1.0, hits * 0.4)
        except Exception:
            continue

    n_detections = int(np.sum(signal > SIGNAL_FIRE_THRESHOLD))
    log.info("Text/brand OCR: %d detection(s)", n_detections)
    return signal


# ── Voting and Pairing Logic ──────────────────────────────────────────────────


def _compute_weighted_boundary_scores(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Compute boundary scores using modality-level max + cross-modal bonus.

    Architecture (log-likelihood ratio framework):
    1. Within each modality, take max(signal * weight) — correlated signals
       shouldn't be summed to avoid double-counting.
    2. Sum across modalities — independent evidence is additive.
    3. Add cross-modal bonus when 2+ modalities fire simultaneously.
    """
    n_seconds = max(len(s) for s in signals.values()) if signals else 0
    if n_seconds == 0:
        return np.zeros(0)

    modality_scores = {}
    for modality, signal_names in MODALITY_GROUPS.items():
        mod_score = np.zeros(n_seconds)
        for name in signal_names:
            if name in signals:
                sig = signals[name]
                w = weights.get(name, 0.5)
                weighted = sig[:n_seconds] * w
                mod_score = np.maximum(mod_score, weighted)
        modality_scores[modality] = mod_score

    # Base score: sum of independent modality contributions
    scores = np.zeros(n_seconds)
    for mod_score in modality_scores.values():
        scores += mod_score

    # Cross-modal bonus: reward time points where multiple modalities fire
    for t in range(n_seconds):
        n_firing = sum(
            1 for ms in modality_scores.values()
            if ms[t] > MODALITY_FIRE_THRESHOLD
        )
        scores[t] += CROSS_MODAL_BONUS.get(n_firing, 1.0)

    return scores


def _find_boundary_peaks(
    scores: np.ndarray,
    min_distance: int = 10,
) -> list[tuple[float, float]]:
    """Find local peaks in the boundary score signal.

    Returns list of (time_seconds, score) for each peak.
    """
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(
        scores,
        height=0.8,
        distance=min_distance,
        prominence=0.3,
    )

    results = [(float(p), float(scores[p])) for p in peaks]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _count_firing_modalities(
    signals: dict[str, np.ndarray],
    t: float,
    window: int,
    n_seconds: int,
) -> int:
    """Count how many modalities have evidence near time t."""
    lo = max(0, int(t) - window)
    hi = min(n_seconds, int(t) + window + 1)
    modalities_firing = 0
    for modality, signal_names in MODALITY_GROUPS.items():
        for name in signal_names:
            if name in signals:
                val = float(np.max(signals[name][lo:hi]))
                if val >= 0.25:
                    modalities_firing += 1
                    break
    return modalities_firing


def _compute_temporal_coincidence_bonus(
    signals: dict[str, np.ndarray],
    t: float,
    n_seconds: int,
    video_duration: float,
) -> float:
    """Compute temporal coincidence bonus based on likelihood ratio.

    If audio and video changes co-occur within ±2s, the probability of this
    happening by chance is approximately (2k+1)/T where T is video duration
    and k=2. The log-likelihood ratio justifies a large bonus.
    """
    k = 2  # ±2s coincidence window
    lo = max(0, int(t) - k)
    hi = min(n_seconds, int(t) + k + 1)

    modalities_in_window = set()
    for modality, signal_names in MODALITY_GROUPS.items():
        for name in signal_names:
            if name in signals and float(np.max(signals[name][lo:hi])) > MODALITY_FIRE_THRESHOLD:
                modalities_in_window.add(modality)
                break

    if len(modalities_in_window) < 2:
        return 0.0

    # Likelihood ratio: T / (2k+1) — how unlikely this coincidence is by chance
    T = max(video_duration, float(n_seconds))
    lr = T / (2 * k + 1)
    # Scale: log(LR) mapped to a bounded bonus (cap at 1.0)
    bonus = min(1.0, float(np.log(lr)) / 6.0)
    return bonus


def _compute_interior_divergence(
    signals: dict[str, np.ndarray],
    t_start: int,
    t_end: int,
    n_seconds: int,
) -> float:
    """Compute Jensen-Shannon Divergence between interior and exterior signals.

    Measures how different the signal profile INSIDE the candidate ad is from
    the signal profile OUTSIDE. Higher JSD = more evidence of foreign content.
    Returns a score in [0, ln(2)] ≈ [0, 0.693].
    """
    from scipy.stats import entropy

    if t_end <= t_start or t_start < 0 or t_end > n_seconds:
        return 0.0

    interior_vec = []
    exterior_vec = []

    for name, sig in signals.items():
        interior_mean = float(np.mean(sig[t_start:t_end]))
        # Exterior: combine before and after regions
        before_mean = float(np.mean(sig[:t_start])) if t_start > 0 else 0.0
        after_mean = float(np.mean(sig[t_end:])) if t_end < n_seconds else 0.0
        exterior_mean = (before_mean + after_mean) / 2.0
        interior_vec.append(interior_mean)
        exterior_vec.append(exterior_mean)

    # Convert to probability distributions for JSD
    p = np.array(interior_vec, dtype=np.float64) + 1e-10
    q = np.array(exterior_vec, dtype=np.float64) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2.0
    jsd = float((entropy(p, m) + entropy(q, m)) / 2.0)
    return jsd


def _pair_boundaries_to_ads(
    boundaries: list[tuple[float, float]],
    signals: dict[str, np.ndarray],
    n_seconds: int,
    frame_histograms: dict[int, np.ndarray] | None = None,
) -> list[AdInterval]:
    """Pair boundary candidates into ad intervals using modality-aware scoring.

    Strategy:
    1. Take all boundaries above threshold
    2. For each pair 15-180s apart, score by:
       - Sum of individual boundary scores
       - Cross-modal corroboration (how many modalities fire at BOTH ends)
       - Temporal coincidence bonus (independent modalities within ±2s)
       - Interior evidence (keywords, speech gaps, JSD divergence)
       - Duration preference and visual return
    3. Keep best non-overlapping intervals
    """
    if len(boundaries) < 2:
        return []

    strong = [(t, s) for t, s in boundaries if s >= BOUNDARY_CONFIRM_THRESHOLD]
    if len(strong) < 2:
        return []

    strong_sorted = sorted(strong, key=lambda x: x[0])
    window = 5

    scored_pairs: list[tuple[float, float, float, dict]] = []
    video_duration = float(n_seconds)

    for i, (t1, s1) in enumerate(strong_sorted):
        for j in range(i + 1, len(strong_sorted)):
            t2, s2 = strong_sorted[j]
            duration = t2 - t1
            if duration < MIN_AD_DURATION:
                continue
            if duration > MAX_AD_DURATION:
                break

            # Base score: sum of individual boundary scores
            pair_score = s1 + s2

            # Cross-modal corroboration: count modalities firing at BOTH boundaries
            mods_at_start = _count_firing_modalities(signals, t1, window, n_seconds)
            mods_at_end = _count_firing_modalities(signals, t2, window, n_seconds)
            min_mods = min(mods_at_start, mods_at_end)
            pair_score += CROSS_MODAL_PAIR_BONUS.get(min_mods, 1.5)

            # Temporal coincidence: bonus for cross-modal co-occurrence within ±2s
            tc_bonus_start = _compute_temporal_coincidence_bonus(
                signals, t1, n_seconds, video_duration,
            )
            tc_bonus_end = _compute_temporal_coincidence_bonus(
                signals, t2, n_seconds, video_duration,
            )
            pair_score += (tc_bonus_start + tc_bonus_end) * 0.5

            # Interior evidence
            si, ei = int(t1), min(int(t2), n_seconds)
            kw_signal = signals.get("sponsor_keywords", np.zeros(n_seconds))
            gap_signal = signals.get("speech_gap", np.zeros(n_seconds))
            if ei > si:
                pair_score += float(np.max(kw_signal[si:ei])) * 1.5
                pair_score += float(np.max(gap_signal[si:ei])) * 0.5

            # Duration preference
            if duration <= 60:
                pair_score += 0.5
            elif duration <= 90:
                pair_score += 0.2
            elif duration > 120:
                pair_score -= 0.3

            # Jensen-Shannon Divergence: verify interior is genuinely different
            if ei > si:
                jsd = _compute_interior_divergence(signals, si, ei, n_seconds)
                if jsd > 0.3:
                    pair_score += 0.8
                elif jsd > 0.15:
                    pair_score += 0.3

            # Visual return check
            if frame_histograms:
                visual_return = _compute_visual_return_score(
                    frame_histograms, int(t1), int(t2), n_seconds,
                )
                pair_score += visual_return

            if pair_score >= PAIR_CONFIRM_THRESHOLD:
                # Multi-modality gate: collect signal evidence and verify
                interval_signals = {}
                modality_evidence = {"audio": False, "video": False, "transcript": False}

                for name, sig in signals.items():
                    lo1 = max(0, int(t1) - window)
                    hi1 = min(n_seconds, int(t1) + window + 1)
                    lo2 = max(0, int(t2) - window)
                    hi2 = min(n_seconds, int(t2) + window + 1)
                    val = max(float(np.max(sig[lo1:hi1])), float(np.max(sig[lo2:hi2])))
                    interval_signals[name] = round(val, 3)

                    if val >= 0.25:
                        for modality, names in MODALITY_GROUPS.items():
                            if name in names:
                                modality_evidence[modality] = True
                                break

                n_modalities = sum(modality_evidence.values())
                n_firing_signals = sum(
                    1 for v in interval_signals.values() if v >= 0.25
                )
                has_video = modality_evidence["video"]
                has_other = modality_evidence["audio"] or modality_evidence["transcript"]
                if has_video and has_other and n_firing_signals >= 3:
                    scored_pairs.append((t1, t2, pair_score, interval_signals))

    # Sort by score descending, resolve overlaps greedily
    scored_pairs.sort(key=lambda x: x[2], reverse=True)
    ads: list[AdInterval] = []
    used_ranges: list[tuple[float, float]] = []

    for t_start, t_end, score, interval_signals in scored_pairs:
        overlaps = any(
            t_start < ur_end and t_end > ur_start
            for ur_start, ur_end in used_ranges
        )
        if overlaps:
            continue

        confidence = min(1.0, score / 6.0)
        ads.append({
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "confidence": round(confidence, 3),
            "signals": interval_signals,
        })
        used_ranges.append((t_start, t_end))

    return ads


# ── Helpers ───────────────────────────────────────────────────────────────────


def _merge_adjacent_ads(ads: list[AdInterval], buffer: float = 10.0) -> list[AdInterval]:
    """Remove overlapping ad detections, keeping the highest-confidence ones.

    Uses a greedy approach: sort by confidence, accept each ad only if it
    doesn't overlap (within buffer seconds) with already-accepted ads.
    """
    if len(ads) <= 1:
        return ads

    # Sort by confidence descending
    ads_by_conf = sorted(ads, key=lambda x: x["confidence"], reverse=True)
    accepted: list[AdInterval] = []

    for ad in ads_by_conf:
        overlaps = any(
            ad["start"] < a["end"] + buffer and ad["end"] > a["start"] - buffer
            for a in accepted
        )
        if not overlaps:
            accepted.append(ad)

    return accepted


def _build_frame_histogram_index(
    frame_paths: list[Path], n_seconds: int,
) -> dict[int, np.ndarray]:
    """Build an index mapping timestamp (seconds) → HSV histogram for each frame.

    Used by the visual return check to compare frames before/after an ad
    to frames inside the ad.
    """
    import cv2

    index: dict[int, np.ndarray] = {}

    for path in frame_paths:
        ts = _frame_timestamp(path)
        sec = int(ts)
        if sec >= n_seconds:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Combined H+S histogram (flattened) as the visual fingerprint
        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        combined = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        index[sec] = combined

    return index


def _compute_visual_return_score(
    frame_histograms: dict[int, np.ndarray],
    t_start: int,
    t_end: int,
    n_seconds: int,
) -> float:
    """Score how well the video "returns" to its original look after an ad.

    Compares frames BEFORE t_start to frames AFTER t_end. If they're similar
    (same production style) but both differ from frames INSIDE [t_start, t_end],
    this is strong evidence of an inserted ad.

    Uses L2 distance on the raw histogram vectors and compares against the
    video's baseline pairwise variation to detect genuine "foreign" insertions.

    Returns a bonus score (0.0 to 0.6) to add to the pair score.
    """
    context_window = 15  # seconds to sample before/after

    before_hists = []
    inside_hists = []
    after_hists = []

    for t, hist in frame_histograms.items():
        if t_start - context_window <= t < t_start:
            before_hists.append(hist)
        elif t_start <= t <= t_end:
            inside_hists.append(hist)
        elif t_end < t <= t_end + context_window:
            after_hists.append(hist)

    if not before_hists or not inside_hists or not after_hists:
        return 0.0

    # Average histogram for each region
    before_avg = np.mean(before_hists, axis=0)
    inside_avg = np.mean(inside_hists, axis=0)
    after_avg = np.mean(after_hists, axis=0)

    # L2 distances between region averages
    d_before_after = float(np.linalg.norm(before_avg - after_avg))
    d_before_inside = float(np.linalg.norm(before_avg - inside_avg))
    d_after_inside = float(np.linalg.norm(after_avg - inside_avg))

    # Compute baseline: typical pairwise distance in the video context
    # (sample nearby frames to get a sense of normal variation)
    context_hists = before_hists + after_hists
    if len(context_hists) >= 2:
        pairwise_dists = []
        for i in range(len(context_hists)):
            for j in range(i + 1, min(len(context_hists), i + 3)):
                pairwise_dists.append(float(np.linalg.norm(
                    context_hists[i] - context_hists[j]
                )))
        baseline_dist = np.median(pairwise_dists) if pairwise_dists else 0.1
    else:
        baseline_dist = 0.1

    # The "visual return" pattern scoring:
    # 1. before↔after should be CLOSE (similar to baseline or less)
    # 2. interior↔context should be FAR (much more than baseline)
    avg_interior_dist = (d_before_inside + d_after_inside) / 2.0

    # Score criteria:
    # - before_after is close (within 2x baseline = normal variation)
    # - interior is far (at least 2x the before_after distance)
    context_similar = d_before_after < baseline_dist * 2.5
    interior_foreign = avg_interior_dist > d_before_after * 1.8 and avg_interior_dist > baseline_dist * 1.5

    if context_similar and interior_foreign:
        # Both conditions met: strong visual return evidence
        # Scale by how much the interior differs
        strength = min(1.0, (avg_interior_dist / (d_before_after + 0.01) - 1.5) / 3.0)
        return min(0.6, strength * 0.6)
    elif interior_foreign:
        # Interior is foreign but context isn't perfectly similar — weaker signal
        return 0.2
    else:
        return 0.0


def _smear_signal(signal: np.ndarray, radius: int = 2) -> np.ndarray:
    """Spread each signal peak ±radius seconds using max-pooling.

    This accounts for timing uncertainty in sparse frame sampling.
    """
    if radius <= 0:
        return signal
    smeared = signal.copy()
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        shifted = np.roll(signal, offset)
        if offset > 0:
            shifted[:offset] = 0
        else:
            shifted[offset:] = 0
        smeared = np.maximum(smeared, shifted * 0.8)
    return smeared


def _frame_timestamp(path: Path) -> float:
    """Extract timestamp from frame filename (frame_NNNNN.jpg where N is ms)."""
    return int(path.stem.split("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voting-based ad splice detection.")
    parser.add_argument("audio", help="Path to audio WAV")
    parser.add_argument("frames_dir", help="Path to frames directory")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--transcript", type=str, default=None)
    args = parser.parse_args()

    transcript_data = None
    if args.transcript:
        with open(args.transcript) as f:
            transcript_data = json.load(f)

    ads = detect_ads_voting(
        args.audio, args.frames_dir,
        video_duration=args.duration,
        transcript=transcript_data,
    )
    print(json.dumps(ads, indent=2))
