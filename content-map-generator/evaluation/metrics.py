"""Evaluation metrics: temporal IoU, precision/recall/F1, skip accuracy."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

ALL_LABELS = ["intro", "main_content", "sponsor", "outro", "dead_air"]
SKIP_LABELS = {"intro", "sponsor", "outro", "dead_air"}


class SegmentMetrics(TypedDict):
    label:     str
    precision: float
    recall:    float
    f1:        float
    iou:       float


# ── Core metric functions ─────────────────────────────────────────────────────

def temporal_iou(
    pred_start: float, pred_end: float,
    gt_start:   float, gt_end:   float,
) -> float:
    """Intersection-over-union for two temporal intervals."""
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def per_label_metrics(
    predictions:    list[dict],
    ground_truth:   list[dict],
    video_duration: float,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, SegmentMetrics]:
    """Precision, recall, F1, and mean IoU per label + overall macro average.

    A predicted segment counts as a true positive when its IoU with any
    ground-truth segment of the same label exceeds *iou_threshold*.
    """
    results: dict[str, SegmentMetrics] = {}

    for label in ALL_LABELS:
        preds = [s for s in predictions  if s["label"] == label]
        gts   = [s for s in ground_truth if s["label"] == label]

        if not preds and not gts:
            results[label] = SegmentMetrics(label=label, precision=1.0, recall=1.0, f1=1.0, iou=1.0)
            continue

        matched_gt  = set()
        tp = 0
        ious: list[float] = []

        for pred in preds:
            best_iou = 0.0
            best_j   = -1
            for j, gt in enumerate(gts):
                iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j
            ious.append(best_iou)
            if best_iou >= iou_threshold and best_j not in matched_gt:
                tp += 1
                matched_gt.add(best_j)

        fp = len(preds) - tp
        fn = len(gts)   - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou  = float(np.mean(ious)) if ious else 0.0

        results[label] = SegmentMetrics(
            label=label, precision=round(precision, 4),
            recall=round(recall, 4), f1=round(f1, 4), iou=round(mean_iou, 4),
        )

    # Macro average over labels that exist in either set
    active = [l for l in ALL_LABELS if any(s["label"] == l for s in predictions + ground_truth)]
    if active:
        results["overall"] = SegmentMetrics(
            label="overall",
            precision=round(np.mean([results[l]["precision"] for l in active]), 4),
            recall   =round(np.mean([results[l]["recall"]    for l in active]), 4),
            f1       =round(np.mean([results[l]["f1"]        for l in active]), 4),
            iou      =round(np.mean([results[l]["iou"]       for l in active]), 4),
        )

    return results


def seconds_correctly_skipped(
    predictions:  list[dict],
    ground_truth: list[dict],
) -> dict[str, float]:
    """True-positive and false-positive skipped seconds.

    Returns
    -------
    dict with keys: tp_skipped, fp_skipped, fn_skipped,
                    skip_precision, skip_recall
    """
    def to_mask(segments: list[dict], skip_only: bool) -> set[int]:
        """1-second resolution set of second indices that are 'skip'."""
        out: set[int] = set()
        for seg in segments:
            if skip_only and not seg.get("skip_recommended", seg["label"] in SKIP_LABELS):
                continue
            for t in range(int(seg["start"]), int(seg["end"])):
                out.add(t)
        return out

    pred_skip = to_mask(predictions,  skip_only=True)
    gt_skip   = to_mask(ground_truth, skip_only=True)

    tp = len(pred_skip & gt_skip)
    fp = len(pred_skip - gt_skip)
    fn = len(gt_skip   - pred_skip)

    return {
        "tp_skipped":     tp,
        "fp_skipped":     fp,
        "fn_skipped":     fn,
        "skip_precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4),
        "skip_recall":    round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
    }


def confusion_matrix(
    predictions:    list[dict],
    ground_truth:   list[dict],
    video_duration: float,
    *,
    resolution: float = 1.0,
) -> np.ndarray:
    """Per-second confusion matrix of shape (n_labels, n_labels).

    Rows = ground-truth label, columns = predicted label.
    """
    label_idx = {l: i for i, l in enumerate(ALL_LABELS)}
    n = len(ALL_LABELS)
    mat = np.zeros((n, n), dtype=int)

    def label_at(segments: list[dict], t: float) -> str:
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                return seg["label"]
        return "main_content"

    t = 0.0
    while t < video_duration:
        gt_label   = label_at(ground_truth, t)
        pred_label = label_at(predictions,  t)
        r = label_idx.get(gt_label,   label_idx["main_content"])
        c = label_idx.get(pred_label, label_idx["main_content"])
        mat[r, c] += 1
        t += resolution

    return mat
