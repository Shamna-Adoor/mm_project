"""CLI comparison tool: print metrics table and visualise GT vs predicted timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import (
    ALL_LABELS,
    per_label_metrics,
    seconds_correctly_skipped,
)

# Colour palette for timeline bars (matplotlib colour names)
_LABEL_COLORS = {
    "intro":        "#4f86c6",
    "main_content": "#5cb85c",
    "sponsor":      "#f0ad4e",
    "outro":        "#9b59b6",
    "dead_air":     "#e74c3c",
}


def compare(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    visualize: bool = True,
    output_image: str | Path | None = None,
) -> None:
    """Print a metrics table and optionally save a timeline visualisation."""
    gt_path   = Path(gt_path)
    pred_path = Path(pred_path)

    with open(gt_path)   as f: gt_data   = json.load(f)
    with open(pred_path) as f: pred_data = json.load(f)

    gt_segs   = gt_data["segments"]
    pred_segs = pred_data["segments"]
    duration  = gt_data.get("duration_seconds") or pred_data.get("duration_seconds", 0.0)

    metrics = per_label_metrics(pred_segs, gt_segs, duration)
    skip    = seconds_correctly_skipped(pred_segs, gt_segs)

    _print_metrics_table(metrics, skip)

    if visualize:
        _draw_timeline(
            gt_segs, pred_segs, duration,
            output_image=Path(output_image) if output_image else None,
        )


def _print_metrics_table(metrics: dict, skip: dict | None = None) -> None:
    """Pretty-print the metrics dict as an ASCII table."""
    order   = [l for l in ALL_LABELS if l in metrics] + (["overall"] if "overall" in metrics else [])
    col_w   = [16, 11, 11, 11, 11]
    headers = ["Label", "Precision", "Recall", "F1", "Mean IoU"]
    sep     = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(cells: list[str]) -> str:
        return "|" + "|".join(f" {c:<{col_w[i]-2}} " for i, c in enumerate(cells)) + "|"

    print(sep)
    print(row(headers))
    print(sep)
    for label in order:
        m = metrics[label]
        cells = [
            label,
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['iou']:.4f}",
        ]
        print(row(cells))
        if label == "dead_air":   # separator before overall
            print(sep)
    print(sep)

    if skip:
        print()
        print("Skip-second accuracy")
        print(f"  TP skipped : {skip['tp_skipped']:>6} s")
        print(f"  FP skipped : {skip['fp_skipped']:>6} s  (false skip)")
        print(f"  FN skipped : {skip['fn_skipped']:>6} s  (missed skip)")
        print(f"  Precision  : {skip['skip_precision']:.4f}")
        print(f"  Recall     : {skip['skip_recall']:.4f}")


def _draw_timeline(
    gt_segments:   list[dict],
    pred_segments: list[dict],
    video_duration: float,
    output_image: Path | None,
) -> None:
    """Draw a two-row horizontal bar chart (GT on top, pred on bottom)."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualisation.")
        return

    fig, ax = plt.subplots(figsize=(16, 3))

    def draw_row(segments: list[dict], y: float, height: float = 0.4) -> None:
        for seg in segments:
            color = _LABEL_COLORS.get(seg["label"], "#cccccc")
            ax.barh(
                y, seg["end"] - seg["start"],
                left=seg["start"], height=height,
                color=color, edgecolor="white", linewidth=0.4,
            )

    draw_row(gt_segments,   1.3)
    draw_row(pred_segments, 0.7)

    ax.set_xlim(0, video_duration)
    ax.set_ylim(0.3, 1.9)
    ax.set_yticks([0.7, 1.3])
    ax.set_yticklabels(["Predicted", "Ground Truth"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Segment Timeline: Ground Truth vs Predicted")

    legend_patches = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in _LABEL_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=5)

    plt.tight_layout()

    if output_image:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_image, dpi=150)
        print(f"Timeline saved to: {output_image}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ground-truth and predicted segment JSON files.")
    parser.add_argument("--gt",           required=True, help="Path to ground-truth JSON")
    parser.add_argument("--pred",         required=True, help="Path to predictions JSON")
    parser.add_argument("--no-viz",       action="store_false", dest="visualize")
    parser.add_argument("--output-image", help="Save figure to this path instead of displaying")
    args = parser.parse_args()

    compare(args.gt, args.pred, visualize=args.visualize, output_image=args.output_image)
