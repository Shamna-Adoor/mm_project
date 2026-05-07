"""Download sample videos defined in scripts/videos.yaml using yt-dlp."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_CONFIG = SCRIPTS_DIR / "videos.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_video(
    url: str,
    video_id: str,
    output_dir: Path,
    *,
    max_height: int = 720,
    fmt: str = "mp4",
    force: bool = False,
) -> Path:
    """Download a single video with yt-dlp.

    Parameters
    ----------
    url:
        YouTube (or other yt-dlp-supported) URL.
    video_id:
        Friendly slug used as the output filename (no extension).
    output_dir:
        Directory to write the file into.
    max_height:
        Cap vertical resolution to limit file size.
    fmt:
        Preferred container format.
    force:
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Expected output path (yt-dlp may append the actual extension).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{video_id}.%(ext)s")

    existing = list(output_dir.glob(f"{video_id}.*"))
    if existing and not force:
        print(f"  [skip] {video_id} — already downloaded ({existing[0].name})")
        return existing[0]

    cmd = [
        "yt-dlp",
        "--format", f"bestvideo[height<={max_height}][ext={fmt}]+bestaudio[ext=m4a]/best[height<={max_height}][ext={fmt}]/best[height<={max_height}]",
        "--merge-output-format", fmt,
        "--output", out_template,
        "--no-playlist",
        "--progress",
        url,
    ]

    print(f"  [download] {video_id} — {url}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [error] yt-dlp exited {result.returncode} for {video_id}", file=sys.stderr)
        sys.exit(result.returncode)

    written = list(output_dir.glob(f"{video_id}.*"))
    return written[0] if written else output_dir / f"{video_id}.{fmt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample videos listed in videos.yaml.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to videos.yaml config (default: scripts/videos.yaml)",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Download only these video IDs (space-separated). Downloads all if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = REPO_ROOT / config.get("output_dir", "data/videos")
    max_height = config.get("max_height", 720)
    fmt = config.get("format", "mp4")

    videos = config.get("videos", [])
    if args.ids:
        id_set = set(args.ids)
        videos = [v for v in videos if v["id"] in id_set]
        if not videos:
            print(f"No videos matched IDs: {args.ids}", file=sys.stderr)
            sys.exit(1)

    print(f"Downloading {len(videos)} video(s) to {output_dir}/\n")
    for entry in videos:
        download_video(
            entry["url"],
            entry["id"],
            output_dir,
            max_height=max_height,
            fmt=fmt,
            force=args.force,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
