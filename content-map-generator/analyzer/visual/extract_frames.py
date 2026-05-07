"""Sample frames from a video at a fixed rate using ffmpeg (fast seek)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from analyzer._logging import get_logger

log = get_logger(__name__)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 0.2,
    force: bool = False,
) -> list[Path]:
    """Extract JPEG frames from *video_path* at *fps* frames per second.

    Uses ffmpeg's fps filter — encodes only the frames needed, not all frames.
    Filenames: ``frame_{timestamp_ms:010d}.jpg`` — lexicographic = chronological.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and not force:
        log.info("Using %d cached frames in %s", len(existing), output_dir)
        return existing

    # Step 1 — extract frames named by sequential index via ffmpeg
    tmp_pattern = str(output_dir / "tmp_%08d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "5",          # JPEG quality (2=best, 31=worst); 5 is good enough for OCR
        "-an",                 # no audio
        tmp_pattern,
    ]
    log.info("Extracting frames from %s at %.2f fps with ffmpeg…", video_path.name, fps)
    subprocess.run(cmd, capture_output=True, check=True)

    # Step 2 — rename tmp_XXXXXXXX.jpg → frame_{ms:010d}.jpg
    #   frame i (1-indexed) corresponds to time (i-1)/fps seconds
    tmp_frames = sorted(output_dir.glob("tmp_*.jpg"))
    written: list[Path] = []
    for i, tmp_path in enumerate(tmp_frames):
        ts_ms  = int(i * (1000.0 / fps))
        out    = output_dir / f"frame_{ts_ms:010d}.jpg"
        tmp_path.rename(out)
        written.append(out)

    log.info("Extracted %d frames to %s", len(written), output_dir)
    return sorted(written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--fps", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = extract_frames(args.video, args.output_dir, fps=args.fps, force=args.force)
    print(f"Extracted {len(paths)} frames to {args.output_dir}")
