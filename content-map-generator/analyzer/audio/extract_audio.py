"""Extract a 16 kHz mono WAV from a video file using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from analyzer._logging import get_logger

log = get_logger(__name__)


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16000,
    force: bool = False,
) -> Path:
    """Extract audio from *video_path* and write a WAV to *output_path*.

    Returns the resolved output path. Skips extraction if the file already
    exists unless *force=True*.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path.exists() and not force:
        log.info("Using cached audio: %s", output_path)
        return output_path.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_ffmpeg_cmd(video_path, output_path, sample_rate)
    log.info("Extracting audio: %s → %s", video_path.name, output_path.name)

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    size_mb = output_path.stat().st_size / 1_000_000
    log.info("Audio extracted: %.1f MB at %d Hz", size_mb, sample_rate)
    return output_path.resolve()


def _build_ffmpeg_cmd(video_path: Path, output_path: Path, sample_rate: int) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                   # drop video stream
        "-acodec", "pcm_s16le",  # 16-bit PCM — Whisper's preferred format
        "-ar", str(sample_rate),
        "-ac", "1",              # mono
        str(output_path),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from a video file.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("output", help="Path for output WAV file")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = extract_audio(args.video, args.output, sample_rate=args.sample_rate, force=args.force)
    print(result)
