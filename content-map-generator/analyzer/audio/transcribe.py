"""Transcribe audio using faster-whisper and return timestamped segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from analyzer._logging import get_logger

log = get_logger(__name__)


class TranscriptWord(TypedDict, total=False):
    start: float
    end: float
    text: str
    # Optional confidence fields (exposed when available from the recognizer).
    # Downstream detectors use these to identify regions where the transcript
    # is unreliable — typical for inserted ads with unfamiliar audio, music
    # backgrounds, or non-host voices Whisper hasn't been primed on. When
    # absent, downstream code treats the segment as "confidence unknown".
    avg_logprob:    float
    no_speech_prob: float


def transcribe(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    force: bool = False,
    cache_dir: str | Path | None = None,
) -> list[TranscriptWord]:
    """Transcribe *audio_path* with faster-whisper (int8, CPU) and return timestamped segments.

    Results are cached as JSON. Subsequent calls return instantly unless *force=True*.
    faster-whisper is 4-5x faster than openai-whisper on CPU.
    """
    audio_path = Path(audio_path)
    cache_path = Path(cache_dir) / "transcript.json" if cache_dir else audio_path.parent / "transcript.json"

    if not force:
        cached = _load_cache(cache_path)
        if cached is not None:
            log.info("Loaded transcript from cache (%d segments)", len(cached))
            return cached

    log.info("Transcribing %s with faster-whisper model=%s…", audio_path.name, model_name)

    from faster_whisper import WhisperModel  # deferred import

    model = WhisperModel(model_name, device="cpu", compute_type="int8_float32")
    raw_segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=False,
        language="en",
    )
    log.info("Detected language: %s (%.0f%%)", info.language, info.language_probability * 100)

    segments: list[TranscriptWord] = []
    for seg in raw_segments:           # generator — consumed once here
        entry: TranscriptWord = {
            "start": round(seg.start, 3),
            "end":   round(seg.end,   3),
            "text":  seg.text.strip(),
        }
        # faster-whisper exposes per-segment confidence proxies. They're
        # forwarded as plain floats so all downstream consumers (including
        # cached JSON) remain readable and version-tolerant.
        avg_lp = getattr(seg, "avg_logprob", None)
        if avg_lp is not None:
            entry["avg_logprob"] = float(avg_lp)
        no_sp = getattr(seg, "no_speech_prob", None)
        if no_sp is not None:
            entry["no_speech_prob"] = float(no_sp)
        segments.append(entry)

    log.info("Transcription complete: %d segments", len(segments))
    _save_cache(segments, cache_path)
    return segments


def _load_cache(cache_path: Path) -> list[TranscriptWord] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("Cache corrupt, re-transcribing: %s", cache_path)
        return None


def _save_cache(data: list[TranscriptWord], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Transcript cached: %s", cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper.")
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--model", default="tiny", dest="model_name")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    segs = transcribe(args.audio, model_name=args.model_name, force=args.force, cache_dir=args.cache_dir)
    print(json.dumps(segs, indent=2))
