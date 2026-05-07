"""Top-level pipeline orchestrator: video in → segment JSON out."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from analyzer._logging import get_logger

log = get_logger(__name__)

INTERMEDIATE_ROOT = Path(__file__).parent.parent / "data" / "intermediate"
PREDICTIONS_ROOT  = Path(__file__).parent.parent / "data" / "predictions"
TOTAL_STAGES      = 10


def run_pipeline(
    video_path: str | Path,
    *,
    whisper_model: str = "base",
    force: bool = False,
    skip_visual: bool = False,
    status_path: Path | None = None,
) -> Path:
    """Run the full analysis pipeline on *video_path*.

    Stages
    ------
    1. Extract audio (16 kHz WAV)
    2. Transcribe with Whisper
    3. Detect silence
    4. Detect music
    5. Find sponsor / boilerplate phrases
    6. Extract frames + scene changes + OCR + static detection  (skipped if skip_visual)
    7-9. (visual sub-stages)
    Final. Fuse signals → segment list + write output JSON
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vid_id   = _video_id(video_path)
    idir     = _intermediate_dir(vid_id)
    duration = _probe_duration(video_path)

    if status_path is None:
        status_path = idir / "status.json"

    log.info("=== Pipeline start: %s (%.1f s) ===", video_path.name, duration)

    try:
        # ── Stage 1: Audio extraction ──────────────────────────────────────────
        _write_status(status_path, 1, "Extracting audio…", 5)
        from analyzer.audio.extract_audio import extract_audio
        audio_path = extract_audio(video_path, idir / "audio.wav", force=force)

        # ── Stage 2: Transcription ─────────────────────────────────────────────
        _write_status(status_path, 2, f"Transcribing audio with Whisper '{whisper_model}' model…", 12)
        from analyzer.audio.transcribe import transcribe
        transcript = transcribe(audio_path, model_name=whisper_model, force=force, cache_dir=idir)

        # ── Stage 3: Silence detection ─────────────────────────────────────────
        _write_status(status_path, 3, "Detecting silence intervals…", 52)
        from analyzer.audio.silence_detect import detect_silence
        silence = detect_silence(audio_path, min_duration=1.0, threshold_db=-38.0)

        # ── Stage 4: Music detection ───────────────────────────────────────────
        _write_status(status_path, 4, "Detecting music…", 58)
        from analyzer.audio.music_detect import detect_music
        music = detect_music(audio_path)

        # ── Stage 5: Transcript signals ────────────────────────────────────────
        _write_status(status_path, 5, "Finding sponsor and boilerplate phrases…", 64)
        from analyzer.audio.transcript_signals import find_sponsor_phrases, find_boilerplate
        sponsors    = find_sponsor_phrases(transcript)
        boilerplate = find_boilerplate(transcript)

        audio_signals = {
            "silence_intervals":   silence,
            "music_intervals":     music,
            "transcript":          transcript,
            "sponsor_phrases":     sponsors,
            "boilerplate_phrases": boilerplate,
        }

        # ── Stage 5b: LLM non-content classification ──────────────────────────
        _write_status(status_path, 5, "AI: classifying non-content segments…", 66)
        from analyzer.audio.llm_classify import classify_transcript
        llm_segments = classify_transcript(transcript, duration)

        # ── Stage 5c: Topic chapter generation ────────────────────────────────
        _write_status(status_path, 5, "AI: generating topic chapters…", 68)
        from analyzer.audio.topic_segments import generate_chapters
        chapters = generate_chapters(transcript, duration)

        # ── Stages 6-10: Visual + multimodal analysis ─────────────────────────
        visual_signals: dict = {
            "scene_changes":    [],
            "static_intervals": [],
            "ocr_detections":   [],
            "ad_intervals":     [],
            "intro_interval":   None,
            "dead_air_multimodal": [],
        }

        if not skip_visual:
            from analyzer.visual.extract_frames  import extract_frames
            from analyzer.visual.scene_detect    import detect_scenes
            from analyzer.visual.ocr_frames      import ocr_frames
            from analyzer.visual.motion_analysis import detect_static_intervals
            from analyzer.visual.ad_detect       import detect_ad_intervals

            frames_dir = idir / "frames"

            # 0.5 fps (1 frame per 2s) — balances detection accuracy with speed
            _write_status(status_path, 6, "Extracting video frames…", 68)
            extract_frames(video_path, frames_dir, fps=0.5, force=force)

            _write_status(status_path, 7, "Detecting scene changes…", 73)
            visual_signals["scene_changes"] = detect_scenes(video_path)

            _write_status(status_path, 8, "Running OCR on frames…", 78)
            visual_signals["ocr_detections"] = ocr_frames(
                frames_dir, sample_every_n_seconds=10
            )

            _write_status(status_path, 9, "Detecting static / motion intervals…", 83)
            visual_signals["static_intervals"] = detect_static_intervals(frames_dir)

            # ── Stage 10: Multimodal ad + intro + dead-air detection ────────────
            _write_status(status_path, 10, "Multimodal ad detection (island + transcript)…", 86)
            from analyzer.multimodal_detect import detect_multimodal_ads, detect_multimodal_dead_air
            from analyzer.intro_detect import detect_intro

            multimodal_ads = detect_multimodal_ads(
                audio_path, frames_dir,
                video_duration=duration,
                transcript=transcript,
            )

            if multimodal_ads:
                visual_signals["ad_intervals"] = multimodal_ads
                log.info("Multimodal ad detection: %d confirmed interval(s)", len(multimodal_ads))
            else:
                # No multimodal ads found — leave ad_intervals empty rather than
                # using the visual-only fallback which generates too many false positives
                log.info("Multimodal detection found no confirmed ads")

            _write_status(status_path, 10, "Detecting intro segment…", 91)
            visual_signals["intro_interval"] = detect_intro(
                audio_path, frames_dir, video_duration=duration
            )

            _write_status(status_path, 10, "Multimodal dead-air detection…", 93)
            visual_signals["dead_air_multimodal"] = detect_multimodal_dead_air(
                audio_path, frames_dir, video_duration=duration
            )
        else:
            log.info("Visual analysis skipped (--skip-visual)")

        # ── Cache signal JSON ──────────────────────────────────────────────────
        signal_path = idir / "signals.json"
        with open(signal_path, "w") as f:
            json.dump({"video_id": vid_id, "audio_signals": audio_signals, "visual_signals": visual_signals}, f, indent=2)
        log.info("Signal JSON written: %s", signal_path)

        # ── Fusion + write output ──────────────────────────────────────────────
        _write_status(status_path, 9, "Fusing signals into segments…", 95)

        if llm_segments:
            # LLM succeeded — overlay silence + music + multimodal signals on top
            from analyzer.audio.llm_classify import to_output_segments
            segments = to_output_segments(
                llm_segments,
                silence_intervals=audio_signals.get("silence_intervals", []),
                music_intervals=audio_signals.get("music_intervals", []),
                ad_intervals=visual_signals.get("ad_intervals", []),
                video_duration=duration,
                intro_interval=visual_signals.get("intro_interval"),
                dead_air_multimodal=visual_signals.get("dead_air_multimodal", []),
            )
            log.info("Using LLM+audio+multimodal segments: %d total", len(segments))
        else:
            # Ollama unavailable — fall back to rule-based fusion
            from analyzer.fusion import classify_segments
            segments = classify_segments(audio_signals, visual_signals, duration)
            log.info("Using rule-based fusion: %d total", len(segments))

        PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = PREDICTIONS_ROOT / f"{vid_id}.json"
        output = {
            "video_id":         vid_id,
            "duration_seconds": round(duration, 3),
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "segments":         segments,
            "chapters":         chapters,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        _write_status(status_path, 9, f"Done — {len(segments)} segments detected.", 100, status="done")
        log.info("=== Pipeline complete: %s ===", out_path)
        return out_path

    except Exception as exc:
        _write_status(status_path, 0, str(exc), 0, status="error")
        raise


# ── Utilities ─────────────────────────────────────────────────────────────────

def _write_status(
    path: Path,
    stage_num: int,
    message: str,
    percent: int,
    *,
    status: str = "processing",
) -> None:
    payload = {
        "stage_num":    stage_num,
        "total_stages": TOTAL_STAGES,
        "message":      message,
        "percent":      percent,
        "status":       status,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)
    except OSError:
        pass  # non-fatal
    log.info("[%d%%] %s", percent, message)


def _video_id(video_path: Path) -> str:
    return video_path.stem.lower().replace(" ", "_")


def _intermediate_dir(video_id: str) -> Path:
    d = INTERMEDIATE_ROOT / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info   = json.loads(result.stdout)
    for stream in info.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration of {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full content-map analysis pipeline.")
    parser.add_argument("video")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--force",        action="store_true")
    parser.add_argument("--skip-visual",  action="store_true")
    args = parser.parse_args()

    out = run_pipeline(args.video, whisper_model=args.whisper_model, force=args.force, skip_visual=args.skip_visual)
    print(f"\nSegment JSON written to: {out}")
