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
TOTAL_STAGES      = 9


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
        from analyzer.audio.transcript_signals import (
            find_sponsor_phrases, find_boilerplate, find_commercial_clusters,
        )
        sponsors            = find_sponsor_phrases(transcript)
        boilerplate         = find_boilerplate(transcript)
        commercial_clusters = find_commercial_clusters(transcript)

        # ── Stage 5a: Audio coherence (style-shift detection) ──────────────────
        # Cheap defensive: never lets a librosa import error break the pipeline.
        _write_status(status_path, 5, "Analysing audio style shifts…", 65)
        try:
            from analyzer.audio.audio_coherence import detect_audio_anomalies
            audio_anomalies = detect_audio_anomalies(audio_path)
        except Exception as exc:
            log.warning("Audio coherence skipped: %s", exc)
            audio_anomalies = []

        audio_signals = {
            "silence_intervals":   silence,
            "music_intervals":     music,
            "transcript":          transcript,
            "sponsor_phrases":     sponsors,
            "boilerplate_phrases": boilerplate,
            "commercial_clusters": commercial_clusters,
            "audio_anomalies":     audio_anomalies,
        }

        # ── Stage 5b: LLM non-content classification ──────────────────────────
        _write_status(status_path, 5, "AI: classifying non-content segments…", 66)
        from analyzer.audio.llm_classify import classify_transcript
        llm_segments = classify_transcript(transcript, duration)

        # ── Stage 5c: Topic chapter generation ────────────────────────────────
        _write_status(status_path, 5, "AI: generating topic chapters…", 68)
        from analyzer.audio.topic_segments import generate_chapters
        chapters = generate_chapters(transcript, duration)

        # ── Stages 6-9: Visual analysis ────────────────────────────────────────
        visual_signals: dict = {
            "scene_changes":    [],
            "static_intervals": [],
            "ocr_detections":   [],
        }

        if not skip_visual:
            from analyzer.visual.extract_frames  import extract_frames
            from analyzer.visual.scene_detect    import detect_scenes
            from analyzer.visual.ocr_frames      import ocr_frames
            from analyzer.visual.motion_analysis import detect_static_intervals

            frames_dir = idir / "frames"

            # 1 frame per 5 s → ~290 frames for a 24-min video (was 1440 at 1 fps)
            _write_status(status_path, 6, "Extracting video frames…", 71)
            extract_frames(video_path, frames_dir, fps=0.2, force=force)

            _write_status(status_path, 7, "Detecting scene changes…", 75)
            visual_signals["scene_changes"] = detect_scenes(video_path)

            # OCR every 10 s on the already-sparse frames
            _write_status(status_path, 8, "Running OCR on frames…", 82)
            visual_signals["ocr_detections"] = ocr_frames(
                frames_dir, sample_every_n_seconds=10
            )

            _write_status(status_path, 9, "Detecting static / motion intervals…", 90)
            visual_signals["static_intervals"] = detect_static_intervals(frames_dir)

            # ── Stage 9b: Visual coherence (color / style shift) ───────────────
            # Reuses the frames already extracted; adds ~5–15s on a 30-min video.
            _write_status(status_path, 9, "Analysing visual style coherence…", 92)
            try:
                from analyzer.visual.visual_coherence import detect_visual_anomalies
                visual_signals["visual_anomalies"] = detect_visual_anomalies(frames_dir)
            except Exception as exc:
                log.warning("Visual coherence skipped: %s", exc)
                visual_signals["visual_anomalies"] = []
        else:
            log.info("Visual analysis skipped (--skip-visual)")
            visual_signals["visual_anomalies"] = []
            frames_dir = None  # type: ignore[assignment]

        # ── Stage 9c: Joint-novelty detection (template-free) ─────────────────
        # The principled "is this region unlike the rest of THIS video?"
        # detector. Combines audio, color, and structural signals into one
        # robustly-standardised distance per window. Catches inserted ads
        # that don't fit any specific template (calm narrative ads, ads in
        # animation, broadcast spots) by leveraging the fundamental fact
        # that an inserted ad is, by definition, a passage produced in a
        # different production environment than the host content.
        _write_status(status_path, 9, "Computing joint-novelty score…", 94)
        novelty_regions: list[dict] = []
        try:
            from analyzer.joint_novelty import detect_novelty_regions
            from analyzer.audio.llm_classify import classify_content_type, get_profile_for
            ct_pre   = classify_content_type(
                visual_signals.get("scene_changes", []),
                visual_signals.get("static_intervals", []),
                duration,
            )
            profile_pre = get_profile_for(ct_pre["type"])
            novelty_k = float(profile_pre.get("novelty_sigmas", 2.5))
            log.info(
                "Joint-novelty using k=%.2f (content-type: %s)",
                novelty_k, ct_pre["type"],
            )
            novelty_regions = detect_novelty_regions(
                audio_path,
                frames_dir if not skip_visual else None,
                visual_signals.get("scene_changes", []),
                audio_signals.get("transcript", []),
                duration,
                novelty_sigmas=novelty_k,
            )
        except Exception as exc:
            log.warning("Joint-novelty skipped: %s", exc)
            novelty_regions = []
        # Surface novelty in both signal namespaces for caching + diagnostics.
        audio_signals["novelty_regions"] = novelty_regions

        # ── Cache signal JSON ──────────────────────────────────────────────────
        signal_path = idir / "signals.json"
        with open(signal_path, "w") as f:
            json.dump({"video_id": vid_id, "audio_signals": audio_signals, "visual_signals": visual_signals}, f, indent=2)
        log.info("Signal JSON written: %s", signal_path)

        # ── Fusion + write output ──────────────────────────────────────────────
        _write_status(status_path, 9, "Fusing signals into segments…", 95)

        if llm_segments:
            # LLM succeeded — overlay silence + music + visual + novelty signals on top
            from analyzer.audio.llm_classify import to_output_segments
            segments = to_output_segments(
                llm_segments,
                silence_intervals=audio_signals.get("silence_intervals", []),
                music_intervals=audio_signals.get("music_intervals", []),
                video_duration=duration,
                scene_changes=visual_signals.get("scene_changes", []),
                transcript=audio_signals.get("transcript", []),
                static_intervals=visual_signals.get("static_intervals", []),
                commercial_clusters=audio_signals.get("commercial_clusters", []),
                audio_anomalies=audio_signals.get("audio_anomalies", []),
                visual_anomalies=visual_signals.get("visual_anomalies", []),
                novelty_regions=novelty_regions,
            )
            log.info("Using LLM+audio+visual+novelty segments: %d total", len(segments))
        else:
            # Ollama unavailable — fall back to rule-based fusion
            from analyzer.fusion import classify_segments
            segments = classify_segments(
                audio_signals, visual_signals, duration,
                novelty_regions=novelty_regions,
            )
            log.info("Using rule-based fusion: %d total", len(segments))

        # Diagnostic content-type tag (purely informational; never affects
        # which segments are returned). Helpful for tuning thresholds and
        # understanding what profile the pipeline applied.
        from analyzer.audio.llm_classify import classify_content_type
        content_info = classify_content_type(
            visual_signals.get("scene_changes", []),
            visual_signals.get("static_intervals", []),
            duration,
        )

        PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = PREDICTIONS_ROOT / f"{vid_id}.json"
        output = {
            "video_id":         vid_id,
            "duration_seconds": round(duration, 3),
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "content_type":     content_info,
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
