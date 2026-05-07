"""Validate that example fixtures conform to both JSON schemas."""

import json
from pathlib import Path

import pytest

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

VALID_SEGMENT = {
    "video_id": "test_video",
    "duration_seconds": 3600.0,
    "generated_at": "2024-01-01T00:00:00Z",
    "segments": [
        {
            "start": 0.0,
            "end": 45.0,
            "label": "intro",
            "confidence": 0.9,
            "skip_recommended": True,
            "reason": "Music + boilerplate greeting detected",
            "signals_used": ["music_intervals", "boilerplate_phrases"],
        },
        {
            "start": 45.0,
            "end": 3550.0,
            "label": "main_content",
            "confidence": 0.6,
            "skip_recommended": False,
            "reason": "Default classification",
            "signals_used": [],
        },
    ],
}

VALID_SIGNAL = {
    "video_id": "test_video",
    "audio_signals": {
        "silence_intervals": [{"start": 0.0, "end": 2.5, "energy": -60.0}],
        "music_intervals": [{"start": 0.0, "end": 45.0, "confidence": 0.85}],
        "transcript": [{"start": 45.0, "end": 60.0, "text": "Welcome back everyone"}],
        "sponsor_phrases": [],
        "boilerplate_phrases": [
            {"start": 45.0, "end": 50.0, "phrase": "welcome back", "type": "intro"}
        ],
    },
    "visual_signals": {
        "scene_changes": [{"timestamp": 45.0, "confidence": 0.9}],
        "static_intervals": [],
        "ocr_detections": [],
    },
}


def _load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / name) as f:
        return json.load(f)


def test_segment_schema_loads():
    schema = _load_schema("segment_schema.json")
    assert schema["title"] == "SegmentMap"


def test_signal_schema_loads():
    schema = _load_schema("signal_schema.json")
    assert schema["title"] == "SignalMap"


def test_valid_segment_fixture_has_required_fields():
    required = {"video_id", "duration_seconds", "generated_at", "segments"}
    assert required.issubset(VALID_SEGMENT.keys())


def test_valid_segment_labels():
    valid_labels = {"intro", "main_content", "sponsor", "outro", "dead_air"}
    for seg in VALID_SEGMENT["segments"]:
        assert seg["label"] in valid_labels


def test_segment_no_overlap():
    segs = VALID_SEGMENT["segments"]
    for i in range(len(segs) - 1):
        assert segs[i]["end"] <= segs[i + 1]["start"], "Segments must not overlap"


def test_valid_signal_fixture_structure():
    assert "audio_signals" in VALID_SIGNAL
    assert "visual_signals" in VALID_SIGNAL
    audio = VALID_SIGNAL["audio_signals"]
    assert "transcript" in audio
    assert "silence_intervals" in audio


def test_confidence_bounds():
    for seg in VALID_SEGMENT["segments"]:
        assert 0.0 <= seg["confidence"] <= 1.0
