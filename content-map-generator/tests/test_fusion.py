"""Test fusion logic with mock signals (stubs expected to raise NotImplementedError)."""

import pytest


MOCK_AUDIO_SIGNALS = {
    "silence_intervals": [{"start": 0.0, "end": 2.0, "energy": -70.0}],
    "music_intervals": [{"start": 0.0, "end": 30.0, "confidence": 0.9}],
    "transcript": [
        {"start": 30.0, "end": 35.0, "text": "Welcome back to the show"},
        {"start": 2700.0, "end": 2710.0, "text": "Thanks for watching, see you next time"},
    ],
    "sponsor_phrases": [
        {"start": 600.0, "end": 660.0, "phrase": "sponsored by", "context": "This video is sponsored by Acme Corp"}
    ],
    "boilerplate_phrases": [
        {"start": 30.0, "end": 35.0, "phrase": "welcome back", "type": "intro"},
        {"start": 2700.0, "end": 2710.0, "phrase": "thanks for watching", "type": "outro"},
    ],
}

MOCK_VISUAL_SIGNALS = {
    "scene_changes": [
        {"timestamp": 30.0, "confidence": 0.85},
        {"timestamp": 600.0, "confidence": 0.95},
        {"timestamp": 660.0, "confidence": 0.90},
        {"timestamp": 2700.0, "confidence": 0.88},
    ],
    "static_intervals": [],
    "ocr_detections": [
        {"timestamp": 620.0, "text": "Use code SAVE20 at checkout", "bbox": [10, 10, 200, 30]},
    ],
}

VIDEO_DURATION = 2730.0


def test_fusion_import():
    from analyzer import fusion  # noqa: F401


def test_classify_segments_returns_segments():
    from analyzer.fusion import classify_segments

    segs = classify_segments(MOCK_AUDIO_SIGNALS, MOCK_VISUAL_SIGNALS, VIDEO_DURATION)
    assert isinstance(segs, list)
    assert len(segs) > 0

def test_classify_segments_covers_full_duration():
    from analyzer.fusion import classify_segments

    segs = classify_segments(MOCK_AUDIO_SIGNALS, MOCK_VISUAL_SIGNALS, VIDEO_DURATION)
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == VIDEO_DURATION

def test_classify_segments_no_overlap():
    from analyzer.fusion import classify_segments

    segs = classify_segments(MOCK_AUDIO_SIGNALS, MOCK_VISUAL_SIGNALS, VIDEO_DURATION)
    for i in range(len(segs) - 1):
        assert segs[i]["end"] <= segs[i + 1]["start"], "Segments must not overlap"

def test_classify_segments_sponsor_detected():
    from analyzer.fusion import classify_segments

    segs = classify_segments(MOCK_AUDIO_SIGNALS, MOCK_VISUAL_SIGNALS, VIDEO_DURATION)
    labels = [s["label"] for s in segs]
    assert "sponsor" in labels, "Expected sponsor segment from mock sponsor phrase signal"

def test_classify_segments_schema():
    from analyzer.fusion import classify_segments

    segs = classify_segments(MOCK_AUDIO_SIGNALS, MOCK_VISUAL_SIGNALS, VIDEO_DURATION)
    required = {"start", "end", "label", "confidence", "skip_recommended", "reason", "signals_used"}
    for seg in segs:
        assert required.issubset(seg.keys())
        assert 0.0 <= seg["confidence"] <= 1.0
        assert isinstance(seg["skip_recommended"], bool)


def test_mock_signals_structure():
    """Ensure mock fixtures have the keys that fusion will expect."""
    assert "silence_intervals" in MOCK_AUDIO_SIGNALS
    assert "music_intervals" in MOCK_AUDIO_SIGNALS
    assert "transcript" in MOCK_AUDIO_SIGNALS
    assert "sponsor_phrases" in MOCK_AUDIO_SIGNALS
    assert "boilerplate_phrases" in MOCK_AUDIO_SIGNALS
    assert "scene_changes" in MOCK_VISUAL_SIGNALS
    assert "static_intervals" in MOCK_VISUAL_SIGNALS
    assert "ocr_detections" in MOCK_VISUAL_SIGNALS
