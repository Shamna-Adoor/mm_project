from analyzer.audio.llm_classify import to_output_segments
from analyzer.source_resolver import make_job_id, normalize_source_url, resolve_source


def test_direct_mp4_source_resolves_without_ytdlp():
    src = resolve_source("https://cdn.example.com/videos/demo.mp4")
    assert src["playback_url"] == "https://cdn.example.com/videos/demo.mp4"
    assert src["mime_type"] == "video/mp4"
    assert src["can_analyze"] is True
    assert src["is_stream"] is False


def test_direct_hls_source_is_stream_only():
    src = resolve_source("https://cdn.example.com/live/master.m3u8?token=abc")
    assert src["mime_type"] == "application/vnd.apple.mpegurl"
    assert src["can_analyze"] is False
    assert src["is_stream"] is True


def test_job_id_is_stable_and_safe():
    job_id = make_job_id(
        "https://www.youtube.com/watch?v=DFts_bm6sM4",
        extractor="Youtube",
        source_id="DFts_bm6sM4",
        title="Ignored",
    )
    assert job_id == "yt_dfts_bm6sm4"
    assert "/" not in job_id


def test_invalid_url_rejected():
    try:
        normalize_source_url("file:///tmp/demo.mp4")
    except ValueError as exc:
        assert "http" in str(exc)
    else:
        raise AssertionError("Expected invalid URL to be rejected")


def test_llm_output_segments_fill_coverage_gaps():
    llm_segments = [
        {"start": 0.0, "end": 10.0, "label": "main_content", "reason": "content"},
        {"start": 10.4, "end": 15.0, "label": "dead_air", "reason": "pause"},
        {"start": 16.0, "end": 20.0, "label": "main_content", "reason": "content"},
    ]
    out = to_output_segments(llm_segments, video_duration=20.0)
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 20.0
    for prev, nxt in zip(out, out[1:]):
        assert prev["end"] <= nxt["start"]
        assert nxt["start"] - prev["end"] <= 0.001
