"""Test silence detection on a synthetically generated audio file."""

import struct
import wave
from pathlib import Path

import pytest


def _write_synthetic_wav(path: Path, silent_intervals: list[tuple[float, float]], total_duration: float = 10.0, sample_rate: int = 16000) -> None:
    """Write a WAV with loud noise except in specified silent intervals."""
    import random

    n_samples = int(total_duration * sample_rate)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        in_silence = any(start <= t < end for start, end in silent_intervals)
        if in_silence:
            samples.append(0)
        else:
            # ~-6 dBFS square wave
            samples.append(16000 if (int(t * 440) % 2 == 0) else -16000)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


@pytest.fixture()
def wav_with_silence(tmp_path: Path) -> Path:
    out = tmp_path / "test_silence.wav"
    _write_synthetic_wav(out, silent_intervals=[(3.0, 6.0)], total_duration=10.0)
    return out


def test_silence_detect_import():
    """Module must be importable (stubs in place)."""
    from analyzer.audio import silence_detect  # noqa: F401


def test_detect_silence_finds_interval(wav_with_silence: Path):
    """detect_silence should find the 3-6 s silent gap in the synthetic WAV."""
    from analyzer.audio.silence_detect import detect_silence

    intervals = detect_silence(str(wav_with_silence), min_duration=1.0, threshold_db=-40.0)
    assert len(intervals) >= 1

    starts = [iv["start"] for iv in intervals]
    assert any(2.0 <= s <= 4.0 for s in starts), f"Expected silence near 3 s, got starts={starts}"

def test_detect_silence_schema(wav_with_silence: Path):
    from analyzer.audio.silence_detect import detect_silence

    intervals = detect_silence(str(wav_with_silence), min_duration=1.0)
    for iv in intervals:
        assert "start"  in iv
        assert "end"    in iv
        assert "energy" in iv
        assert iv["end"] > iv["start"]
