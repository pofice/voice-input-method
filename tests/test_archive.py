"""Tests for long-recording archival (engine.archive_recording_*).

The audio copy must happen at stop time — before transcription — so a
network/model failure can never lose the recording (previously the save
only ran after a successful transcription, and the fixed temp filename
meant the next recording overwrote the audio).
"""

from __future__ import annotations

from voice_input_method.engine import (
    archive_recording_audio,
    archive_recording_text,
)


def test_archive_audio_copies_at_stop_time(tmp_path):
    src = tmp_path / "voice_input_audio.wav"
    src.write_bytes(b"RIFFfakewav")
    save_dir = tmp_path / "recordings"

    base = archive_recording_audio(src, duration=65, save_dir=save_dir)

    assert base is not None
    wav = base.with_suffix(".wav")
    assert wav.exists()
    assert wav.read_bytes() == b"RIFFfakewav"
    assert base.name.endswith("_65s")


def test_archive_audio_missing_source_returns_none(tmp_path):
    base = archive_recording_audio(
        tmp_path / "nonexistent.wav", duration=10, save_dir=tmp_path / "recordings"
    )
    assert base is None


def test_archive_audio_survives_source_overwrite(tmp_path):
    """The archived copy must be independent of the fixed temp file."""
    src = tmp_path / "voice_input_audio.wav"
    src.write_bytes(b"LONG_RECORDING")
    save_dir = tmp_path / "recordings"

    base = archive_recording_audio(src, duration=3600, save_dir=save_dir)
    src.write_bytes(b"NEXT_SHORT_ONE")  # next recording overwrites temp file

    assert base.with_suffix(".wav").read_bytes() == b"LONG_RECORDING"


def test_archive_text_writes_next_to_audio(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    base = archive_recording_audio(src, duration=5, save_dir=tmp_path / "rec")

    archive_recording_text(base, "识别结果文字")

    assert base.with_suffix(".txt").read_text(encoding="utf-8") == "识别结果文字"


def test_archive_text_skips_empty(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    base = archive_recording_audio(src, duration=5, save_dir=tmp_path / "rec")

    archive_recording_text(base, "   ")

    assert not base.with_suffix(".txt").exists()


def test_archive_text_none_base_is_noop():
    archive_recording_text(None, "text")  # must not raise
