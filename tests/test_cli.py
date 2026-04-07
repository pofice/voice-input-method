"""Tests for the headless CLI entry point."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from voice_input_method import cli


class TestParser:
    def test_no_args_shows_error(self, capsys):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_info_command(self, capsys):
        rc = cli.main(["info"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "version" in data
        assert "default_offline_model" in data
        assert "default_streaming_model" in data

    def test_transcribe_missing_file(self, capsys):
        rc = cli.main(["transcribe", "/nonexistent/file.wav"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err

    def test_batch_not_a_directory(self, tmp_path, capsys):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("nope")
        rc = cli.main(["batch", str(f)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not a directory" in err

    def test_batch_empty_directory(self, tmp_path, capsys):
        rc = cli.main(["batch", str(tmp_path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "no .wav files" in err


class TestTranscribeWithMocks:
    """Test cmd_transcribe with mocked SpeechRecognizer."""

    def test_transcribe_offline_to_stdout(self, tmp_path, capsys):
        # Create a fake wav file
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            instance = MockRec.return_value
            instance.transcribe.return_value = "你 好 世界"

            rc = cli.main(["transcribe", str(wav)])
            assert rc == 0
            out = capsys.readouterr().out.strip()
            assert "你好世界" == out  # spaces cleaned

    def test_transcribe_offline_to_file(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        out_file = tmp_path / "out.txt"

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = "测试结果"
            rc = cli.main(["transcribe", str(wav), "-o", str(out_file)])
            assert rc == 0
            assert out_file.read_text(encoding="utf-8").strip() == "测试结果"

    def test_transcribe_json_output(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = "结果文本"
            rc = cli.main(["transcribe", str(wav), "--json"])
            assert rc == 0
            data = json.loads(capsys.readouterr().out)
            assert data["text"] == "结果文本"
            assert data["mode"] == "offline"
            assert "elapsed_ms" in data
            assert data["input"] == str(wav)

    def test_transcribe_with_hotwords(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = "测试"
            cli.main(["transcribe", str(wav), "--hotwords", "遍历 数组"])

            # Verify hotwords were passed through
            MockRec.return_value.transcribe.assert_called_once()
            call_args = MockRec.return_value.transcribe.call_args
            assert "遍历 数组" in call_args[0]

    def test_transcribe_no_quantize_flag(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = ""
            cli.main(["transcribe", str(wav), "--no-quantize"])

            # Verify quantize=False was passed to constructor
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs.get("quantize") is False


class TestBatchWithMocks:
    def test_batch_to_jsonl(self, tmp_path, capsys):
        # Create some fake wav files
        for name in ["a.wav", "b.wav", "c.wav"]:
            (tmp_path / name).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        out_file = tmp_path / "results.jsonl"

        with patch("voice_input_method.cli.SpeechRecognizer") as MockRec:
            MockRec.return_value.transcribe.side_effect = [
                "结果一", "结果二", "结果三"
            ]
            rc = cli.main(["batch", str(tmp_path), "-o", str(out_file)])
            assert rc == 0

        lines = out_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        assert records[0]["file"] == "a.wav"
        assert records[0]["text"] == "结果一"
        assert records[1]["text"] == "结果二"
        assert records[2]["text"] == "结果三"
        assert all("elapsed_ms" in r for r in records)


class TestRealAudio:
    """Integration test using real fixture audio file."""

    pytestmark = pytest.mark.integration

    def test_cli_transcribe_real_audio(self, capsys):
        fixture = Path(__file__).parent / "fixtures" / "chinese_speech_16k.wav"
        if not fixture.exists():
            pytest.skip("Test fixture not present")

        rc = cli.main(["transcribe", str(fixture)])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        # Expected content includes these keywords
        for kw in ["今天", "天气", "公园"]:
            assert kw in out, f"Expected '{kw}' in CLI output: {out}"
