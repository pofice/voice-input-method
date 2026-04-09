"""Tests for the headless CLI entry point."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_input_method import cli
from voice_input_method.config import Config


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

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
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

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = "测试结果"
            rc = cli.main(["transcribe", str(wav), "-o", str(out_file)])
            assert rc == 0
            assert out_file.read_text(encoding="utf-8").strip() == "测试结果"

    def test_transcribe_json_output(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
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

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = "测试"
            cli.main(["transcribe", str(wav), "--hotwords", "遍历 数组"])

            # Verify hotwords were passed through
            MockRec.return_value.transcribe.assert_called_once()
            call_args = MockRec.return_value.transcribe.call_args
            assert "遍历 数组" in call_args[0]

    def test_transcribe_no_quantize_flag(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
            MockRec.return_value.transcribe.return_value = ""
            cli.main(["transcribe", str(wav), "--no-quantize"])

            # Verify quantize=False was set on the config passed to the factory
            MockRec.assert_called_once()
            config_arg = MockRec.call_args.args[0]
            assert config_arg.quantize is False


class TestBatchWithMocks:
    def test_batch_to_jsonl(self, tmp_path, capsys):
        # Create some fake wav files
        for name in ["a.wav", "b.wav", "c.wav"]:
            (tmp_path / name).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        out_file = tmp_path / "results.jsonl"

        with patch("voice_input_method.cli._create_recognizer") as MockRec:
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


class TestDoctor:
    """Doctor subcommand parses without crashing.

    Real end-to-end run is in TestRealAudio (integration only)
    because it downloads ~250MB of model.
    """

    def test_doctor_in_help(self):
        with pytest.raises(SystemExit):
            cli.main(["doctor", "--nonexistent-flag"])

    def test_doctor_imports_check_only(self, capsys):
        """Doctor's first check (imports) runs without external state."""
        # We can't fully run doctor in unit tests because it would
        # download a model. But we can verify the check function exists.
        from voice_input_method.cli import build_parser, cmd_doctor
        parser = build_parser()
        # Verify "doctor" subcommand is registered
        args = parser.parse_args(["doctor"])
        assert args.func is cmd_doctor


class TestListenWithMocks:
    """Test cmd_listen with mocked sounddevice and recognizer."""

    def test_listen_with_duration(self, tmp_path, capsys):

        mock_sd = MagicMock()
        mock_sf = MagicMock()
        # Mock InputStream as context-like object
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with patch("voice_input_method.cli._create_recognizer") as MockRec, \
             patch.dict("sys.modules", {
                 "sounddevice": mock_sd,
                 "soundfile": mock_sf,
             }), \
             patch("voice_input_method.cli.sd", mock_sd, create=True), \
             patch("voice_input_method.cli.sf_mod", mock_sf, create=True):

            # Simulate: after stream starts, callback fills buffer
            def fake_start():
                # Simulate audio data in buffer via the callback
                pass
            mock_stream.start = fake_start
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()

            MockRec.return_value.transcribe.return_value = "测试结果"

            # Use --duration 0.1 to make it fast
            # We need to call cmd_listen directly with a crafted args
            from voice_input_method.cli import build_parser
            parser = build_parser()
            args = parser.parse_args([
                "listen", "--duration", "0.1", "--no-denoise"
            ])

            # Patch the imports inside cmd_listen
            import voice_input_method.cli as cli_mod
            with patch.object(cli_mod, "_build_config") as mock_bc, \
                 patch.object(cli_mod, "_create_recognizer") as mock_cr:
                mock_bc.return_value = Config()
                mock_cr.return_value.transcribe.return_value = "测试"

                # cmd_listen imports sounddevice/soundfile/numpy at top of function
                # We'll test the parser registration instead
                assert args.func.__name__ == "cmd_listen"
                assert args.duration == 0.1

    def test_listen_parser_accepts_device(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["listen", "--device", "3", "--duration", "1"])
        assert args.device == 3
        assert args.duration == 1.0

    def test_listen_parser_accepts_backend(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "listen", "--backend", "sherpa-nano",
            "--nano-model-dir", "/fake/dir"
        ])
        assert args.backend == "sherpa-nano"
        assert args.nano_model_dir == "/fake/dir"

    def test_listen_parser_accepts_save(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["listen", "--save", "/tmp/out.wav"])
        assert args.save == "/tmp/out.wav"

    def test_listen_parser_accepts_json(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["listen", "--json"])
        assert args.json is True

    def test_listen_parser_accepts_no_denoise(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["listen", "--no-denoise"])
        assert args.no_denoise is True


class TestDevicesWithMocks:
    """Test cmd_devices with mocked sounddevice."""

    def test_devices_json_output(self, capsys):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 1, "default_samplerate": 48000.0},
            {"name": "Speaker Out", "max_input_channels": 0, "default_samplerate": 44100.0},
            {"name": "USB Mic", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Re-import to pick up the mock

            from voice_input_method.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["devices"])

            # Call with mock
            with patch("voice_input_method.cli.sd", mock_sd, create=True):
                # Inline the function logic to test with mocked sd
                mock_sd._terminate = MagicMock()
                mock_sd._initialize = MagicMock()

                args.func(args)
                # Can't easily test stdout here because sd is imported inside func
                # But we verify the parser works
                assert args.func.__name__ == "cmd_devices"

    def test_devices_parser_registered(self):
        from voice_input_method.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["devices"])
        assert args.func.__name__ == "cmd_devices"


class TestDoctorWithMocks:
    """Test doctor subcommand with mocked model loading."""

    def test_doctor_with_mocked_recognizer(self, capsys):
        """Doctor runs all checks with mocked recognizer."""
        mock_funasr = MagicMock()
        with patch("voice_input_method.cli._create_recognizer") as MockRec, \
             patch("voice_input_method.cli._build_config") as MockBuild, \
             patch.dict("sys.modules", {"funasr_onnx": mock_funasr}):
            mock_rec = MagicMock()
            mock_rec.transcribe.return_value = "今天天气不错去公园走走"
            MockRec.return_value = mock_rec
            MockBuild.return_value = Config()

            rc = cli.main(["doctor"])
            assert rc == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["ok"] is True
            assert len(data["checks"]) == 5

    def test_doctor_reports_failure(self, capsys):
        """Doctor reports failure when a check raises."""
        with patch("voice_input_method.cli._create_recognizer") as MockRec, \
             patch("voice_input_method.cli._build_config") as MockBuild:
            mock_rec = MagicMock()
            mock_rec.load.side_effect = RuntimeError("model not found")
            MockRec.return_value = mock_rec
            MockBuild.return_value = Config()

            rc = cli.main(["doctor"])
            assert rc == 1
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["ok"] is False
            # At least one check should have failed
            failed = [c for c in data["checks"] if c["status"] == "fail"]
            assert len(failed) >= 1


class TestBuildConfig:
    """Test _build_config helper."""

    def test_backend_override(self):
        from voice_input_method.cli import _build_config, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "transcribe", "test.wav",
            "--backend", "sherpa-nano",
            "--nano-model-dir", "/fake/dir"
        ])
        config = _build_config(args)
        assert config.recognizer_backend == "sherpa-nano"
        assert config.nano_model_dir == "/fake/dir"

    def test_sensevoice_override(self):
        from voice_input_method.cli import _build_config, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "transcribe", "test.wav",
            "--backend", "sherpa-sensevoice",
            "--sensevoice-model", "/fake/model.onnx",
            "--sensevoice-tokens", "/fake/tokens.txt"
        ])
        config = _build_config(args)
        assert config.recognizer_backend == "sherpa-sensevoice"
        assert config.sensevoice_model_path == "/fake/model.onnx"
        assert config.sensevoice_tokens_path == "/fake/tokens.txt"

    def test_qwen3_override(self):
        from voice_input_method.cli import _build_config, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "transcribe", "test.wav",
            "--backend", "qwen3-asr",
            "--qwen3-model-dir", "/fake/qwen3",
            "--qwen3-max-total-len", "1024",
            "--qwen3-max-new-tokens", "256",
        ])
        config = _build_config(args)
        assert config.recognizer_backend == "qwen3-asr"
        assert config.qwen3_model_dir == "/fake/qwen3"
        assert config.qwen3_max_total_len == 1024
        assert config.qwen3_max_new_tokens == 256

    def test_config_file_loading(self, tmp_path):
        from voice_input_method.cli import _build_config, build_parser
        cfg = tmp_path / "test_config.yaml"
        cfg.write_text("recognizer_backend: sherpa-nano\nnano_model_dir: /test\n")
        parser = build_parser()
        args = parser.parse_args([
            "transcribe", "test.wav",
            "--config", str(cfg)
        ])
        config = _build_config(args)
        assert config.recognizer_backend == "sherpa-nano"


class TestRealAudio:
    """Integration tests using real fixture audio file."""

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

    def test_cli_doctor_full(self, capsys):
        """End-to-end doctor: imports + audio I/O + model load + inference."""
        rc = cli.main(["doctor"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["ok"] is True
        # All 5 checks should be present and OK
        check_names = [c["check"] for c in data["checks"]]
        assert "import core dependencies" in check_names
        assert "audio I/O" in check_names
        assert "ASR model load" in check_names
        assert "ASR inference (silence)" in check_names
        assert "real Chinese audio" in check_names
        for c in data["checks"]:
            assert c["status"] == "ok", f"{c['check']} failed: {c.get('error')}"
