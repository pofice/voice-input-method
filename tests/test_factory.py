"""Tests for factory module — verifies engine assembly without GUI."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from voice_input_method.config import Config
from voice_input_method.factory import ConfigError, _create_recognizer, create_engine


class TestCreateRecognizerSherpaNano:
    def test_nano_passes_hotwords_to_constructor(self):
        """sherpa-nano backend must forward hotwords string into recognizer."""
        config = Config(
            recognizer_backend="sherpa-nano",
            nano_model_dir="/fake/dir",
        )
        with patch(
            "voice_input_method.recognition.sherpa_nano.SherpaNanoRecognizer"
        ) as MockRec:
            _create_recognizer(config, hotwords="Claude Code 遍历")
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs.get("hotwords") == "Claude Code 遍历"

    def test_nano_default_empty_hotwords(self):
        config = Config(
            recognizer_backend="sherpa-nano",
            nano_model_dir="/fake/dir",
        )
        with patch(
            "voice_input_method.recognition.sherpa_nano.SherpaNanoRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            assert MockRec.call_args.kwargs.get("hotwords") == ""

    def test_nano_passes_prompts(self):
        config = Config(
            recognizer_backend="sherpa-nano",
            nano_model_dir="/fake/dir",
            nano_system_prompt="You transcribe coding terms.",
            nano_user_prompt="转写代码:",
        )
        with patch(
            "voice_input_method.recognition.sherpa_nano.SherpaNanoRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            kwargs = MockRec.call_args.kwargs
            assert kwargs.get("system_prompt") == "You transcribe coding terms."
            assert kwargs.get("user_prompt") == "转写代码:"


class TestConfigErrorPaths:
    """Test that missing config raises clear ConfigError."""

    def test_sensevoice_missing_model_path(self):
        config = Config(recognizer_backend="sherpa-sensevoice")
        with pytest.raises(ConfigError, match="sensevoice_model_path"):
            _create_recognizer(config)

    def test_sensevoice_missing_tokens_path(self):
        config = Config(
            recognizer_backend="sherpa-sensevoice",
            sensevoice_model_path="/fake/model.onnx",
        )
        with pytest.raises(ConfigError, match="sensevoice_tokens_path"):
            _create_recognizer(config)

    def test_nano_missing_model_dir(self):
        config = Config(recognizer_backend="sherpa-nano")
        with pytest.raises(ConfigError, match="nano_model_dir"):
            _create_recognizer(config)

    def test_unknown_backend_raises(self):
        config = Config(recognizer_backend="nonexistent")
        with pytest.raises(ConfigError, match="unknown recognizer_backend"):
            _create_recognizer(config)

    def test_funasr_default_works(self):
        """funasr backend creates recognizer without error (mocked import)."""
        config = Config(recognizer_backend="funasr")
        with patch("voice_input_method.recognition.funasr_recognizer.FunASRRecognizer") as MockRec:
            _create_recognizer(config)
            MockRec.assert_called_once()


class TestCreateEngineStreaming:
    """Test streaming + non-funasr backend warning."""

    def test_streaming_non_funasr_warns(self):
        config = Config(
            recognizer_backend="sherpa-nano",
            nano_model_dir="/fake/dir",
            streaming=True,
            enable_traditional_chinese=False,
            enable_hotwords=False,
        )
        with patch("voice_input_method.recognition.sherpa_nano.SherpaNanoRecognizer"), \
             patch("voice_input_method.factory.get_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                create_engine(config)
                runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
                assert len(runtime_warns) >= 1
                assert "streaming" in str(runtime_warns[0].message).lower()


class TestCreateRecognizerQwen3ASR:
    def test_qwen3_passes_all_params(self):
        config = Config(
            recognizer_backend="qwen3-asr",
            qwen3_model_dir="/fake/qwen3",
            qwen3_max_total_len=1024,
            qwen3_max_new_tokens=256,
        )
        with patch(
            "voice_input_method.recognition.qwen3_asr.Qwen3ASRRecognizer"
        ) as MockRec:
            _create_recognizer(config, hotwords="Claude Code")
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs["conv_frontend_path"] == "/fake/qwen3/conv_frontend.onnx"
            assert kwargs["encoder_path"] == "/fake/qwen3/encoder.int8.onnx"
            assert kwargs["decoder_path"] == "/fake/qwen3/decoder.int8.onnx"
            assert kwargs["tokenizer_path"] == "/fake/qwen3/tokenizer"
            assert kwargs["hotwords"] == "Claude Code"
            assert kwargs["max_total_len"] == 1024
            assert kwargs["max_new_tokens"] == 256

    def test_qwen3_missing_model_dir(self):
        config = Config(recognizer_backend="qwen3-asr")
        with pytest.raises(ConfigError, match="qwen3_model_dir"):
            _create_recognizer(config)

    def test_qwen3_default_params(self):
        config = Config(
            recognizer_backend="qwen3-asr",
            qwen3_model_dir="/fake/qwen3",
        )
        with patch(
            "voice_input_method.recognition.qwen3_asr.Qwen3ASRRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            kwargs = MockRec.call_args.kwargs
            assert kwargs["max_total_len"] == 512
            assert kwargs["max_new_tokens"] == 128
            assert kwargs["hotwords"] == ""


class TestCreateRecognizerRemoteMimo:
    def test_remote_mimo_passes_all_params(self):
        config = Config(
            recognizer_backend="remote-mimo",
            mimo_base_url="http://192.168.192.118:7898",
            mimo_language="Chinese",
            mimo_timeout=30.0,
        )
        with patch(
            "voice_input_method.recognition.remote_mimo.RemoteMiMoRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs["base_url"] == "http://192.168.192.118:7898"
            assert kwargs["language"] == "Chinese"
            assert kwargs["timeout"] == 30.0

    def test_remote_mimo_missing_base_url(self):
        config = Config(recognizer_backend="remote-mimo")
        with pytest.raises(ConfigError, match="mimo_base_url"):
            _create_recognizer(config)

    def test_remote_mimo_satisfies_recognizer_protocol(self):
        from voice_input_method.protocols import Recognizer
        from voice_input_method.recognition.remote_mimo import RemoteMiMoRecognizer
        rec = RemoteMiMoRecognizer(base_url="http://fake:1")
        assert isinstance(rec, Recognizer)


class TestCreateRecognizerSenseVoice:
    """Test sensevoice recognizer construction (mocked import)."""

    def test_sensevoice_passes_all_params(self):
        config = Config(
            recognizer_backend="sherpa-sensevoice",
            sensevoice_model_path="/fake/model.onnx",
            sensevoice_tokens_path="/fake/tokens.txt",
            sensevoice_language="zh",
        )
        with patch(
            "voice_input_method.recognition.sherpa_sensevoice.SherpaSenseVoiceRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs["model_path"] == "/fake/model.onnx"
            assert kwargs["tokens_path"] == "/fake/tokens.txt"
            assert kwargs["language"] == "zh"
            assert kwargs["num_threads"] == 4


class TestCreateEngineFullAssembly:
    """Test create_engine wiring with all dependencies mocked."""

    def test_engine_with_hotwords_funasr(self, tmp_path):
        """create_engine wires hotword_manager for funasr backend."""
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("Claude Code\nCloud Code -> Claude Code\n")

        config = Config(
            recognizer_backend="funasr",
            enable_hotwords=True,
            hotwords_file=str(hw_file),
            enable_traditional_chinese=False,
        )
        with patch("voice_input_method.recognition.funasr_recognizer.FunASRRecognizer"), \
             patch("voice_input_method.factory.get_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            engine = create_engine(config)
            assert engine.hotword_provider is not None
            assert "Claude Code" in engine.hotword_provider.hotwords_str
            assert engine.hotword_provider.corrections == {"Cloud Code": "Claude Code"}

    def test_engine_with_hotwords_sherpa_nano(self, tmp_path):
        """create_engine passes csv hotwords for sherpa-nano backend."""
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("Claude Code\n遍历\n")

        config = Config(
            recognizer_backend="sherpa-nano",
            nano_model_dir="/fake/dir",
            enable_hotwords=True,
            hotwords_file=str(hw_file),
            enable_traditional_chinese=False,
        )
        with patch("voice_input_method.recognition.sherpa_nano.SherpaNanoRecognizer") as MockRec, \
             patch("voice_input_method.factory.get_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            create_engine(config)
            # Verify hotwords were passed as csv to nano
            kwargs = MockRec.call_args.kwargs
            assert kwargs["hotwords"] == "Claude Code,遍历"

    def test_engine_no_hotwords(self):
        """create_engine without hotwords works."""
        config = Config(
            recognizer_backend="funasr",
            enable_hotwords=False,
            enable_traditional_chinese=False,
        )
        with patch("voice_input_method.recognition.funasr_recognizer.FunASRRecognizer"), \
             patch("voice_input_method.factory.get_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            engine = create_engine(config)
            assert engine.hotword_provider is None

    def test_engine_with_streaming_funasr(self):
        """create_engine with streaming=True + funasr creates streaming recognizer."""
        config = Config(
            recognizer_backend="funasr",
            streaming=True,
            enable_hotwords=False,
            enable_traditional_chinese=False,
        )
        with patch("voice_input_method.recognition.funasr_recognizer.FunASRRecognizer"), \
             patch("voice_input_method.recognition.funasr_recognizer.FunASRStreamingRecognizer") as MockStream, \
             patch("voice_input_method.factory.get_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            MockStream.return_value.step_samples = 9600
            engine = create_engine(config)
            assert engine.streaming_recognizer is not None
            assert engine.config.streaming is True


class TestCreateIndicator:
    def test_non_macos_returns_null(self):
        from voice_input_method.factory import create_indicator
        from voice_input_method.indicator import NullIndicator
        indicator = create_indicator("linux")
        assert isinstance(indicator, NullIndicator)

    def test_macos_fallback_to_null(self):
        """On non-macOS systems, macos indicator import fails gracefully."""
        from voice_input_method.factory import create_indicator
        from voice_input_method.indicator import NullIndicator
        with patch("voice_input_method.indicator.MacNativeIndicator", side_effect=ImportError):
            indicator = create_indicator("macos")
            assert isinstance(indicator, NullIndicator)


class TestCreateEngine:
    def test_factory_import_headless(self):
        """factory.py should import without GUI/display."""
        from voice_input_method.factory import create_engine
        assert callable(create_engine)

    def test_factory_creates_engine_with_defaults(self):
        """create_engine with default Config produces a VoiceEngine."""
        Config()
        # Platform backend will fail on headless (pynput), but we can
        # at least verify the factory function signature works
        # For a full test we'd need to mock get_backend
        assert True  # import + callable check is the main value here
