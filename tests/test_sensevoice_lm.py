"""Tests for SenseVoice + LM rescoring backend."""

from unittest.mock import patch

import numpy as np
import pytest

from voice_input_method.config import Config
from voice_input_method.factory import ConfigError, _create_recognizer


class TestSenseVoiceLMUnit:
    """Unit tests for SenseVoiceLMRecognizer internals (no model needed)."""

    def _make_recognizer(self, **kwargs):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        defaults = dict(
            model_path="/fake/model.onnx",
            tokens_path="/fake/tokens.txt",
            lm_path="",
        )
        defaults.update(kwargs)
        return SenseVoiceLMRecognizer(**defaults)

    def test_log_softmax_basic(self):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
        log_probs = SenseVoiceLMRecognizer._log_softmax(logits)
        # Should sum to ~1 after exp
        probs = np.exp(log_probs)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_log_softmax_numerical_stability(self):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        # Large values that would overflow naive softmax
        logits = np.array([[1000.0, 999.0, 998.0]], dtype=np.float32)
        log_probs = SenseVoiceLMRecognizer._log_softmax(logits)
        assert not np.any(np.isnan(log_probs))
        assert not np.any(np.isinf(log_probs))

    def test_clean_cjk_spaces(self):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        assert SenseVoiceLMRecognizer._clean_cjk_spaces("你 好 世 界") == "你好世界"
        assert SenseVoiceLMRecognizer._clean_cjk_spaces("hello world") == "hello world"
        assert SenseVoiceLMRecognizer._clean_cjk_spaces("你好 world") == "你好 world"
        assert SenseVoiceLMRecognizer._clean_cjk_spaces("中 文 abc 测 试") == "中文 abc 测试"

    def test_load_vocab(self, tmp_path):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        tokens_file = tmp_path / "tokens.txt"
        tokens_file.write_text("<unk> 0\n<s> 1\n</s> 2\n你 3\n好 4\n")
        vocab = SenseVoiceLMRecognizer._load_vocab(str(tokens_file))
        assert vocab == ["<unk>", "<s>", "</s>", "你", "好"]

    def test_apply_lfr_shape(self):
        rec = self._make_recognizer()
        rec._lfr_window_size = 7
        rec._lfr_window_shift = 6
        # 100 frames of 80-dim fbank
        features = np.random.randn(100, 80).astype(np.float32)
        lfr = rec._apply_lfr(features)
        # Output should be [T', 560] where 560 = 7 * 80
        assert lfr.shape[1] == 560
        assert lfr.shape[0] > 0

    def test_apply_cmvn(self):
        rec = self._make_recognizer()
        rec._neg_mean = np.ones(560, dtype=np.float32) * (-2.0)
        rec._inv_stddev = np.ones(560, dtype=np.float32) * 0.5
        features = np.zeros((10, 560), dtype=np.float32)
        result = rec._apply_cmvn(features)
        # (0 + (-2)) * 0.5 = -1.0
        np.testing.assert_allclose(result, -1.0)

    def test_resample(self):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        # 44100 Hz → 16000 Hz, 1 second
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32)
        resampled = SenseVoiceLMRecognizer._resample(audio, 44100, 16000)
        assert len(resampled) == 16000

    def test_extract_unigrams_from_arpa(self, tmp_path):
        from voice_input_method.recognition.sensevoice_lm import SenseVoiceLMRecognizer
        arpa_file = tmp_path / "test.arpa"
        arpa_file.write_text(
            "\\data\\\nngram 1=5\n\n\\1-grams:\n"
            "-1.0\t<unk>\t0\n-0.5\t<s>\t-1.0\n-0.3\t</s>\t0\n"
            "-1.2\t你\t-0.5\n-1.3\t好\t-0.4\n\n\\2-grams:\n"
            "-0.5\t你 好\n\n\\end\\\n"
        )
        unigrams = SenseVoiceLMRecognizer._extract_unigrams_from_arpa(str(arpa_file))
        assert "你" in unigrams
        assert "好" in unigrams
        assert "<unk>" not in unigrams
        assert "<s>" not in unigrams

    def test_transcribe_returns_empty_when_not_loaded(self):
        rec = self._make_recognizer()
        assert rec.transcribe("/fake/audio.wav") == ""


class TestSenseVoiceLMFactory:
    """Test factory integration for sensevoice-lm backend."""

    def test_sensevoice_lm_missing_model_path(self):
        config = Config(recognizer_backend="sensevoice-lm")
        with pytest.raises(ConfigError, match="sensevoice_model_path"):
            _create_recognizer(config)

    def test_sensevoice_lm_missing_tokens_path(self):
        config = Config(
            recognizer_backend="sensevoice-lm",
            sensevoice_model_path="/fake/model.onnx",
        )
        with pytest.raises(ConfigError, match="sensevoice_tokens_path"):
            _create_recognizer(config)

    def test_sensevoice_lm_passes_all_params(self):
        config = Config(
            recognizer_backend="sensevoice-lm",
            sensevoice_model_path="/fake/model.onnx",
            sensevoice_tokens_path="/fake/tokens.txt",
            sensevoice_language="zh",
            sensevoice_lm_path="/fake/lm.bin",
            sensevoice_lm_alpha=0.7,
            sensevoice_lm_beta=1.5,
            sensevoice_lm_beam_width=30,
        )
        with patch(
            "voice_input_method.recognition.sensevoice_lm.SenseVoiceLMRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            MockRec.assert_called_once()
            kwargs = MockRec.call_args.kwargs
            assert kwargs["model_path"] == "/fake/model.onnx"
            assert kwargs["tokens_path"] == "/fake/tokens.txt"
            assert kwargs["lm_path"] == "/fake/lm.bin"
            assert kwargs["language"] == "zh"
            assert kwargs["lm_alpha"] == 0.7
            assert kwargs["lm_beta"] == 1.5
            assert kwargs["beam_width"] == 30

    def test_sensevoice_lm_default_params(self):
        config = Config(
            recognizer_backend="sensevoice-lm",
            sensevoice_model_path="/fake/model.onnx",
            sensevoice_tokens_path="/fake/tokens.txt",
        )
        with patch(
            "voice_input_method.recognition.sensevoice_lm.SenseVoiceLMRecognizer"
        ) as MockRec:
            _create_recognizer(config)
            kwargs = MockRec.call_args.kwargs
            assert kwargs["lm_path"] == ""
            assert kwargs["lm_alpha"] == 0.5
            assert kwargs["lm_beta"] == 1.0
            assert kwargs["beam_width"] == 20


class TestSenseVoiceLMConfig:
    """Test Config dataclass has sensevoice-lm fields."""

    def test_config_defaults(self):
        config = Config()
        assert config.sensevoice_lm_path == ""
        assert config.sensevoice_lm_alpha == 0.5
        assert config.sensevoice_lm_beta == 1.0
        assert config.sensevoice_lm_beam_width == 20

    def test_config_from_yaml(self, tmp_path):
        from voice_input_method.config import load_config
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "recognizer_backend: sensevoice-lm\n"
            "sensevoice_model_path: /path/to/model.onnx\n"
            "sensevoice_tokens_path: /path/to/tokens.txt\n"
            "sensevoice_lm_path: /path/to/lm.bin\n"
            "sensevoice_lm_alpha: 0.8\n"
            "sensevoice_lm_beta: 2.0\n"
            "sensevoice_lm_beam_width: 50\n"
        )
        config = load_config(str(yaml_file))
        assert config.recognizer_backend == "sensevoice-lm"
        assert config.sensevoice_lm_path == "/path/to/lm.bin"
        assert config.sensevoice_lm_alpha == 0.8
        assert config.sensevoice_lm_beta == 2.0
        assert config.sensevoice_lm_beam_width == 50
