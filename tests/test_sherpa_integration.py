"""Integration tests for sherpa-onnx backends (SenseVoice + Fun-ASR-Nano).

These tests require sherpa-onnx and pre-downloaded models.
Run with:  pytest tests/test_sherpa_integration.py -v
Skip with: pytest -m "not sherpa"

Model locations (relative to repo root):
- SenseVoice: benchmark/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
- Fun-ASR-Nano: benchmark/sherpa-onnx-funasr-nano-int8-2025-12-30/
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.sherpa

REPO_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHINESE_SPEECH_WAV = FIXTURES_DIR / "chinese_speech_16k.wav"
CHINESE_SHORT_WAV = FIXTURES_DIR / "chinese_short_16k.wav"

# SenseVoice model
SENSEVOICE_DIR = REPO_ROOT / "benchmark" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSEVOICE_MODEL = SENSEVOICE_DIR / "model.int8.onnx"
SENSEVOICE_TOKENS = SENSEVOICE_DIR / "tokens.txt"

# Fun-ASR-Nano model
NANO_DIR = REPO_ROOT / "benchmark" / "sherpa-onnx-funasr-nano-int8-2025-12-30"
NANO_ENCODER = NANO_DIR / "encoder_adaptor.int8.onnx"
NANO_LLM = NANO_DIR / "llm.int8.onnx"
NANO_EMBEDDING = NANO_DIR / "embedding.int8.onnx"
NANO_TOKENIZER = NANO_DIR / "Qwen3-0.6B"

EXPECTED_KEYWORDS = ["今天", "天气", "公园"]


def _sherpa_available():
    try:
        import sherpa_onnx  # noqa
        return True
    except ImportError:
        return False


# ---- SenseVoice ----

class TestSenseVoice:
    """Integration tests for SherpaSenseVoiceRecognizer."""

    @pytest.fixture(scope="class")
    def recognizer(self):
        if not _sherpa_available():
            pytest.skip("sherpa-onnx not installed")
        if not SENSEVOICE_MODEL.exists():
            pytest.skip(f"SenseVoice model not found at {SENSEVOICE_DIR}")
        if not CHINESE_SPEECH_WAV.exists():
            pytest.skip("Test audio fixtures not found")

        from voice_input_method.recognition.sherpa_sensevoice import SherpaSenseVoiceRecognizer
        rec = SherpaSenseVoiceRecognizer(
            model_path=str(SENSEVOICE_MODEL),
            tokens_path=str(SENSEVOICE_TOKENS),
            language="zh",
            use_itn=True,
            num_threads=4,
        )
        rec.load()
        return rec

    def test_load_succeeds(self, recognizer):
        assert recognizer._recognizer is not None

    def test_transcribe_chinese_speech(self, recognizer):
        text = recognizer.transcribe(str(CHINESE_SPEECH_WAV))
        print(f"SenseVoice transcription: {text}")
        assert text, "Transcription should not be empty"
        for keyword in EXPECTED_KEYWORDS:
            assert keyword in text, f"Expected '{keyword}' in: {text}"

    def test_transcribe_short_clip(self, recognizer):
        text = recognizer.transcribe(str(CHINESE_SHORT_WAV))
        print(f"SenseVoice short: {text}")
        assert text, "Short clip transcription should not be empty"
        assert "今天" in text or "天气" in text

    def test_no_metadata_tags_in_output(self, recognizer):
        """SenseVoice tags like <|zh|><|NEUTRAL|> should be stripped."""
        text = recognizer.transcribe(str(CHINESE_SPEECH_WAV))
        assert "<|" not in text, f"Metadata tags should be stripped: {text}"

    def test_warmup_does_not_crash(self, recognizer):
        recognizer.warmup(str(CHINESE_SHORT_WAV))

    def test_satisfies_recognizer_protocol(self):
        if not _sherpa_available():
            pytest.skip("sherpa-onnx not installed")
        from voice_input_method.protocols import Recognizer
        from voice_input_method.recognition.sherpa_sensevoice import SherpaSenseVoiceRecognizer
        rec = SherpaSenseVoiceRecognizer(model_path="x", tokens_path="x")
        assert isinstance(rec, Recognizer)


# ---- Fun-ASR-Nano ----

class TestFunASRNano:
    """Integration tests for SherpaNanoRecognizer."""

    @pytest.fixture(scope="class")
    def recognizer(self):
        if not _sherpa_available():
            pytest.skip("sherpa-onnx not installed")
        if not NANO_ENCODER.exists():
            pytest.skip(f"Fun-ASR-Nano model not found at {NANO_DIR}")
        if not CHINESE_SPEECH_WAV.exists():
            pytest.skip("Test audio fixtures not found")

        from voice_input_method.recognition.sherpa_nano import SherpaNanoRecognizer
        rec = SherpaNanoRecognizer(
            encoder_adaptor_path=str(NANO_ENCODER),
            llm_path=str(NANO_LLM),
            embedding_path=str(NANO_EMBEDDING),
            tokenizer_path=str(NANO_TOKENIZER),
            language="zh",
            num_threads=4,
        )
        rec.load()
        return rec

    def test_load_succeeds(self, recognizer):
        assert recognizer._recognizer is not None

    def test_transcribe_chinese_speech(self, recognizer):
        text = recognizer.transcribe(str(CHINESE_SPEECH_WAV))
        print(f"Fun-ASR-Nano transcription: {text}")
        assert text, "Transcription should not be empty"
        for keyword in EXPECTED_KEYWORDS:
            assert keyword in text, f"Expected '{keyword}' in: {text}"

    def test_transcribe_short_clip(self, recognizer):
        text = recognizer.transcribe(str(CHINESE_SHORT_WAV))
        print(f"Fun-ASR-Nano short: {text}")
        assert text, "Short clip transcription should not be empty"

    def test_warmup_does_not_crash(self, recognizer):
        recognizer.warmup(str(CHINESE_SHORT_WAV))

    def test_satisfies_recognizer_protocol(self):
        if not _sherpa_available():
            pytest.skip("sherpa-onnx not installed")
        from voice_input_method.protocols import Recognizer
        from voice_input_method.recognition.sherpa_nano import SherpaNanoRecognizer
        rec = SherpaNanoRecognizer(
            encoder_adaptor_path="x", llm_path="x",
            embedding_path="x", tokenizer_path="x",
        )
        assert isinstance(rec, Recognizer)
