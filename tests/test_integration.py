"""Integration tests with real FunASR models and Chinese audio.

These tests download and run the actual Paraformer ONNX model against
real Chinese speech audio. They verify the full pipeline end-to-end.

Run with:  pytest tests/test_integration.py -v
Skip with: pytest -m "not integration"

Requirements:
- Internet access (first run downloads ~250MB model from ModelScope)
- ~500MB disk for model cache (~/.cache/modelscope/)
- ~30s for first run (model download), ~5s subsequent runs
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.mocks import MockPaster, MockRecorder
from voice_input_method.engine import EngineConfig, VoiceEngine
from voice_input_method.recognition.funasr_recognizer import FunASRRecognizer as SpeechRecognizer
from voice_input_method.recognition.funasr_recognizer import (
    FunASRStreamingRecognizer as StreamingRecognizer,
)
from voice_input_method.text_processing import clean_spaces

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHINESE_SPEECH_WAV = FIXTURES_DIR / "chinese_speech_16k.wav"
CHINESE_SHORT_WAV = FIXTURES_DIR / "chinese_short_16k.wav"
SILENCE_WAV = FIXTURES_DIR / "silence_16k.wav"

# Model IDs (auto-download from ModelScope)
SEACO_MODEL = "pofice/speech_seaco_paraformer_large_onnx"
STREAMING_MODEL = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"

# Expected content in the test audio: "今天天气真不错，我们一起去公园散步吧"
EXPECTED_KEYWORDS = ["今天", "天气", "不错", "公园", "散步"]


@pytest.fixture(scope="module")
def paraformer():
    """Load the SeacoParaformer model once for all tests in this module."""
    recognizer = SpeechRecognizer(
        model_dir=SEACO_MODEL,
        quantize=True,
    )
    recognizer.load()
    return recognizer


class TestOfflineRecognition:
    """Test real offline ASR with Paraformer model."""

    def test_transcribe_chinese_speech(self, paraformer):
        """Full audio → text transcription of Chinese speech."""
        text = paraformer.transcribe(str(CHINESE_SPEECH_WAV))
        text = clean_spaces(text)
        print(f"Transcription: {text}")

        for keyword in EXPECTED_KEYWORDS:
            assert keyword in text, f"Expected '{keyword}' in transcription: {text}"

    def test_transcribe_short_clip(self, paraformer):
        """Shorter audio clip still produces meaningful output."""
        text = paraformer.transcribe(str(CHINESE_SHORT_WAV))
        text = clean_spaces(text)
        print(f"Short transcription: {text}")

        # At least some of the early keywords should appear
        assert any(kw in text for kw in ["今天", "天气", "不错"]), \
            f"Expected at least one keyword in: {text}"

    def test_transcribe_silence(self, paraformer):
        """Silence produces no meaningful Chinese content.

        Note: ASR models may hallucinate short phrases on pure silence
        (e.g. "好的好的"), so we only check it doesn't produce long text.
        """
        text = paraformer.transcribe(str(SILENCE_WAV))
        text = clean_spaces(text)
        # Silence should not produce a sentence-length hallucination
        assert len(text) < 20, f"Expected short/empty for silence, got: {text}"
        # And none of the expected keywords from our real audio
        for kw in EXPECTED_KEYWORDS:
            assert kw not in text, f"Silence should not contain '{kw}', got: {text}"

    def test_warmup_does_not_crash(self, paraformer):
        """Warmup with a real WAV file should not raise."""
        paraformer.warmup(str(SILENCE_WAV))


class TestStreamingRecognition:
    """Test real streaming ASR with Paraformer-online model."""

    @pytest.fixture(scope="class")
    def streaming(self):
        """Load the real streaming model once for all streaming tests."""
        recognizer = StreamingRecognizer(
            model_dir=STREAMING_MODEL,
            quantize=True,
            chunk_size=[5, 10, 5],
        )
        recognizer.load()
        return recognizer

    def test_streaming_produces_text_incrementally(self, streaming):
        """Feed real audio in chunks → get incremental text output."""
        audio, sr = sf.read(str(CHINESE_SPEECH_WAV), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        streaming.reset()
        step = streaming.step_samples
        offset = 0
        partials: list[str] = []
        full_text = ""

        while offset < len(audio):
            remaining = len(audio) - offset
            is_final = remaining <= step
            chunk = audio[offset:offset + min(step, remaining)]

            text = streaming.feed_chunk(chunk, is_final=is_final)
            if text:
                full_text += text
                partials.append(text)

            offset += step

        full_text = clean_spaces(full_text)
        print(f"Streaming partials: {partials}")
        print(f"Streaming full text: {full_text}")

        # Should have received multiple partial results
        assert len(partials) >= 3, f"Expected ≥3 partials, got {len(partials)}: {partials}"

        # Final accumulated text should contain expected keywords
        for keyword in EXPECTED_KEYWORDS:
            assert keyword in full_text, f"Expected '{keyword}' in streaming result: {full_text}"

    def test_streaming_reset_between_utterances(self, streaming):
        """Reset clears state — second utterance gets fresh results."""
        audio, sr = sf.read(str(CHINESE_SHORT_WAV), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # First utterance
        streaming.reset()
        text1 = streaming.transcribe_array(audio)
        text1 = clean_spaces(text1)

        # Second utterance (same audio, should get same result)
        text2 = streaming.transcribe_array(audio)
        text2 = clean_spaces(text2)

        print(f"Utterance 1: {text1}")
        print(f"Utterance 2: {text2}")

        # Both should produce meaningful text
        assert len(text1) > 0, "First utterance produced no text"
        assert len(text2) > 0, "Second utterance produced no text"

    def test_streaming_silence(self, streaming):
        """Streaming on silence should produce little/no text."""
        silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds

        streaming.reset()
        text = streaming.transcribe_array(silence)
        text = clean_spaces(text)

        assert len(text) < 20, f"Expected short/empty for silence, got: {text}"


class TestFullPipeline:
    """Test the complete VoiceEngine pipeline with real ASR."""

    def test_engine_with_real_recognizer(self):
        """VoiceEngine with real Paraformer, mock recorder, mock paster."""
        recognizer = SpeechRecognizer(
            model_dir=SEACO_MODEL,
            quantize=True,
        )

        paster = MockPaster()
        recorder = MockRecorder()

        results: list[str] = []
        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=recorder,
            recognizer=recognizer,
            paster=paster,
            on_result=lambda t: results.append(t),
        )

        # Start engine (loads model) — use silence for warmup
        engine.warmup_wav = str(SILENCE_WAV)
        recognizer.load()
        recognizer.warmup(str(SILENCE_WAV))
        recorder.start()

        # Copy the test WAV to the engine's temp audio path
        # so the offline transcription picks it up
        import shutil
        shutil.copy(str(CHINESE_SPEECH_WAV), engine._audio_path)

        # Simulate recording cycle
        engine.start_recording()
        engine.stop_recording()

        # Wait for offline transcription thread
        import time
        time.sleep(10)  # model inference can take a few seconds

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        text = results[0]
        print(f"Engine pipeline result: {text}")

        for keyword in EXPECTED_KEYWORDS:
            assert keyword in text, f"Expected '{keyword}' in: {text}"

        engine.shutdown()


class TestAudioProcessing:
    """Test real audio file loading and resampling."""

    def test_load_and_resample(self):
        """Load a real WAV file and resample to 16kHz mono."""
        from voice_input_method.audio import resample_to_16k_mono

        audio, sr = sf.read(str(CHINESE_SPEECH_WAV), dtype="float32")
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        resampled = resample_to_16k_mono(audio, orig_sr=sr, channels=audio.shape[1])
        assert resampled.ndim == 1
        assert resampled.dtype == np.float32
        # Should be approximately 4 seconds at 16kHz
        assert 60000 < len(resampled) < 80000, f"Unexpected length: {len(resampled)}"

    def test_fixture_files_exist(self):
        """Verify all test fixture audio files are present."""
        assert CHINESE_SPEECH_WAV.exists(), f"Missing: {CHINESE_SPEECH_WAV}"
        assert CHINESE_SHORT_WAV.exists(), f"Missing: {CHINESE_SHORT_WAV}"
        assert SILENCE_WAV.exists(), f"Missing: {SILENCE_WAV}"
