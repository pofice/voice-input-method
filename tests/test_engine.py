"""Tests for VoiceEngine — the core pipeline, fully mocked."""

import time
import threading

import numpy as np
import pytest

from voice_input_method.engine import VoiceEngine, EngineConfig
from tests.mocks import (
    MockRecorder,
    MockRecognizer,
    MockStreamingRecognizer,
    MockPaster,
    MockHotwordProvider,
)


class TestEngineLifecycle:
    def test_start_loads_models(self, engine, mock_recorder, mock_recognizer):
        engine.start()
        assert mock_recognizer.loaded
        assert mock_recognizer.warmed_up
        assert mock_recorder.started

    def test_shutdown_stops_recorder(self, engine, mock_recorder):
        engine.start()
        engine.shutdown()
        assert mock_recorder.stopped


class TestOfflineRecognition:
    def test_offline_pipeline(self, mock_recorder, mock_paster):
        """Offline mode: stop_recording → transcribe → paste."""
        recognizer = MockRecognizer(text="测试 结果")
        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
        )
        engine.start()
        engine.start_recording()
        assert mock_recorder.recording

        engine.stop_recording()
        # Offline transcription runs in a thread, wait briefly
        time.sleep(0.3)
        assert len(mock_paster.pasted) == 1
        assert mock_paster.pasted[0] == "测试结果"  # spaces cleaned

    def test_strips_trailing_punctuation_when_enabled(self, mock_recorder, mock_paster):
        recognizer = MockRecognizer(text="今天天气不错。")
        engine = VoiceEngine(
            config=EngineConfig(streaming=False, strip_trailing_punctuation=True),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert mock_paster.pasted == ["今天天气不错"]

    def test_applies_corrections_before_strip(self, mock_recorder, mock_paster):
        recognizer = MockRecognizer(text="用Cloud Code写代码。")
        hotword_provider = MockHotwordProvider(
            corrections={"Cloud Code": "Claude Code"}
        )
        engine = VoiceEngine(
            config=EngineConfig(streaming=False, strip_trailing_punctuation=True),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
            hotword_provider=hotword_provider,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        # correction applied first, then trailing punctuation stripped
        assert mock_paster.pasted == ["用Claude Code写代码"]

    def test_keeps_trailing_punctuation_when_disabled(self, mock_recorder, mock_paster):
        recognizer = MockRecognizer(text="今天天气不错。")
        engine = VoiceEngine(
            config=EngineConfig(streaming=False, strip_trailing_punctuation=False),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert mock_paster.pasted == ["今天天气不错。"]

    def test_empty_transcription_no_paste(self, mock_recorder, mock_paster):
        recognizer = MockRecognizer(text="")
        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert len(mock_paster.pasted) == 0


class TestStreamingRecognition:
    def test_streaming_pipeline(self, mock_recorder, mock_paster):
        """Pure streaming: chunks → accumulated text → paste."""
        streaming = MockStreamingRecognizer(chunks_text=["你", "好"])
        engine = VoiceEngine(
            config=EngineConfig(streaming=True, two_pass=False),
            recorder=mock_recorder,
            recognizer=MockRecognizer(),
            paster=mock_paster,
            streaming_recognizer=streaming,
        )
        engine.start()
        engine.start_recording()

        # Simulate audio chunks
        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))
        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))

        engine.stop_recording()
        assert len(mock_paster.pasted) == 1
        assert mock_paster.pasted[0] == "你好"

    def test_streaming_partial_callback(self, mock_recorder, mock_paster):
        """on_partial is called with intermediate text."""
        streaming = MockStreamingRecognizer(chunks_text=["你", "好", "世界"])
        partials: list[str] = []
        engine = VoiceEngine(
            config=EngineConfig(streaming=True, two_pass=False),
            recorder=mock_recorder,
            recognizer=MockRecognizer(),
            paster=mock_paster,
            streaming_recognizer=streaming,
            on_partial=lambda t: partials.append(t),
        )
        engine.start()
        engine.start_recording()

        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))
        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))
        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))

        assert len(partials) == 3
        assert partials[-1] == "你好世界"

    def test_two_pass_uses_offline(self, mock_recorder, mock_paster):
        """2pass mode: streaming for partials, offline for final result."""
        streaming = MockStreamingRecognizer(chunks_text=["你"])
        recognizer = MockRecognizer(text="你好 世界")
        engine = VoiceEngine(
            config=EngineConfig(streaming=True, two_pass=True),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
            streaming_recognizer=streaming,
        )
        engine.start()
        engine.start_recording()
        engine.on_audio_chunk(np.zeros(9600, dtype=np.float32))
        engine.stop_recording()

        time.sleep(0.3)
        # Final result comes from offline recognizer, not streaming
        assert len(mock_paster.pasted) == 1
        assert mock_paster.pasted[0] == "你好世界"


class TestTextProcessing:
    def test_on_result_callback(self, mock_recorder, mock_paster):
        results: list[str] = []
        recognizer = MockRecognizer(text="回调 测试")
        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=recognizer,
            paster=mock_paster,
            on_result=lambda t: results.append(t),
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert len(results) == 1
        assert results[0] == "回调测试"


class TestHotwords:
    def test_hotwords_passed_to_recognizer(self, mock_recorder, mock_paster):
        """Hotword provider's string is used during transcription."""
        hotwords = MockHotwordProvider("测试 热词")
        transcribed_hotwords: list[str] = []

        class SpyRecognizer(MockRecognizer):
            def transcribe(self, wav_path, hotwords=""):
                transcribed_hotwords.append(hotwords)
                return super().transcribe(wav_path, hotwords)

        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=SpyRecognizer(),
            paster=mock_paster,
            hotword_provider=hotwords,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert len(transcribed_hotwords) == 1
        assert transcribed_hotwords[0] == "测试 热词"


class TestConvertChinese:
    def test_convert_returns_none_without_converter(self, engine):
        result = engine.convert_chinese("你好")
        assert result is None

    def test_convert_empty_string(self, engine):
        engine.chinese_converter = type("FakeCC", (), {"convert": lambda s, t: t})()
        result = engine.convert_chinese("")
        assert result is None
