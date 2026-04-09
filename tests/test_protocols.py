"""Tests that mock objects satisfy Protocol contracts at runtime."""

from tests.mocks import (
    MockHotwordProvider,
    MockIndicator,
    MockPaster,
    MockRecognizer,
    MockRecorder,
    MockStreamingRecognizer,
)
from voice_input_method.protocols import (
    AudioSource,
    HotwordProvider,
    Recognizer,
    RecordingIndicator,
    StreamingRecognizerProto,
    TextPaster,
)


def test_mock_recorder_satisfies_audio_source():
    assert isinstance(MockRecorder(), AudioSource)


def test_mock_recognizer_satisfies_recognizer():
    assert isinstance(MockRecognizer(), Recognizer)


def test_mock_streaming_satisfies_streaming_proto():
    assert isinstance(MockStreamingRecognizer(), StreamingRecognizerProto)


def test_mock_paster_satisfies_text_paster():
    assert isinstance(MockPaster(), TextPaster)


def test_mock_hotword_satisfies_hotword_provider():
    assert isinstance(MockHotwordProvider(), HotwordProvider)


def test_mock_indicator_satisfies_recording_indicator():
    assert isinstance(MockIndicator(), RecordingIndicator)
