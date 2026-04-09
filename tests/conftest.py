"""Shared fixtures for voice-input-method tests."""

import pytest

from tests.mocks import MockPaster, MockRecognizer, MockRecorder
from voice_input_method.engine import EngineConfig, VoiceEngine


@pytest.fixture
def engine_config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture
def mock_recorder() -> MockRecorder:
    return MockRecorder()


@pytest.fixture
def mock_recognizer() -> MockRecognizer:
    return MockRecognizer()


@pytest.fixture
def mock_paster() -> MockPaster:
    return MockPaster()


@pytest.fixture
def engine(engine_config, mock_recorder, mock_recognizer, mock_paster) -> VoiceEngine:
    """A fully wired VoiceEngine with all mocks."""
    return VoiceEngine(
        config=engine_config,
        recorder=mock_recorder,
        recognizer=mock_recognizer,
        paster=mock_paster,
    )
