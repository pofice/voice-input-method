"""Tests for VAD segmenter — pure logic tests with mocked sherpa-onnx."""

import time
from unittest.mock import patch

import numpy as np

from voice_input_method.vad import VADSegmenter, _find_vad_model


class TestFindVadModel:
    def test_finds_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vad_file = tmp_path / "silero_vad.onnx"
        vad_file.write_text("fake")
        result = _find_vad_model()
        assert result is not None
        assert "silero_vad.onnx" in result

    def test_finds_in_models_subdir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        vad_file = models_dir / "silero_vad.onnx"
        vad_file.write_text("fake")
        result = _find_vad_model()
        assert result is not None

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _find_vad_model()
        assert result is None


class TestVADSegmenterShortAudio:
    """Short audio bypasses VAD entirely."""

    def test_short_audio_passthrough(self):
        vad = VADSegmenter(max_speech_duration=15.0)
        # 10 seconds of audio at 16kHz — shorter than max
        audio = np.zeros(160000, dtype=np.float32)
        segments = vad.segment_audio(audio, sample_rate=16000)
        assert len(segments) == 1
        assert len(segments[0]) == 160000

    def test_short_audio_resampled(self):
        vad = VADSegmenter(max_speech_duration=15.0)
        # 10 seconds at 44100Hz — will be resampled to 16kHz
        audio = np.zeros(441000, dtype=np.float32)
        segments = vad.segment_audio(audio, sample_rate=44100)
        assert len(segments) == 1
        # Should be approximately 160000 samples at 16kHz
        assert abs(len(segments[0]) - 160000) < 100


class TestVADSegmenterAvailability:
    def test_not_available_without_model(self):
        vad = VADSegmenter(vad_model_path=None)
        vad._model_path = None
        assert vad.available is False

    def test_not_available_without_sherpa(self):
        vad = VADSegmenter(vad_model_path="/fake/silero_vad.onnx")
        with patch.dict("sys.modules", {"sherpa_onnx": None}):
            # When sherpa_onnx import fails
            assert vad.available is False

    def test_long_audio_no_vad_returns_whole(self):
        """Long audio without VAD model returns the whole audio."""
        vad = VADSegmenter(max_speech_duration=5.0)
        vad._model_path = None  # no VAD model
        # 30 seconds
        audio = np.zeros(480000, dtype=np.float32)
        segments = vad.segment_audio(audio, sample_rate=16000)
        assert len(segments) == 1


class TestVADSegmenterFileIO:
    def test_segment_file_short(self, tmp_path):
        """Short file returns original path."""
        import soundfile as sf
        wav_path = tmp_path / "short.wav"
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        sf.write(str(wav_path), audio, 16000)

        vad = VADSegmenter(max_speech_duration=15.0)
        paths = vad.segment_file(str(wav_path))
        assert len(paths) == 1
        assert paths[0] == str(wav_path)

    def test_cleanup_temp_files(self, tmp_path):
        """cleanup only removes _vad_ files."""
        normal = tmp_path / "normal.wav"
        normal.write_text("keep")
        vad_temp = tmp_path / "audio_vad_0.wav"
        vad_temp.write_text("delete")

        VADSegmenter.cleanup_temp_files([str(normal), str(vad_temp)])
        assert normal.exists()
        assert not vad_temp.exists()


class TestEngineWithVAD:
    """Test VoiceEngine integration with VAD segmenter."""

    def test_engine_vad_segments_long_audio(self, mock_recorder, mock_paster):
        """Engine with VAD calls transcribe per segment and concatenates."""
        from tests.mocks import MockRecognizer
        from voice_input_method.engine import EngineConfig, VoiceEngine

        transcribe_calls = []

        class TrackingRecognizer(MockRecognizer):
            def transcribe(self, wav_path, hotwords=""):
                transcribe_calls.append(wav_path)
                return f"段{len(transcribe_calls)}"

        class FakeVAD:
            def segment_file(self, wav_path):
                return [wav_path + "_seg0", wav_path + "_seg1"]

            @staticmethod
            def cleanup_temp_files(paths):
                pass

        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=TrackingRecognizer(),
            paster=mock_paster,
            vad_segmenter=FakeVAD(),
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.5)
        assert len(transcribe_calls) == 2
        assert mock_paster.pasted == ["段1段2"]

    def test_engine_no_vad_single_transcribe(self, mock_recorder, mock_paster):
        """Engine without VAD calls transcribe once."""
        from tests.mocks import MockRecognizer
        from voice_input_method.engine import EngineConfig, VoiceEngine

        engine = VoiceEngine(
            config=EngineConfig(streaming=False),
            recorder=mock_recorder,
            recognizer=MockRecognizer(text="正常结果"),
            paster=mock_paster,
            vad_segmenter=None,
        )
        engine.start()
        engine.start_recording()
        engine.stop_recording()
        time.sleep(0.3)
        assert mock_paster.pasted == ["正常结果"]
