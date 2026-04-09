"""Tests for audio module — resample function is pure numpy, testable anywhere."""

from unittest.mock import MagicMock

import numpy as np

from voice_input_method.audio import AudioRecorder, resample_to_16k_mono


class TestResampleTo16kMono:
    def test_mono_passthrough_at_16k(self):
        data = np.random.randn(16000).astype(np.float32)
        result = resample_to_16k_mono(data, orig_sr=16000, channels=1)
        assert result.shape == (16000,)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, data)

    def test_stereo_to_mono(self):
        data = np.random.randn(16000, 2).astype(np.float32)
        result = resample_to_16k_mono(data, orig_sr=16000, channels=2)
        assert result.ndim == 1
        assert result.shape == (16000,)

    def test_resample_44100_to_16k(self):
        data = np.random.randn(44100).astype(np.float32)
        result = resample_to_16k_mono(data, orig_sr=44100, channels=1)
        # Should be approximately 16000 samples for 1 second of audio
        assert abs(result.shape[0] - 16000) < 10
        assert result.dtype == np.float32

    def test_resample_48k_to_16k(self):
        data = np.random.randn(48000).astype(np.float32)
        result = resample_to_16k_mono(data, orig_sr=48000, channels=1)
        assert abs(result.shape[0] - 16000) < 10

    def test_empty_array(self):
        data = np.array([], dtype=np.float32)
        result = resample_to_16k_mono(data, orig_sr=16000, channels=1)
        assert result.shape == (0,)

    def test_single_channel_2d(self):
        data = np.random.randn(16000, 1).astype(np.float32)
        result = resample_to_16k_mono(data, orig_sr=16000, channels=1)
        assert result.ndim == 1
        assert result.shape == (16000,)


class TestAudioRecorderWithMocks:
    """Test AudioRecorder methods with mocked sounddevice."""

    def _make_recorder(self, mock_sd):
        """Create a recorder with mocked sounddevice."""
        rec = AudioRecorder(sample_rate=44100, channels=2)
        rec._sd = mock_sd
        return rec

    def test_switch_device(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = {
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        }
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        rec = self._make_recorder(mock_sd)
        rec.stream = MagicMock()  # simulate existing stream

        rec.switch_device(3)
        assert rec.sample_rate == 48000
        assert rec.channels == 2
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_switch_device_none_uses_default(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = {
            "max_input_channels": 1,
            "default_samplerate": 44100.0,
        }
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        rec = self._make_recorder(mock_sd)
        rec.stream = MagicMock()

        rec.switch_device(None)
        mock_stream.start.assert_called_once()

    def test_switch_device_fallback_on_error(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError("device not found")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        rec = self._make_recorder(mock_sd)
        rec.stream = MagicMock()

        # Should not raise, falls back to default
        rec.switch_device(99)

    def test_refresh_and_list_devices(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Mic 1", "max_input_channels": 1, "default_samplerate": 48000.0},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100.0},
            {"name": "USB Mic", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        rec = self._make_recorder(mock_sd)
        rec.stream = MagicMock()

        devices = rec.refresh_and_list_devices()
        assert len(devices) == 2  # only input devices
        assert devices[0]["name"] == "Mic 1"
        assert devices[1]["name"] == "USB Mic"
        mock_sd._terminate.assert_called_once()
        mock_sd._initialize.assert_called_once()
        # Stream should be restored
        mock_stream.start.assert_called_once()

    def test_refresh_and_list_devices_no_stream(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []

        rec = self._make_recorder(mock_sd)
        rec.stream = None

        devices = rec.refresh_and_list_devices()
        assert devices == []

    def test_start_stop_recording(self):
        rec = AudioRecorder()
        rec.start_recording()
        assert rec.is_recording is True
        assert rec.buffer == []

        # Simulate some audio data
        rec.buffer.append(np.zeros((1600,), dtype=np.float32))
        path = rec.stop_recording("/tmp/test_out.wav")
        assert rec.is_recording is False
        assert path == "/tmp/test_out.wav"

    def test_get_recording_16k_mono_empty(self):
        rec = AudioRecorder()
        result = rec.get_recording_16k_mono()
        assert result.shape == (0,)

    def test_audio_callback_not_recording(self):
        rec = AudioRecorder()
        rec.is_recording = False
        rec._audio_callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
        assert len(rec.buffer) == 0

    def test_audio_callback_recording(self):
        rec = AudioRecorder()
        rec.is_recording = True
        rec._audio_callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
        assert len(rec.buffer) == 1

    def test_audio_callback_streaming(self):
        """Streaming callback fires on_chunk when enough samples accumulate."""
        chunks_received = []
        rec = AudioRecorder(
            sample_rate=16000,
            channels=1,
            on_chunk=lambda c: chunks_received.append(c.copy()),
            chunk_samples=1600,  # 100ms at 16kHz
        )
        rec.is_recording = True
        # Feed 3200 samples (2 chunks worth)
        rec._audio_callback(np.zeros((3200, 1), dtype=np.float32), 3200, None, None)
        assert len(chunks_received) == 2
        assert len(chunks_received[0]) == 1600

    def test_flush_streaming_buffer(self):
        """flush_streaming_buffer sends remaining samples."""
        chunks_received = []
        rec = AudioRecorder(
            sample_rate=16000,
            channels=1,
            on_chunk=lambda c: chunks_received.append(c.copy()),
            chunk_samples=1600,
        )
        rec.is_recording = True
        # Feed less than one chunk
        rec._audio_callback(np.zeros((800, 1), dtype=np.float32), 800, None, None)
        assert len(chunks_received) == 0
        rec.flush_streaming_buffer()
        assert len(chunks_received) == 1
        assert len(chunks_received[0]) == 800

    def test_flush_empty_buffer(self):
        """Flushing empty streaming buffer does nothing."""
        chunks_received = []
        rec = AudioRecorder(
            sample_rate=16000,
            channels=1,
            on_chunk=lambda c: chunks_received.append(c),
            chunk_samples=1600,
        )
        rec.flush_streaming_buffer()
        assert chunks_received == []

    def test_get_recording_16k_mono_with_data(self):
        rec = AudioRecorder(sample_rate=16000, channels=1)
        rec.buffer = [np.ones(16000, dtype=np.float32)]
        result = rec.get_recording_16k_mono()
        assert result.shape == (16000,)
        assert result.dtype == np.float32

    def test_concat_buffer_multiple(self):
        rec = AudioRecorder()
        rec.buffer = [
            np.ones(100, dtype=np.float32),
            np.ones(200, dtype=np.float32),
        ]
        result = rec._concat_buffer()
        assert result.shape == (300,)

    def test_detect_device_default(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = {
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        }
        rec = AudioRecorder(sample_rate=44100, channels=2)
        rec._sd = mock_sd
        sr, ch = rec._detect_device()
        assert sr == 48000
        assert ch == 2

    def test_detect_device_fallback_to_enumerate(self):
        mock_sd = MagicMock()
        # First call (kind="input") raises
        mock_sd.query_devices.side_effect = [
            RuntimeError("no default"),
            [
                {"max_input_channels": 0, "default_samplerate": 44100.0},
                {"max_input_channels": 1, "default_samplerate": 48000.0},
            ],
        ]
        rec = AudioRecorder()
        rec._sd = mock_sd
        sr, ch = rec._detect_device()
        assert sr == 48000
        assert ch == 1

    def test_detect_device_all_fail(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError("total failure")
        rec = AudioRecorder(sample_rate=44100, channels=1)
        rec._sd = mock_sd
        sr, ch = rec._detect_device()
        assert sr == 44100
        assert ch == 1

    def test_open_stream_fallback(self):
        """_open_stream tries multiple params and falls back."""
        mock_sd = MagicMock()
        # First two attempts fail, third succeeds
        mock_stream = MagicMock()
        mock_stream.samplerate = 44100
        mock_stream.channels = 1
        mock_sd.InputStream.side_effect = [
            RuntimeError("bad params"),
            RuntimeError("bad params"),
            mock_stream,
        ]
        rec = AudioRecorder()
        rec._sd = mock_sd
        rec.sample_rate = 44100
        rec.channels = 2
        stream = rec._open_stream()
        assert stream is mock_stream

    def test_open_stream_all_fail(self):
        mock_sd = MagicMock()
        mock_sd.InputStream.side_effect = RuntimeError("all fail")
        rec = AudioRecorder()
        rec._sd = mock_sd
        rec.sample_rate = 44100
        rec.channels = 1
        stream = rec._open_stream()
        assert stream is None

    def test_stop_closes_stream(self):
        rec = AudioRecorder()
        mock_stream = MagicMock()
        rec.stream = mock_stream
        rec.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert rec.stream is None

    def test_stop_no_stream(self):
        rec = AudioRecorder()
        rec.stream = None
        rec.stop()  # Should not raise
