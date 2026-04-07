"""Tests for audio module — resample function is pure numpy, testable anywhere."""

import numpy as np
import pytest

from voice_input_method.audio import resample_to_16k_mono


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
