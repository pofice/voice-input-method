"""VAD (Voice Activity Detection) segmenter for long audio.

Uses sherpa-onnx Silero VAD to split long audio into speech segments,
so each segment fits within the KV cache limits of LLM-based ASR models
(sherpa-nano, qwen3-asr). Also prevents memory issues on encoder-based
models (funasr, sensevoice) with very long recordings.

Short audio (<= max_speech_duration) bypasses VAD entirely — zero overhead.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import resample_to_16k_mono


def _find_vad_model() -> str | None:
    """Search common locations for silero_vad.onnx."""
    candidates = [
        Path.cwd() / "silero_vad.onnx",
        Path.cwd() / "models" / "silero_vad.onnx",
        Path.home() / ".cache" / "sherpa-onnx" / "silero_vad.onnx",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


class VADSegmenter:
    """Split long audio into speech segments using Silero VAD.

    Args:
        vad_model_path: Path to silero_vad.onnx. Auto-detected if None.
        max_speech_duration: Maximum seconds per segment (default 15s).
        min_silence_duration: Minimum silence gap to split (default 0.3s).
        threshold: VAD speech probability threshold (default 0.5).
        sample_rate: Expected sample rate (always 16000 for VAD).
    """

    def __init__(
        self,
        vad_model_path: str | None = None,
        max_speech_duration: float = 15.0,
        min_silence_duration: float = 0.3,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        self._model_path = vad_model_path or _find_vad_model()
        self._max_speech_duration = max_speech_duration
        self._min_silence_duration = min_silence_duration
        self._threshold = threshold
        self._sample_rate = sample_rate

    @property
    def available(self) -> bool:
        """True if the VAD model file exists and sherpa-onnx is installed."""
        if not self._model_path:
            return False
        try:
            import sherpa_onnx  # noqa: F401
            return True
        except ImportError:
            return False

    def segment_audio(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Split audio into speech segments.

        Args:
            audio: Audio data (any sample rate, mono or stereo).
            sample_rate: Sample rate of the input audio.

        Returns:
            List of 16kHz mono float32 audio segments. If VAD is unavailable
            or audio is short enough, returns [resampled_original].
        """
        # Resample to 16kHz mono
        channels = 2 if audio.ndim == 2 and audio.shape[1] > 1 else 1
        audio_16k = resample_to_16k_mono(audio, sample_rate, channels)

        # Short audio: skip VAD
        duration = len(audio_16k) / self._sample_rate
        if duration <= self._max_speech_duration:
            return [audio_16k]

        # No VAD model: return whole audio (will degrade on LLM backends)
        if not self.available:
            return [audio_16k]

        return self._run_vad(audio_16k)

    def segment_file(self, wav_path: str) -> list[str]:
        """Split a WAV file into segment files.

        Returns list of file paths. If only one segment (short audio),
        returns [original_path].
        """
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            channels = data.shape[1]
        else:
            channels = 1
        audio_16k = resample_to_16k_mono(data, sr, channels)

        duration = len(audio_16k) / self._sample_rate
        if duration <= self._max_speech_duration:
            return [wav_path]

        if not self.available:
            return [wav_path]

        segments = self._run_vad(audio_16k)
        if len(segments) <= 1:
            return [wav_path]

        # Write segments to temp files
        paths = []
        for i, seg in enumerate(segments):
            tmp = tempfile.NamedTemporaryFile(
                suffix=f"_vad_{i}.wav", delete=False
            )
            tmp.close()
            sf.write(tmp.name, seg, self._sample_rate)
            paths.append(tmp.name)
        return paths

    def _run_vad(self, audio_16k: np.ndarray) -> list[np.ndarray]:
        """Run Silero VAD on 16kHz mono audio."""
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = self._model_path
        config.silero_vad.threshold = self._threshold
        config.silero_vad.min_silence_duration = self._min_silence_duration
        config.silero_vad.max_speech_duration = self._max_speech_duration
        config.sample_rate = self._sample_rate

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=600)

        # Feed audio in 512-sample windows (required by Silero VAD)
        window_size = 512
        offset = 0
        while offset + window_size <= len(audio_16k):
            vad.accept_waveform(audio_16k[offset:offset + window_size])
            offset += window_size
        # Feed remaining samples
        if offset < len(audio_16k):
            remaining = audio_16k[offset:]
            padded = np.zeros(window_size, dtype=np.float32)
            padded[:len(remaining)] = remaining
            vad.accept_waveform(padded)

        vad.flush()

        # Extract segments
        segments = []
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            if len(samples) > 0:
                segments.append(samples)
            vad.pop()

        if not segments:
            return [audio_16k]

        return segments

    @staticmethod
    def cleanup_temp_files(paths: list[str]) -> None:
        """Remove temporary segment files created by segment_file()."""
        for p in paths:
            if "_vad_" in p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
