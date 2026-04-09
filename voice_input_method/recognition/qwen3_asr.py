"""Qwen3-ASR backend via sherpa-onnx.

Requires: pip install sherpa-onnx
Model: sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 (int8: ~500MB)

Qwen3-ASR is an LLM-based ASR supporting 52 languages and dialects
(30 languages + 22 Chinese dialects). Built on Qwen3-0.6B with built-in
ITN, punctuation, and hotword support.
"""

from __future__ import annotations

import soundfile as sf


class Qwen3ASRRecognizer:
    """Offline speech recognizer using Qwen3-ASR via sherpa-onnx."""

    def __init__(
        self,
        conv_frontend_path: str,
        encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        num_threads: int = 4,
        provider: str = "cpu",
        hotwords: str = "",
        max_total_len: int = 512,
        max_new_tokens: int = 128,
    ):
        self._conv_frontend_path = conv_frontend_path
        self._encoder_path = encoder_path
        self._decoder_path = decoder_path
        self._tokenizer_path = tokenizer_path
        self._num_threads = num_threads
        self._provider = provider
        self._hotwords = hotwords
        self._max_total_len = max_total_len
        self._max_new_tokens = max_new_tokens
        self._recognizer = None

    def load(self) -> None:
        import sherpa_onnx
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=self._conv_frontend_path,
            encoder=self._encoder_path,
            decoder=self._decoder_path,
            tokenizer=self._tokenizer_path,
            num_threads=self._num_threads,
            provider=self._provider,
            hotwords=self._hotwords,
            max_total_len=self._max_total_len,
            max_new_tokens=self._max_new_tokens,
        )

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        if self._recognizer is None or not warmup_wav:
            return
        data, sr = sf.read(warmup_wav, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        s = self._recognizer.create_stream()
        s.accept_waveform(sr, data)
        self._recognizer.decode_stream(s)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        if self._recognizer is None:
            return ""
        if hotwords and hotwords != self._hotwords:
            import warnings
            warnings.warn(
                "qwen3-asr hotwords are baked at load() time. "
                "Per-call hotwords are ignored — reload the recognizer "
                "with the new hotwords to take effect.",
                RuntimeWarning,
                stacklevel=2,
            )
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        s = self._recognizer.create_stream()
        s.accept_waveform(sr, data)
        self._recognizer.decode_stream(s)
        return s.result.text.strip()
