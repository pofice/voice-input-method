"""Fun-ASR-Nano backend via sherpa-onnx.

Requires: pip install sherpa-onnx
Model: sherpa-onnx-funasr-nano-int8-2025-12-30 (int8: ~800MB)

Fun-ASR-Nano is an LLM-based ASR (Qwen3-0.6B) with built-in ITN,
punctuation, and hotword support. Trained on tens of millions of hours
of real speech, excels at dialects, accents, and noisy environments.
"""

from __future__ import annotations

import soundfile as sf


class SherpaNanoRecognizer:
    """Offline speech recognizer using Fun-ASR-Nano via sherpa-onnx."""

    def __init__(
        self,
        encoder_adaptor_path: str,
        llm_path: str,
        embedding_path: str,
        tokenizer_path: str,
        language: str = "zh",
        num_threads: int = 4,
        provider: str = "cpu",
    ):
        self._encoder_adaptor_path = encoder_adaptor_path
        self._llm_path = llm_path
        self._embedding_path = embedding_path
        self._tokenizer_path = tokenizer_path
        self._language = language
        self._num_threads = num_threads
        self._provider = provider
        self._recognizer = None

    def load(self) -> None:
        import sherpa_onnx
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=self._encoder_adaptor_path,
            llm=self._llm_path,
            embedding=self._embedding_path,
            tokenizer=self._tokenizer_path,
            language=self._language,
            num_threads=self._num_threads,
            provider=self._provider,
        )

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        if self._recognizer is None or not warmup_wav:
            return
        data, sr = sf.read(warmup_wav, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        s = self._recognizer.create_stream()
        s.accept_waveform(sr, data.tolist())
        self._recognizer.decode_stream(s)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        if self._recognizer is None:
            return ""
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        s = self._recognizer.create_stream()
        s.accept_waveform(sr, data.tolist())
        self._recognizer.decode_stream(s)
        return s.result.text.strip()
