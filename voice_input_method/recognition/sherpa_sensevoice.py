"""SenseVoice-Small backend via sherpa-onnx.

Requires: pip install sherpa-onnx
Model: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (int8: ~229MB)

SenseVoice has built-in ITN (inverse text normalization) and punctuation,
so no external cn2an or punctuation module is needed.
"""

from __future__ import annotations

import re

import soundfile as sf


class SherpaSenseVoiceRecognizer:
    """Offline speech recognizer using SenseVoice-Small via sherpa-onnx."""

    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 4,
        provider: str = "cpu",
    ):
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._language = language
        self._use_itn = use_itn
        self._num_threads = num_threads
        self._provider = provider
        self._recognizer = None

    def load(self) -> None:
        import sherpa_onnx
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self._model_path,
            tokens=self._tokens_path,
            language=self._language,
            use_itn=self._use_itn,
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
        # sherpa-onnx accepts a numpy float32 array directly; avoid
        # the costly .tolist() conversion for long audio.
        s.accept_waveform(sr, data)
        self._recognizer.decode_stream(s)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        if self._recognizer is None:
            return ""
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        s = self._recognizer.create_stream()
        # sherpa-onnx accepts a numpy float32 array directly; avoid
        # the costly .tolist() conversion for long audio.
        s.accept_waveform(sr, data)
        self._recognizer.decode_stream(s)
        text = s.result.text
        # Strip SenseVoice metadata tags like <|zh|><|NEUTRAL|><|Speech|>
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        return text
