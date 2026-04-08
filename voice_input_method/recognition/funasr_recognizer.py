"""FunASR backend — SeacoParaformer (offline) + Paraformer-online (streaming).

This is the original recognizer, extracted from the monolithic recognition.py.
Uses funasr_onnx for ONNX Runtime inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np


def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a ModelScope model ID to a local cache path if available."""
    if os.path.isdir(model_dir):
        return model_dir
    cache_path = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_dir
    if cache_path.is_dir():
        return str(cache_path)
    return model_dir


class FunASRRecognizer:
    """Offline (batch) speech recognizer via funasr_onnx SeacoParaformer."""

    def __init__(self, model_dir: str, quantize: bool = True):
        self.model_dir = model_dir
        self.quantize = quantize
        self.model = None

    def load(self) -> None:
        from funasr_onnx import SeacoParaformer
        model_dir = _resolve_model_dir(self.model_dir)
        self.model = SeacoParaformer(model_dir, batch_size=1, quantize=self.quantize)

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        if self.model is None or not warmup_wav:
            return
        self.model([warmup_wav], hotwords=hotwords)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        if self.model is None:
            return ""
        result = self.model([wav_path], hotwords=hotwords)
        if result and "preds" in result[0]:
            preds = result[0]["preds"]
            if isinstance(preds, tuple):
                return preds[0]
            return preds
        return ""


class FunASRStreamingRecognizer:
    """Real-time streaming speech recognizer using Paraformer-online via funasr_onnx."""

    def __init__(self, model_dir: str, quantize: bool = True,
                 chunk_size: list[int] | None = None):
        self.model_dir = model_dir
        self.quantize = quantize
        self.chunk_size = chunk_size or [5, 10, 5]
        self.model = None
        self._cache: dict = {}

    @property
    def step_samples(self) -> int:
        """Number of 16kHz samples per chunk (body frames * 960)."""
        return self.chunk_size[1] * 960

    def load(self) -> None:
        from funasr_onnx.paraformer_online_bin import Paraformer
        self.model = Paraformer(
            _resolve_model_dir(self.model_dir),
            batch_size=1,
            quantize=self.quantize,
            chunk_size=self.chunk_size,
        )

    def reset(self) -> None:
        self._cache = {}

    def feed_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        if self.model is None:
            return ""
        param_dict = {"cache": self._cache, "is_final": is_final}
        result = self.model(audio_in=audio_chunk, param_dict=param_dict)
        if result and len(result) > 0 and "preds" in result[0]:
            preds = result[0]["preds"]
            if isinstance(preds, (list, tuple)):
                return str(preds[0]) if preds else ""
            return str(preds)
        return ""

    def transcribe_array(self, audio: np.ndarray,
                         on_partial: Callable[[str], None] | None = None) -> str:
        """Transcribe a complete audio array in streaming chunks."""
        self.reset()
        full_text = ""
        step = self.step_samples
        offset = 0

        while offset < len(audio):
            remaining = len(audio) - offset
            is_final = remaining <= step
            chunk = audio[offset:offset + min(step, remaining)]
            text = self.feed_chunk(chunk, is_final=is_final)
            if text:
                full_text += text
                if on_partial:
                    on_partial(full_text)
            offset += step

        return full_text
