"""Speech recognition module wrapping FunASR models (offline + streaming)."""

import numpy as np
from typing import Callable


class SpeechRecognizer:
    """Offline (batch) speech recognizer."""

    def __init__(self, model_type: str, model_dir: str, quantize: bool = True):
        self.model_type = model_type
        self.model_dir = model_dir
        self.quantize = quantize
        self.model = None

    def load(self):
        """Load the ASR model."""
        if self.model_type == "seaco_paraformer":
            from funasr_onnx import SeacoParaformer
            self.model = SeacoParaformer(self.model_dir, batch_size=1, quantize=self.quantize)
        else:
            from funasr_onnx import Paraformer
            self.model = Paraformer(self.model_dir, batch_size=1, quantize=self.quantize)

    def warmup(self, warmup_wav: str, hotwords: str = ""):
        """Run a warmup inference to avoid first-call latency."""
        if self.model is None:
            return
        kwargs = {"hotwords": hotwords} if hotwords and self.model_type == "seaco_paraformer" else {}
        self.model([warmup_wav], **kwargs)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        """Transcribe a WAV file and return the text."""
        if self.model is None:
            return ""
        kwargs = {"hotwords": hotwords} if hotwords and self.model_type == "seaco_paraformer" else {}
        result = self.model([wav_path], **kwargs)
        if result and "preds" in result[0]:
            preds = result[0]["preds"]
            if isinstance(preds, tuple):
                return preds[0]
            return preds
        return ""


class StreamingRecognizer:
    """Real-time streaming speech recognizer using Paraformer-online."""

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

    def load(self):
        """Load the streaming ASR model."""
        from funasr_onnx.paraformer_online_bin import Paraformer
        self.model = Paraformer(
            self.model_dir,
            batch_size=1,
            quantize=self.quantize,
            chunk_size=self.chunk_size,
        )

    def reset(self):
        """Reset streaming state for a new utterance."""
        self._cache = {}

    def feed_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        """Feed a chunk of 16kHz mono float32 audio. Returns recognized text for this chunk."""
        if self.model is None:
            return ""
        param_dict = {"cache": self._cache, "is_final": is_final}
        result = self.model(audio_in=audio_chunk, param_dict=param_dict)
        if result and len(result) > 0 and "preds" in result[0]:
            preds = result[0]["preds"]
            if isinstance(preds, tuple):
                return preds[0]
            return preds
        return ""

    def transcribe_array(self, audio: np.ndarray,
                         on_partial: Callable[[str], None] | None = None) -> str:
        """Transcribe a complete audio array in streaming chunks. Returns full text."""
        self.reset()
        full_text = ""
        step = self.step_samples
        offset = 0

        while offset < len(audio):
            remaining = len(audio) - offset
            if remaining <= step:
                chunk = audio[offset:offset + remaining]
                is_final = True
            else:
                chunk = audio[offset:offset + step]
                is_final = False

            text = self.feed_chunk(chunk, is_final=is_final)
            if text:
                full_text += text
                if on_partial:
                    on_partial(full_text)

            offset += step

        return full_text
