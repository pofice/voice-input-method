"""Speech recognition module wrapping FunASR models."""

from pathlib import Path


class SpeechRecognizer:
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
