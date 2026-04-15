"""SenseVoice + LM rescoring backend.

Runs the SenseVoice CTC model directly via onnxruntime and decodes with
pyctcdecode + KenLM language model for improved accuracy.

Requires: pip install pyctcdecode kenlm onnxruntime kaldi-native-fbank soundfile

Architecture:
  audio → fbank → LFR → CMVN → ONNX inference → CTC logits
                                                       ↓
                                            pyctcdecode beam search + KenLM
                                                       ↓
                                                  final text
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf


class SenseVoiceLMRecognizer:
    """Offline speech recognizer using SenseVoice CTC + KenLM rescoring."""

    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        lm_path: str = "",
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 4,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.0,
        beam_width: int = 20,
    ):
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._lm_path = lm_path
        self._language = language
        self._use_itn = use_itn
        self._num_threads = num_threads
        self._lm_alpha = lm_alpha
        self._lm_beta = lm_beta
        self._beam_width = beam_width

        self._session = None
        self._decoder = None
        self._vocab: list[str] = []
        self._neg_mean: np.ndarray | None = None
        self._inv_stddev: np.ndarray | None = None
        self._lfr_window_size: int = 7
        self._lfr_window_shift: int = 6
        self._lang_id: int = 3  # zh default
        self._textnorm_id: int = 14  # with_itn default

    def load(self) -> None:
        import onnxruntime as ort
        from pyctcdecode import build_ctcdecoder

        # Load ONNX model
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = self._num_threads
        self._session = ort.InferenceSession(
            self._model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        # Extract metadata
        meta = self._session.get_modelmeta().custom_metadata_map
        self._lfr_window_size = int(meta.get("lfr_window_size", "7"))
        self._lfr_window_shift = int(meta.get("lfr_window_shift", "6"))
        self._neg_mean = np.array(
            [float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32
        )
        self._inv_stddev = np.array(
            [float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32
        )

        # Language and text norm IDs
        lang_key = f"lang_{self._language}"
        self._lang_id = int(meta.get(lang_key, "3"))
        self._textnorm_id = int(
            meta.get("with_itn" if self._use_itn else "without_itn", "14")
        )

        # Load vocabulary
        self._vocab = self._load_vocab(self._tokens_path)

        # Build CTC decoder with optional LM
        lm_path = self._lm_path if self._lm_path and Path(self._lm_path).exists() else None
        # pyctcdecode expects labels[0] to be the CTC blank token as ""
        labels = [""] + self._vocab[1:]  # Replace <unk> with "" for blank

        # Extract unigram frequencies from ARPA file if available
        unigrams = None
        if lm_path and lm_path.endswith(".arpa"):
            unigrams = self._extract_unigrams_from_arpa(lm_path)

        self._decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=lm_path,
            alpha=self._lm_alpha,
            beta=self._lm_beta,
            unigrams=unigrams,
        )

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        if self._session is None or not warmup_wav:
            return
        self.transcribe(warmup_wav, hotwords)

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        if self._session is None:
            return ""

        # Load audio
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = self._resample(audio, sr, 16000)
            sr = 16000

        # Scale to int16 range (SenseVoice expects [-32768, 32767])
        audio_scaled = audio * 32768.0

        # Extract features: fbank → LFR → CMVN
        features = self._extract_features(audio_scaled, sr)

        # Run ONNX inference
        logits = self._infer(features)

        # Strip first 4 metadata timesteps (language, event, emotion, text_norm)
        logits = logits[:, 4:, :]

        # Decode with pyctcdecode (beam search + LM)
        log_probs = self._log_softmax(logits[0])
        text = self._decoder.decode(
            log_probs,
            beam_width=self._beam_width,
        )

        # Strip SenseVoice metadata tags
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        # Clean up spaces between CJK characters
        text = self._clean_cjk_spaces(text)
        return text

    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract fbank features, apply LFR and CMVN.

        Returns: np.ndarray of shape [1, T', 560] (batch=1)
        """
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.samp_freq = sr
        opts.mel_opts.num_bins = 80

        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(sr, audio.tolist())
        fbank.input_finished()

        num_frames = fbank.num_frames_ready
        if num_frames == 0:
            return np.zeros((1, 1, 560), dtype=np.float32)

        # Collect fbank features [T, 80]
        features = np.stack(
            [np.array(fbank.get_frame(i), dtype=np.float32) for i in range(num_frames)]
        )

        # Apply LFR (Low Frame Rate): stack consecutive frames
        features = self._apply_lfr(features)

        # Apply CMVN
        features = self._apply_cmvn(features)

        # Add batch dimension [1, T', 560]
        return features[np.newaxis, :, :]

    def _apply_lfr(self, features: np.ndarray) -> np.ndarray:
        """Low Frame Rate: stack lfr_window_size frames with lfr_window_shift stride.

        Input: [T, 80]
        Output: [T', 560] where T' = ceil((T - window_size) / shift) + 1
        """
        t, d = features.shape
        win = self._lfr_window_size
        shift = self._lfr_window_shift

        # Pad features so we don't lose frames at the end
        pad_len = (win - 1) // 2
        if pad_len > 0:
            features = np.pad(
                features, ((0, pad_len), (0, 0)), mode="edge"
            )

        t_padded = features.shape[0]
        indices = list(range(0, t_padded - win + 1, shift))
        if not indices:
            indices = [0]

        lfr_features = []
        for i in indices:
            stacked = features[i : i + win].reshape(-1)  # [win * 80]
            lfr_features.append(stacked)

        return np.stack(lfr_features).astype(np.float32)

    def _apply_cmvn(self, features: np.ndarray) -> np.ndarray:
        """Apply CMVN normalization using model metadata."""
        if self._neg_mean is not None and self._inv_stddev is not None:
            features = (features + self._neg_mean) * self._inv_stddev
        return features

    def _infer(self, features: np.ndarray) -> np.ndarray:
        """Run ONNX model inference.

        Returns: logits [1, T, vocab_size]
        """
        t = features.shape[1]
        x_length = np.array([t], dtype=np.int32)
        language = np.array([self._lang_id], dtype=np.int32)
        text_norm = np.array([self._textnorm_id], dtype=np.int32)

        outputs = self._session.run(
            None,
            {
                "x": features,
                "x_length": x_length,
                "language": language,
                "text_norm": text_norm,
            },
        )
        return outputs[0]  # logits [N, T, vocab_size]

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable log softmax along last axis."""
        max_val = logits.max(axis=-1, keepdims=True)
        shifted = logits - max_val
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return shifted - log_sum_exp

    @staticmethod
    def _load_vocab(tokens_path: str) -> list[str]:
        """Load vocabulary from tokens.txt (format: token_text token_id)."""
        vocab = []
        with open(tokens_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vocab.append(parts[0])
        return vocab

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        ratio = target_sr / orig_sr
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _extract_unigrams_from_arpa(arpa_path: str) -> list[str]:
        """Extract unigram words from an ARPA language model file."""
        unigrams = []
        in_unigram_section = False
        with open(arpa_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "\\1-grams:":
                    in_unigram_section = True
                    continue
                if line.startswith("\\") and line.endswith(":"):
                    in_unigram_section = False
                    continue
                if in_unigram_section and line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        word = parts[1]
                        if word not in ("<unk>", "<s>", "</s>"):
                            unigrams.append(word)
        return unigrams

    @staticmethod
    def _clean_cjk_spaces(text: str) -> str:
        """Remove spaces between CJK characters inserted by CTC decoding."""
        import re as _re
        # Remove spaces between CJK chars
        cjk_range = r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
        text = _re.sub(
            f"({cjk_range})\\s+({cjk_range})", r"\1\2", text
        )
        # Repeat to catch consecutive pairs
        text = _re.sub(
            f"({cjk_range})\\s+({cjk_range})", r"\1\2", text
        )
        return text.strip()
