"""Configuration management."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Default model IDs (pre-exported ONNX, auto-download from ModelScope)
DEFAULT_OFFLINE_MODELS = {
    "seaco_paraformer": "pofice/speech_seaco_paraformer_large_onnx",
}
DEFAULT_STREAMING_MODEL = (
    "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
)


@dataclass
class Config:
    # Model
    model_type: str = "seaco_paraformer"  # currently only "seaco_paraformer"
    model_dir: str = ""
    quantize: bool = True

    # Audio
    sample_rate: int = 44100
    channels: int = 2

    # Recognizer backend: "funasr", "sherpa-sensevoice", "sensevoice-lm",
    # "sherpa-nano", "qwen3-asr", "remote-mimo"
    recognizer_backend: str = "funasr"

    # sherpa-sensevoice backend paths (required when recognizer_backend="sherpa-sensevoice")
    sensevoice_model_path: str = ""   # path to model.int8.onnx
    sensevoice_tokens_path: str = ""  # path to tokens.txt
    sensevoice_language: str = "zh"

    # sensevoice-lm backend (SenseVoice CTC + KenLM rescoring)
    # Uses same model/tokens as sherpa-sensevoice, plus a KenLM language model
    sensevoice_lm_path: str = ""      # path to KenLM .bin or .arpa file
    sensevoice_lm_alpha: float = 0.5  # LM weight (higher = trust LM more)
    sensevoice_lm_beta: float = 1.0   # word insertion bonus
    sensevoice_lm_beam_width: int = 20  # beam search width

    # sherpa-nano backend paths (required when recognizer_backend="sherpa-nano")
    nano_model_dir: str = ""          # dir containing encoder_adaptor/llm/embedding/tokenizer
    # sherpa-nano LLM prompts (optional; defaults match sherpa-onnx)
    nano_system_prompt: str = "You are a helpful assistant."
    nano_user_prompt: str = "语音转写:"

    # qwen3-asr backend paths (required when recognizer_backend="qwen3-asr")
    qwen3_model_dir: str = ""    # dir containing conv_frontend/encoder/decoder/tokenizer
    qwen3_max_total_len: int = 512   # KV cache length (increase for longer audio)
    qwen3_max_new_tokens: int = 128  # max output tokens (increase for longer audio)

    # remote-mimo backend (required when recognizer_backend="remote-mimo")
    mimo_base_url: str = ""      # MiMo-V2.5-ASR Gradio server, e.g. http://192.168.192.118:7898
    mimo_language: str = "Auto"  # Auto / Chinese / English
    mimo_timeout: float = 60.0   # per-request HTTP timeout in seconds

    # Hotkey
    hotkey: str = "scroll_lock"
    toggle_hotkey: str = ""  # Toggle mode: press once to start, again to stop (e.g. "alt")

    # UI
    window_title: str = "Rtxime"
    window_width: int = 200
    window_height: int = 100
    window_opacity: float = 0.8

    # Streaming
    streaming: bool = False  # Enable real-time streaming recognition
    streaming_model_dir: str = ""  # Path to online model (required if streaming=True)
    chunk_size: list = None  # [left, body, right] in frames, default [5, 10, 5]
    two_pass: bool = False  # 2pass mode: stream + offline correction

    # Features
    enable_hotwords: bool = True
    enable_traditional_chinese: bool = True
    enable_noise_reduction: bool = True
    strip_trailing_punctuation: bool = True  # remove "。", "！", "?", etc. at end

    # VAD (Voice Activity Detection) for long audio segmentation
    enable_vad: bool = True              # auto-segment long audio before ASR
    vad_max_speech_duration: float = 15.0  # max seconds per segment
    vad_model_path: str = ""             # path to silero_vad.onnx (auto-detected if empty)

    # Platform override (auto-detected if empty)
    platform: str = ""

    # Paths (resolved relative to config file or package)
    hotwords_file: str = "hotwords.txt"
    library_file: str = "library.txt"
    warmup_file: str = "warmup.wav"
    icon_file: str = "icon.png"
    style_file: str = "style.css"


def detect_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform == "win32":
        return "windows"
    elif sys.platform.startswith("linux"):
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            return "wayland"
        return "x11"
    return "x11"


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    config = Config()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    if not config.platform:
        config.platform = detect_platform()

    return config


def resolve_resource_path(config: Config, attr: str) -> Path:
    """Resolve a resource file path. Checks CWD first, then package resources dir."""
    filename = getattr(config, attr)
    # Check CWD
    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return cwd_path
    # Check package resources
    pkg_path = Path(__file__).parent / "resources" / filename
    if pkg_path.exists():
        return pkg_path
    # Fall back to CWD path (will fail later with a clear error)
    return cwd_path
