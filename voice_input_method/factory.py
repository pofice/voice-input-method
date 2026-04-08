"""Factory functions for assembling VoiceEngine from Config.

This module bridges Config (what the user wants) and VoiceEngine (how it
runs).  It is completely GUI-free — a CLI, web service, or test harness
can call create_engine() without importing PySide6.
"""

from __future__ import annotations

from .config import Config, DEFAULT_OFFLINE_MODELS, resolve_resource_path
from .audio import AudioRecorder
from .indicator import NullIndicator
from .text_processing import ChineseConverter
from .hotwords import HotwordManager
from .platform import get_backend
from .engine import VoiceEngine, EngineConfig
from .protocols import RecordingIndicator, Recognizer


class ConfigError(ValueError):
    """Raised when required configuration for a backend is missing or invalid."""


def _create_recognizer(config: Config) -> Recognizer:
    """Create the appropriate recognizer based on config.recognizer_backend.

    Raises ConfigError when a sherpa backend is selected but required paths
    are missing, so the user gets a clear message instead of a cryptic
    FileNotFoundError from deep inside sherpa-onnx.
    """
    backend = config.recognizer_backend

    if backend == "sherpa-sensevoice":
        if not config.sensevoice_model_path or not config.sensevoice_tokens_path:
            raise ConfigError(
                "sherpa-sensevoice backend requires both 'sensevoice_model_path' "
                "and 'sensevoice_tokens_path' in config"
            )
        from .recognition.sherpa_sensevoice import SherpaSenseVoiceRecognizer
        return SherpaSenseVoiceRecognizer(
            model_path=config.sensevoice_model_path,
            tokens_path=config.sensevoice_tokens_path,
            language=config.sensevoice_language,
            num_threads=4,
        )

    if backend == "sherpa-nano":
        if not config.nano_model_dir:
            raise ConfigError(
                "sherpa-nano backend requires 'nano_model_dir' in config "
                "(directory containing encoder_adaptor/llm/embedding .onnx files)"
            )
        from .recognition.sherpa_nano import SherpaNanoRecognizer
        model_dir = config.nano_model_dir.rstrip("/")
        return SherpaNanoRecognizer(
            encoder_adaptor_path=f"{model_dir}/encoder_adaptor.int8.onnx",
            llm_path=f"{model_dir}/llm.int8.onnx",
            embedding_path=f"{model_dir}/embedding.int8.onnx",
            tokenizer_path=f"{model_dir}/Qwen3-0.6B",
            language="zh",
            num_threads=4,
        )

    if backend != "funasr":
        raise ConfigError(f"unknown recognizer_backend: {backend!r}")

    # Default: "funasr" — SeacoParaformer via funasr_onnx
    from .recognition.funasr_recognizer import FunASRRecognizer
    model_dir = config.model_dir or DEFAULT_OFFLINE_MODELS.get(
        config.model_type, DEFAULT_OFFLINE_MODELS["seaco_paraformer"]
    )
    return FunASRRecognizer(model_dir=model_dir, quantize=config.quantize)


def create_engine(
    config: Config,
    on_partial=None,
    on_result=None,
    on_error=None,
) -> VoiceEngine:
    """Assemble a fully-wired VoiceEngine from a Config object.

    Returns a VoiceEngine that is ready to .start().  The caller is
    responsible for calling .shutdown() when done.
    """
    # Platform backend (text pasting)
    backend = get_backend(config.platform)

    # Streaming recognizer (optional, funasr backend only)
    if config.streaming and config.recognizer_backend != "funasr":
        import warnings
        warnings.warn(
            f"streaming=True requires recognizer_backend='funasr', but got "
            f"'{config.recognizer_backend}'. Streaming will be disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
    streaming_recognizer = None
    if config.streaming and config.recognizer_backend == "funasr":
        from .recognition.funasr_recognizer import FunASRStreamingRecognizer
        chunk_size = config.chunk_size or [5, 10, 5]
        streaming_recognizer = FunASRStreamingRecognizer(
            model_dir=config.streaming_model_dir,
            quantize=config.quantize,
            chunk_size=chunk_size,
        )

    # Engine config (subset of Config without UI fields)
    engine_config = EngineConfig(
        streaming=config.streaming and config.recognizer_backend == "funasr",
        two_pass=config.two_pass and config.recognizer_backend == "funasr",
        enable_traditional_chinese=config.enable_traditional_chinese,
        enable_noise_reduction=config.enable_noise_reduction,
        chunk_size=config.chunk_size or [5, 10, 5],
    )

    # Chinese converter (optional)
    chinese_converter: ChineseConverter | None = None
    if config.enable_traditional_chinese:
        lib_path = resolve_resource_path(config, "library_file")
        chinese_converter = ChineseConverter(lib_path)

    # Hotword manager (optional)
    hotword_manager: HotwordManager | None = None
    if config.enable_hotwords:
        hw_path = resolve_resource_path(config, "hotwords_file")
        hotword_manager = HotwordManager(hw_path)

    # Offline recognizer — selected by backend config
    recognizer = _create_recognizer(config)

    # Audio recorder — chunk callback wired after engine creation
    chunk_samples = streaming_recognizer.step_samples if streaming_recognizer else 0
    recorder = AudioRecorder(
        sample_rate=config.sample_rate,
        channels=config.channels,
        on_chunk=None,
        chunk_samples=chunk_samples,
    )

    engine = VoiceEngine(
        config=engine_config,
        recorder=recorder,
        recognizer=recognizer,
        paster=backend,
        streaming_recognizer=streaming_recognizer,
        hotword_provider=hotword_manager,
        chinese_converter=chinese_converter,
        on_partial=on_partial,
        on_result=on_result,
        on_error=on_error,
    )

    # Wire streaming chunk callback
    if engine_config.streaming:
        recorder._on_chunk = engine.on_audio_chunk

    return engine


def create_indicator(platform: str) -> RecordingIndicator:
    """Create a platform-appropriate recording indicator.

    macOS: native AppKit floating panel (PyObjC subprocess).
    Other platforms / missing deps: silent no-op.
    """
    if platform == "macos":
        try:
            from .indicator import MacNativeIndicator
            return MacNativeIndicator()
        except (ImportError, OSError):
            pass
    return NullIndicator()
