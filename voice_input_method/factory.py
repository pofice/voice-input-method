"""Factory functions for assembling VoiceEngine from Config.

This module bridges Config (what the user wants) and VoiceEngine (how it
runs).  It is completely GUI-free — a CLI, web service, or test harness
can call create_engine() without importing PySide6.
"""

from __future__ import annotations

from .config import Config, resolve_resource_path
from .audio import AudioRecorder
from .recognition import SpeechRecognizer, StreamingRecognizer
from .text_processing import ChineseConverter
from .hotwords import HotwordManager
from .platform import get_backend
from .engine import VoiceEngine, EngineConfig


def create_engine(
    config: Config,
    on_partial=None,
    on_result=None,
) -> VoiceEngine:
    """Assemble a fully-wired VoiceEngine from a Config object.

    Returns a VoiceEngine that is ready to .start().  The caller is
    responsible for calling .shutdown() when done.
    """
    # Platform backend (text pasting)
    backend = get_backend(config.platform)

    # Streaming recognizer (optional)
    streaming_recognizer: StreamingRecognizer | None = None
    if config.streaming:
        chunk_size = config.chunk_size or [5, 10, 5]
        streaming_recognizer = StreamingRecognizer(
            model_dir=config.streaming_model_dir,
            quantize=config.quantize,
            chunk_size=chunk_size,
        )

    # Engine config (subset of Config without UI fields)
    engine_config = EngineConfig(
        streaming=config.streaming,
        two_pass=config.two_pass,
        enable_number_conversion=config.enable_number_conversion,
        enable_traditional_chinese=config.enable_traditional_chinese,
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

    # Offline recognizer
    recognizer = SpeechRecognizer(
        model_type=config.model_type,
        model_dir=config.model_dir,
        quantize=config.quantize,
    )

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
    )

    # Wire streaming chunk callback
    if config.streaming:
        recorder._on_chunk = engine.on_audio_chunk

    return engine
