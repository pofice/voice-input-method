"""VoiceEngine — the non-GUI core of voice-input-method.

Orchestrates: audio capture → ASR (offline / streaming / 2pass) → text
post-processing → paste.  Every dependency is injected via the constructor
so the whole pipeline can be tested with lightweight mocks.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    import noisereduce as _nr  # pre-import to avoid delay on first use
except ImportError:
    _nr = None

from .protocols import (
    AudioSource,
    HotwordProvider,
    Recognizer,
    StreamingRecognizerProto,
    TextPaster,
)
from .text_processing import clean_spaces


@dataclass
class EngineConfig:
    """Subset of Config that the engine actually needs (no UI fields)."""

    streaming: bool = False
    two_pass: bool = False
    enable_traditional_chinese: bool = False
    enable_noise_reduction: bool = True
    chunk_size: list[int] = field(default_factory=lambda: [5, 10, 5])


class VoiceEngine:
    """Core voice-input pipeline, independent of any GUI framework.

    Lifecycle::

        engine = VoiceEngine(...)
        engine.start()           # init audio, load models
        engine.start_recording() # user presses hotkey
        engine.stop_recording()  # user releases hotkey → triggers ASR
        engine.shutdown()        # cleanup
    """

    def __init__(
        self,
        config: EngineConfig,
        recorder: AudioSource,
        recognizer: Recognizer,
        paster: TextPaster,
        streaming_recognizer: StreamingRecognizerProto | None = None,
        hotword_provider: HotwordProvider | None = None,
        chinese_converter=None,
        on_partial: Callable[[str], None] | None = None,
        on_result: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self.config = config
        self.recorder = recorder
        self.recognizer = recognizer
        self.paster = paster
        self.streaming_recognizer = streaming_recognizer
        self.hotword_provider = hotword_provider
        self.chinese_converter = chinese_converter

        # Callbacks — UI layer hooks into these
        self.on_partial = on_partial
        self.on_result = on_result
        self.on_error = on_error

        self._audio_path = os.path.join(tempfile.gettempdir(), "voice_input_audio.wav")
        self._streaming_text = ""
        self._convert_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load models and start the audio stream."""
        self.recognizer.load()
        warmup_wav = ""  # caller can set a real path via warmup()
        self.recognizer.warmup(warmup_wav, self._hotwords())

        if self.streaming_recognizer:
            self.streaming_recognizer.load()

        self.recorder.start()

    def warmup(self, warmup_wav: str) -> None:
        """Explicit warmup with a specific WAV file."""
        self.recognizer.warmup(warmup_wav, self._hotwords())

    def shutdown(self) -> None:
        """Release resources."""
        self.recorder.stop()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin a new recording segment."""
        self._streaming_text = ""
        if self.streaming_recognizer:
            self.streaming_recognizer.reset()
        self.recorder.start_recording()

    def stop_recording(self) -> None:
        """End recording, trigger recognition, and paste the result."""
        # Flush remaining streaming audio
        if self.config.streaming:
            self.recorder.flush_streaming_buffer()
            if self.streaming_recognizer:
                final_text = self.streaming_recognizer.feed_chunk(
                    np.array([], dtype=np.float32), is_final=True
                )
                if final_text:
                    self._streaming_text += final_text

        self.recorder.stop_recording(self._audio_path)

        if self.config.streaming and not self.config.two_pass:
            # Pure streaming: use accumulated result
            text = clean_spaces(self._streaming_text)
            if text:
                self._deliver(text)
        else:
            # Offline or 2pass final correction
            threading.Thread(target=self._transcribe_offline, daemon=True).start()

    # ------------------------------------------------------------------
    # Audio chunk callback (wired to AudioRecorder.on_chunk)
    # ------------------------------------------------------------------

    def on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Called from the audio thread with each 16 kHz mono chunk."""
        if self.streaming_recognizer:
            text = self.streaming_recognizer.feed_chunk(chunk, is_final=False)
            if text:
                self._streaming_text += text
                display = clean_spaces(self._streaming_text)
                if self.on_partial:
                    self.on_partial(display)

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def convert_chinese(self, text: str) -> str | None:
        """Convert between traditional/simplified Chinese. Returns converted text or None."""
        if not self.chinese_converter:
            return None
        with self._convert_lock:
            if not text:
                return None
            return self.chinese_converter.convert(text)

    def convert_chinese_async(self, text: str) -> None:
        """Convert text in a background thread, then fire on_result callback."""
        def _run():
            converted = self.convert_chinese(text)
            if converted and self.on_result:
                self.on_result(converted)
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hotwords(self) -> str:
        if self.hotword_provider:
            return self.hotword_provider.hotwords_str
        return ""

    def _transcribe_offline(self) -> None:
        """Run offline transcription (non-streaming or 2pass final pass).

        Runs in a background thread. Exceptions are routed to the
        on_error callback (if set) so the UI layer can surface them,
        instead of being silently lost on the thread.
        """
        try:
            audio_path = self._audio_path
            if self.config.enable_noise_reduction:
                audio_path = self._denoise(audio_path)
            text = self.recognizer.transcribe(audio_path, self._hotwords())
            if text:
                text = clean_spaces(text)
                self._deliver(text)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if self.on_error:
                try:
                    self.on_error(exc)
                except Exception:
                    traceback.print_exc()

    def _denoise(self, audio_path: str) -> str:
        """Apply noise reduction to the recorded audio."""
        try:
            import soundfile as sf
            if _nr is None:
                return audio_path
            data, sr = sf.read(audio_path, dtype="float32")
            reduced = _nr.reduce_noise(y=data, sr=sr, prop_decrease=0.8)
            denoised_path = audio_path.replace(".wav", "_denoised.wav")
            sf.write(denoised_path, reduced, sr)
            return denoised_path
        except Exception:
            return audio_path

    def _deliver(self, text: str) -> None:
        """Post-process and deliver the final result."""
        if self.on_result:
            self.on_result(text)
        self.paster.paste_text(text)
