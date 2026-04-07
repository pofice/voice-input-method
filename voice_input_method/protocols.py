"""Protocol definitions for dependency injection and testability.

Each protocol defines the minimal interface a component must satisfy.
Test code can provide lightweight mock implementations without importing
heavy dependencies (funasr_onnx, sounddevice, PySide6, pynput, etc.).
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioSource(Protocol):
    """Captures audio from a microphone or other source."""

    def start(self) -> None:
        """Initialize and begin listening (but not recording yet)."""
        ...

    def start_recording(self) -> None:
        """Begin capturing audio data."""
        ...

    def stop_recording(self, output_path: str) -> str:
        """Stop capturing and save to *output_path*. Returns the path."""
        ...

    def flush_streaming_buffer(self) -> None:
        """Send any remaining buffered audio to the streaming callback."""
        ...

    def stop(self) -> None:
        """Tear down the audio stream."""
        ...


@runtime_checkable
class Recognizer(Protocol):
    """Offline (batch) speech-to-text."""

    def load(self) -> None:
        ...

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        ...

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        ...


@runtime_checkable
class StreamingRecognizerProto(Protocol):
    """Real-time streaming speech-to-text."""

    @property
    def step_samples(self) -> int:
        """Number of 16 kHz samples per chunk."""
        ...

    def load(self) -> None:
        ...

    def reset(self) -> None:
        ...

    def feed_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        ...


@runtime_checkable
class TextPaster(Protocol):
    """Pastes text at the current cursor position (platform-specific)."""

    def paste_text(self, text: str) -> None:
        ...

    def check_permissions(self) -> list[str]:
        ...


@runtime_checkable
class HotwordProvider(Protocol):
    """Supplies a hotword string for the ASR model."""

    @property
    def hotwords_str(self) -> str:
        ...

    def reload(self) -> None:
        ...


@runtime_checkable
class RecordingIndicator(Protocol):
    """Shows/hides a visual recording indicator overlay."""

    def show(self, style: str = "dot") -> None:
        """Display the recording indicator. Style: 'dot' or 'ring'."""
        ...

    def hide(self) -> None:
        """Hide the recording indicator."""
        ...

    def shutdown(self) -> None:
        """Release resources."""
        ...
