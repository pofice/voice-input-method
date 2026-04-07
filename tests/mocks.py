"""Shared mock implementations for voice-input-method tests.

All heavy dependencies (funasr_onnx, sounddevice, PySide6, pynput) are
replaced with lightweight mocks so tests run in headless CI environments.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class MockRecorder:
    """Fake audio source that replays a pre-set buffer."""

    def __init__(self):
        self.started = False
        self.recording = False
        self.stopped = False
        self._on_chunk: Callable | None = None
        self._saved_path: str | None = None

    def start(self) -> None:
        self.started = True

    def start_recording(self) -> None:
        self.recording = True

    def stop_recording(self, output_path: str) -> str:
        self.recording = False
        self._saved_path = output_path
        return output_path

    def flush_streaming_buffer(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


class MockRecognizer:
    """Fake ASR that returns a canned transcription."""

    def __init__(self, text: str = "你好世界"):
        self._text = text
        self.loaded = False
        self.warmed_up = False

    def load(self) -> None:
        self.loaded = True

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        self.warmed_up = True

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        return self._text


class MockStreamingRecognizer:
    """Fake streaming ASR that accumulates chunks and returns text."""

    def __init__(self, chunks_text: list[str] | None = None):
        self._chunks_text = chunks_text or ["你", "好", "世界"]
        self._idx = 0
        self.loaded = False
        self._reset_count = 0

    @property
    def step_samples(self) -> int:
        return 9600  # 600ms at 16kHz

    def load(self) -> None:
        self.loaded = True

    def reset(self) -> None:
        self._idx = 0
        self._reset_count += 1

    def feed_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        if is_final or self._idx >= len(self._chunks_text):
            return ""
        text = self._chunks_text[self._idx]
        self._idx += 1
        return text


class MockPaster:
    """Fake paster that records what was pasted."""

    def __init__(self):
        self.pasted: list[str] = []

    def paste_text(self, text: str) -> None:
        self.pasted.append(text)

    def check_permissions(self) -> list[str]:
        return []


class MockHotwordProvider:
    """Fake hotword provider."""

    def __init__(self, hotwords: str = "测试 热词"):
        self._hotwords_str = hotwords

    @property
    def hotwords_str(self) -> str:
        return self._hotwords_str

    def reload(self) -> None:
        pass


class MockIndicator:
    """Fake recording indicator that tracks show/hide calls."""

    def __init__(self):
        self.visible = False
        self.show_count = 0
        self.hide_count = 0
        self.shutdown_called = False

    def show(self) -> None:
        self.visible = True
        self.show_count += 1

    def hide(self) -> None:
        self.visible = False
        self.hide_count += 1

    def shutdown(self) -> None:
        self.shutdown_called = True
