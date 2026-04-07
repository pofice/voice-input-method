"""Hotword loading and file watching.

Core loading logic is pure Python (no Qt dependency).
Qt-based file watching is optional and only used when start_watching() is called.
"""

from pathlib import Path
from typing import Callable


class HotwordManager:
    MAX_LENGTH = 10

    def __init__(self, hotwords_path: Path, on_reload: Callable[[], None] | None = None):
        self.path = hotwords_path
        self._hotwords_str: str = ""
        self._on_reload = on_reload
        self._watcher = None  # lazy: created only if start_watching() is called
        self.reload()

    @property
    def hotwords_str(self) -> str:
        return self._hotwords_str

    def reload(self) -> None:
        """Load hotwords from file. Pure I/O, no Qt needed."""
        try:
            hotwords: list[str] = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    while len(line) > self.MAX_LENGTH:
                        hotwords.append(line[: self.MAX_LENGTH])
                        line = line[self.MAX_LENGTH :]
                    hotwords.append(line)
            self._hotwords_str = " ".join(hotwords)
        except FileNotFoundError:
            self._hotwords_str = ""
        except Exception as e:
            print(f"Error loading hotwords: {e}")

    def start_watching(self) -> None:
        """Start watching the hotwords file for changes (requires PySide6)."""
        from PySide6.QtCore import QFileSystemWatcher

        self._watcher = QFileSystemWatcher()
        if self.path.exists():
            self._watcher.addPath(str(self.path))
        self._watcher.fileChanged.connect(self._on_file_changed)

    def _on_file_changed(self) -> None:
        self.reload()
        # Re-add watch (some systems remove it after file modification)
        path_str = str(self.path)
        if self._watcher and path_str not in self._watcher.files():
            self._watcher.addPath(path_str)
        if self._on_reload:
            self._on_reload()

    def stop_watching(self) -> None:
        if self._watcher:
            self._watcher.fileChanged.disconnect(self._on_file_changed)
            self._watcher = None
