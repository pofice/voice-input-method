"""Hotword loading and file watching."""

from pathlib import Path
from PySide6.QtCore import QFileSystemWatcher


class HotwordManager:
    MAX_LENGTH = 10

    def __init__(self, hotwords_path: Path, on_reload=None):
        self.path = hotwords_path
        self.hotwords_str: str = ""
        self._on_reload = on_reload
        self._watcher: QFileSystemWatcher | None = None
        self.reload()

    def reload(self):
        """Load hotwords from file."""
        try:
            hotwords = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    while len(line) > self.MAX_LENGTH:
                        hotwords.append(line[:self.MAX_LENGTH])
                        line = line[self.MAX_LENGTH:]
                    hotwords.append(line)
            self.hotwords_str = " ".join(hotwords)
            print("Hotwords loaded successfully")
        except FileNotFoundError:
            print(f"Hotwords file not found: {self.path}")
            self.hotwords_str = ""
        except Exception as e:
            print(f"Error loading hotwords: {e}")

    def start_watching(self):
        """Start watching the hotwords file for changes."""
        self._watcher = QFileSystemWatcher()
        if self.path.exists():
            self._watcher.addPath(str(self.path))
        self._watcher.fileChanged.connect(self._on_file_changed)

    def _on_file_changed(self):
        self.reload()
        # Re-add watch (some systems remove it after file modification)
        path_str = str(self.path)
        if path_str not in self._watcher.files():
            self._watcher.addPath(path_str)
        if self._on_reload:
            self._on_reload()

    def stop_watching(self):
        if self._watcher:
            self._watcher.fileChanged.disconnect(self._on_file_changed)
            self._watcher = None
