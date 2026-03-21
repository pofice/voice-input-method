"""Windows backend: clipboard + Ctrl+V paste via pynput."""

from PySide6.QtWidgets import QApplication
from pynput.keyboard import Controller, Key

from .base import PlatformBackend


class WindowsBackend(PlatformBackend):
    def __init__(self):
        self._keyboard = Controller()

    def paste_text(self, text: str):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")
