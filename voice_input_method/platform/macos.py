"""macOS backend: clipboard + Cmd+V paste via pynput.

Requires Accessibility permissions in System Settings > Privacy & Security.
"""

import sys

from PyQt5.QtWidgets import QApplication
from pynput.keyboard import Controller, Key

from .base import PlatformBackend


class MacOSBackend(PlatformBackend):
    def __init__(self):
        self._keyboard = Controller()

    def paste_text(self, text: str):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        with self._keyboard.pressed(Key.cmd):
            self._keyboard.press("v")
            self._keyboard.release("v")

    def check_permissions(self) -> list[str]:
        """Check macOS Accessibility permissions."""
        missing = []
        if sys.platform != "darwin":
            return missing
        try:
            from ApplicationServices import AXIsProcessTrusted
            if not AXIsProcessTrusted():
                missing.append(
                    "Accessibility permission required: "
                    "System Settings > Privacy & Security > Accessibility"
                )
        except ImportError:
            # PyObjC not installed, can't check
            pass
        return missing
