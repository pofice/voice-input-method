"""Windows backend: native clipboard + Ctrl+V paste via pynput."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from pynput.keyboard import Controller, Key

from .base import PlatformBackend

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002


class WindowsBackend(PlatformBackend):
    def __init__(self):
        self._keyboard = Controller()
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        self._kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        self._kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalLock.restype = ctypes.c_void_p
        self._kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        self._user32.OpenClipboard.argtypes = [wintypes.HWND]
        self._user32.OpenClipboard.restype = wintypes.BOOL
        self._user32.EmptyClipboard.restype = wintypes.BOOL
        self._user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HGLOBAL]
        self._user32.SetClipboardData.restype = wintypes.HGLOBAL
        self._user32.CloseClipboard.restype = wintypes.BOOL

    def paste_text(self, text: str):
        """Write text synchronously to Win32 clipboard, then paste it.

        This runs on the engine worker thread, so it deliberately avoids
        QApplication.clipboard(), whose Qt object belongs to the GUI thread.
        """
        if not self._set_clipboard(text):
            return
        # Give the target application a moment to observe the new clipboard
        # owner before sending Ctrl+V, especially after a previous paste.
        time.sleep(0.03)
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")

    def _set_clipboard(self, text: str) -> bool:
        """Set CF_UNICODETEXT and return False if Windows rejects the write."""
        encoded = (text + "\0").encode("utf-16-le")
        handle = self._kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(encoded))
        if not handle:
            return False

        pointer = self._kernel32.GlobalLock(handle)
        if not pointer:
            self._kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, encoded, len(encoded))
        self._kernel32.GlobalUnlock(handle)

        opened = False
        try:
            for _ in range(10):
                if self._user32.OpenClipboard(None):
                    opened = True
                    break
                time.sleep(0.01)
            if not opened:
                return False

            self._user32.EmptyClipboard()
            # Ownership transfers to Windows on success; do not free handle.
            if not self._user32.SetClipboardData(_CF_UNICODETEXT, handle):
                return False
            handle = None
            return True
        finally:
            if opened:
                self._user32.CloseClipboard()
            if handle:
                self._kernel32.GlobalFree(handle)
