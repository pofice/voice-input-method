"""Wayland backend: clipboard + Ctrl+V paste via ydotool.

Wayland has no equivalent of XTEST, so neither pynput nor xdotool can send
keystrokes to native Wayland windows — they only reach XWayland clients.
ydotool sidesteps this by injecting events through the kernel's uinput
device, which the compositor consumes like any real keyboard.

Requires:
  - ydotoold running (systemd user service, ships with the ydotool package)
  - the user in the `input` group, for /dev/uinput access

Pasting beats typing character-by-character here: it is one keystroke
regardless of length, and CJK text needs no keymap juggling.
"""

import os
import subprocess

from PySide6.QtWidgets import QApplication

from .base import PlatformBackend

_SOCKET = os.environ.get("YDOTOOL_SOCKET", f"/run/user/{os.getuid()}/.ydotool_socket")


class WaylandBackend(PlatformBackend):
    def paste_text(self, text: str):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        # 29=KEY_LEFTCTRL, 47=KEY_V; ":1" is press, ":0" is release
        subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            env={**os.environ, "YDOTOOL_SOCKET": _SOCKET},
            check=False,
        )

    def check_permissions(self) -> list[str]:
        """Report what's missing for ydotool to work."""
        missing = []
        if not os.path.exists(_SOCKET):
            missing.append(
                f"ydotoold is not running (socket {_SOCKET} missing). "
                "Start it with: systemctl --user start ydotool"
            )
        if not os.access("/dev/uinput", os.W_OK):
            missing.append(
                "No write access to /dev/uinput. "
                "Run: sudo usermod -aG input $USER, then re-login"
            )
        return missing
