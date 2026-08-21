"""Wayland backend: clipboard + Ctrl+V paste via wl-copy + ydotool.

Wayland has no equivalent of XTEST, so neither pynput nor xdotool can send
keystrokes to native Wayland windows — they only reach XWayland clients.
ydotool sidesteps this by injecting events through the kernel's uinput
device, which the compositor consumes like any real keyboard.

Requires:
  - ydotoold running (systemd user service, ships with the ydotool package)
  - the user in the `input` group, for /dev/uinput access
  - wl-clipboard (wl-copy / wl-paste)

Pasting beats typing character-by-character here: it is one keystroke
regardless of length, and CJK text needs no keymap juggling.

The clipboard is written with wl-copy rather than QClipboard on purpose:
paste_text runs on the transcription worker thread, and QClipboard is
GUI-thread-only — on Wayland the selection is handed over by the Qt event
loop, so a worker-thread write can land *after* the Ctrl+V and paste stale
text. wl-copy owns the selection in its own process, and we read it back
before sending the keystroke.
"""

import os
import shutil
import subprocess
import time

from .base import PlatformBackend

_SOCKET = os.environ.get("YDOTOOL_SOCKET", f"/run/user/{os.getuid()}/.ydotool_socket")

# How long to wait for the compositor to report our text as the current
# selection before pasting anyway. Handover is normally sub-millisecond.
_CLIPBOARD_TIMEOUT = 1.0
_CLIPBOARD_POLL = 0.02


class WaylandBackend(PlatformBackend):
    def paste_text(self, text: str):
        if not self._set_clipboard(text):
            # Clipboard is not ours — pasting now would insert whatever the
            # previous owner holds, which is worse than doing nothing.
            return
        # 29=KEY_LEFTCTRL, 47=KEY_V; ":1" is press, ":0" is release
        subprocess.run(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            env={**os.environ, "YDOTOOL_SOCKET": _SOCKET},
            check=False,
        )

    def _set_clipboard(self, text: str) -> bool:
        """Write text to the Wayland clipboard, then confirm it took effect."""
        try:
            subprocess.run(
                ["wl-copy", "--type", "text/plain;charset=utf-8"],
                input=text.encode("utf-8"),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False

        deadline = time.monotonic() + _CLIPBOARD_TIMEOUT
        while time.monotonic() < deadline:
            if self._read_clipboard() == text:
                return True
            time.sleep(_CLIPBOARD_POLL)
        return False

    @staticmethod
    def _read_clipboard() -> str | None:
        try:
            done = subprocess.run(
                ["wl-paste", "--no-newline", "--type", "text/plain;charset=utf-8"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.decode("utf-8", errors="replace")

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
        for tool in ("wl-copy", "wl-paste"):
            if shutil.which(tool) is None:
                missing.append(
                    f"{tool} not found. Install wl-clipboard: "
                    "sudo apt install wl-clipboard"
                )
        return missing
