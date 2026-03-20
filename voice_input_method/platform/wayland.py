"""Wayland backend: xdotool character-by-character input."""

import subprocess

from .base import PlatformBackend


class WaylandBackend(PlatformBackend):
    def paste_text(self, text: str):
        for char in text:
            subprocess.run(["xdotool", "type", "--clearmodifiers", char])
