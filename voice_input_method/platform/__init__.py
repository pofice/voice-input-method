"""Platform-specific backends for text input and clipboard operations.

Imports are lazy so that headless/CI environments don't fail when
pynput or PySide6 are unavailable or require a display server.
"""

from __future__ import annotations

from .base import PlatformBackend


def get_backend(platform: str) -> PlatformBackend:
    """Get the appropriate platform backend (lazy import)."""
    if platform == "wayland":
        from .wayland import WaylandBackend
        return WaylandBackend()
    elif platform == "windows":
        from .windows import WindowsBackend
        return WindowsBackend()
    elif platform == "macos":
        from .macos import MacOSBackend
        return MacOSBackend()
    else:
        from .x11 import X11Backend
        return X11Backend()
