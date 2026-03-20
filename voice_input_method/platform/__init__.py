"""Platform-specific backends for text input and clipboard operations."""

from .base import PlatformBackend
from .x11 import X11Backend
from .wayland import WaylandBackend
from .windows import WindowsBackend
from .macos import MacOSBackend


def get_backend(platform: str) -> PlatformBackend:
    """Get the appropriate platform backend."""
    backends = {
        "x11": X11Backend,
        "wayland": WaylandBackend,
        "windows": WindowsBackend,
        "macos": MacOSBackend,
    }
    cls = backends.get(platform, X11Backend)
    return cls()
