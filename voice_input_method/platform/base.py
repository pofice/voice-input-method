"""Abstract base class for platform backends."""

from abc import ABC, abstractmethod


class PlatformBackend(ABC):
    """Handles platform-specific text input and clipboard operations."""

    @abstractmethod
    def paste_text(self, text: str):
        """Input text at the current cursor position."""
        ...

    def check_permissions(self) -> list[str]:
        """Check required platform permissions. Returns list of missing permissions."""
        return []
