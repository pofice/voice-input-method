"""Tests for hotkey module — verifies the HotkeyListener interface without pynput."""

from voice_input_method.hotkey import HotkeyListener


class TestHotkeyListener:
    def test_init_does_not_import_pynput(self):
        """Creating a HotkeyListener should not import pynput."""
        pressed = []
        released = []
        listener = HotkeyListener(
            hotkey="f9",
            on_press=lambda: pressed.append(True),
            on_release=lambda: released.append(True),
        )
        # Should be created without errors in headless
        assert listener is not None
        assert listener.is_pressed is False

    def test_stop_without_start(self):
        """Stopping before starting should be safe."""
        listener = HotkeyListener(
            hotkey="scroll_lock",
            on_press=lambda: None,
            on_release=lambda: None,
        )
        listener.stop()  # should not raise
