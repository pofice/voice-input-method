"""Global hotkey listener — independent of any GUI framework.

Uses pynput for cross-platform keyboard monitoring. The module only
imports pynput when start() is called, so headless environments can
import this module without error.
"""

from __future__ import annotations

from typing import Callable


# Map config hotkey names to pynput Key objects (resolved lazily)
_HOTKEY_NAMES = {
    "scroll_lock", "pause", "fn",
    "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}


class HotkeyListener:
    """Listens for a global hotkey press/release and fires callbacks.

    Usage::

        listener = HotkeyListener(
            hotkey="scroll_lock",
            on_press=lambda: engine.start_recording(),
            on_release=lambda: engine.stop_recording(),
        )
        listener.start()
        # ...
        listener.stop()
    """

    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ):
        self._hotkey_name = hotkey
        self._on_press = on_press
        self._on_release = on_release
        self._listener = None
        self._pressed = False

    def start(self) -> None:
        """Begin listening for the hotkey (imports pynput here)."""
        from pynput import keyboard

        key_map = {}
        for name in _HOTKEY_NAMES:
            if hasattr(keyboard.Key, name):
                key_map[name] = getattr(keyboard.Key, name)
        # macOS fn key (vk=63) is not in pynput's Key enum
        key_map["fn"] = keyboard.KeyCode.from_vk(63)
        target_key = key_map.get(self._hotkey_name, keyboard.Key.f6)

        # macOS fn key only fires release events, so use toggle mode for it
        fn_toggle = self._hotkey_name == "fn"

        if fn_toggle:
            def on_press(key):
                pass

            def on_release(key):
                if key == target_key:
                    if not self._pressed:
                        self._pressed = True
                        self._on_press()
                    else:
                        self._pressed = False
                        self._on_release()
        else:
            def on_press(key):
                try:
                    if key == target_key and not self._pressed:
                        self._pressed = True
                        self._on_press()
                except AttributeError:
                    pass

            def on_release(key):
                if key == target_key and self._pressed:
                    self._pressed = False
                    self._on_release()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        """Stop listening."""
        if self._listener:
            self._listener.stop()
            self._listener = None

    @property
    def is_pressed(self) -> bool:
        return self._pressed
