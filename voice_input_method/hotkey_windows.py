"""Windows global hotkeys using the low-level ``keyboard`` package."""

from __future__ import annotations

import threading
import time
from typing import Callable


_SHORT_PRESS_THRESHOLD = 0.3


_KEY_NAMES = {
    "caps_lock": "caps lock",
    "scroll_lock": "scroll lock",
    "alt_l": "left alt",
    "alt_r": "right alt",
    "ctrl_l": "left ctrl",
    "ctrl_r": "right ctrl",
    "shift_l": "left shift",
    "shift_r": "right shift",
}


def _names_of(hotkey: str | list[str] | None) -> list[str]:
    if not hotkey:
        return []
    return [hotkey] if isinstance(hotkey, str) else list(hotkey)


def _resolve_name(name: str) -> str:
    """Convert this project's snake_case names to keyboard's names."""
    return _KEY_NAMES.get(name.lower(), name.lower().replace("_", " "))


class WindowsKeyboardHotkeyListener:
    """Combined hold/toggle listener backed by Windows keyboard hooks.

    The ``keyboard`` package is imported only when the listener starts, so
    importing this module remains safe in headless environments and tests.
    CapsLock hooks are suppressed so speaking does not toggle uppercase.
    """

    def __init__(
        self,
        hold_hotkey: str | list[str],
        hold_on_press: Callable,
        hold_on_release: Callable,
        toggle_hotkey: str | list[str] | None = None,
        toggle_on_start: Callable | None = None,
        toggle_on_stop: Callable | None = None,
    ):
        self._hold_names = {_resolve_name(name) for name in _names_of(hold_hotkey)}
        self._toggle_names = {_resolve_name(name) for name in _names_of(toggle_hotkey)}
        self._hold_on_press = hold_on_press
        self._hold_on_release = hold_on_release
        self._toggle_on_start = toggle_on_start
        self._toggle_on_stop = toggle_on_stop
        self._hold_pressed = False
        self._toggle_recording = False
        self._toggle_key_down: set[str] = set()
        self._hold_started_at: float | None = None
        self._replaying_caps = False
        self._hooks = []
        self._keyboard = None
        self._lock = threading.RLock()

    def start(self) -> None:
        import keyboard

        with self._lock:
            if self._hooks:
                return
            self._keyboard = keyboard
            names = self._hold_names | self._toggle_names
            try:
                for name in names:
                    hook = keyboard.hook_key(
                        name,
                        lambda event, key_name=name: self._handle(key_name, event),
                        suppress=name == "caps lock",
                    )
                    self._hooks.append(hook)
            except Exception:
                for hook in self._hooks:
                    keyboard.unhook(hook)
                self._hooks.clear()
                self._keyboard = None
                raise

    def _handle(self, name: str, event) -> None:
        event_type = getattr(event, "event_type", "")
        if event_type not in ("down", "up"):
            return

        with self._lock:
            if self._replaying_caps and name == "caps lock":
                return
            if event_type == "down":
                if name in self._hold_names and not self._toggle_recording:
                    if not self._hold_pressed:
                        self._hold_pressed = True
                        self._hold_started_at = time.monotonic()
                        self._hold_on_press()
                if name in self._toggle_names and not self._hold_pressed:
                    if name not in self._toggle_key_down:
                        self._toggle_key_down.add(name)
                        self._do_toggle()
            else:
                self._toggle_key_down.discard(name)
                if name in self._hold_names and self._hold_pressed:
                    duration = time.monotonic() - (self._hold_started_at or time.monotonic())
                    self._hold_pressed = False
                    self._hold_started_at = None
                    self._hold_on_release()
                    if name == "caps lock" and duration < _SHORT_PRESS_THRESHOLD:
                        # A short CapsLock press retains its normal system behavior.
                        self._replaying_caps = True
                        try:
                            self._keyboard.press("caps lock")
                            self._keyboard.release("caps lock")
                        finally:
                            self._replaying_caps = False

    def _do_toggle(self) -> None:
        if not self._toggle_recording:
            self._toggle_recording = True
            if self._toggle_on_start:
                self._toggle_on_start()
        else:
            self._toggle_recording = False
            if self._toggle_on_stop:
                self._toggle_on_stop()

    def stop(self) -> None:
        with self._lock:
            if self._keyboard:
                for hook in self._hooks:
                    self._keyboard.unhook(hook)
            self._hooks.clear()
            self._keyboard = None
            self._hold_pressed = False
            self._hold_started_at = None
            self._toggle_recording = False
            self._toggle_key_down.clear()

    @property
    def hold_pressed(self) -> bool:
        return self._hold_pressed

    @property
    def toggle_recording(self) -> bool:
        return self._toggle_recording
