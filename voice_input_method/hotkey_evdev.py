"""Global hotkey listener for Wayland — reads /dev/input via evdev.

Why this exists: Wayland deliberately prevents applications from observing
keystrokes they don't have focus for, so pynput's X11 backend only sees keys
typed into XWayland windows. Reading the kernel's input devices bypasses the
display server entirely, which is the only way to get a truly global hotkey
on a native Wayland session.

Requires the user to be in the `input` group (see README). Without it,
opening /dev/input/event* raises PermissionError.

Mirrors CombinedHotkeyListener's interface so factory can swap the two.
"""

from __future__ import annotations

import selectors
import threading
from typing import Callable

# Hotkey name -> evdev key code name. Keeps config.yaml's vocabulary
# (shared with the pynput backend) independent of evdev's KEY_* spelling.
_KEY_NAMES = {
    "scroll_lock": "KEY_SCROLLLOCK",
    "pause": "KEY_PAUSE",
    "caps_lock": "KEY_CAPSLOCK",
    "alt": "KEY_LEFTALT",
    "alt_r": "KEY_RIGHTALT",
    "ctrl": "KEY_LEFTCTRL",
    "ctrl_r": "KEY_RIGHTCTRL",
    "shift": "KEY_LEFTSHIFT",
    "shift_r": "KEY_RIGHTSHIFT",
    **{f"f{n}": f"KEY_F{n}" for n in range(1, 13)},
}

_KEY_RELEASE = 0
_KEY_PRESS = 1
# value 2 is auto-repeat; ignored so holding a key doesn't re-trigger


def _resolve_code(name: str) -> int:
    """Map a hotkey name to an evdev key code. Falls back to F6."""
    from evdev import ecodes

    key_name = _KEY_NAMES.get(name, "KEY_F6")
    return getattr(ecodes, key_name, ecodes.KEY_F6)


def find_keyboards() -> list:
    """Return evdev devices that look like keyboards.

    A device qualifies if it reports EV_KEY and covers the letter range —
    this filters out mice, power buttons and other EV_KEY-capable devices
    that would otherwise be opened for nothing.
    """
    from evdev import InputDevice, ecodes, list_devices

    keyboards = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except (PermissionError, OSError):
            continue
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_A in keys and ecodes.KEY_Z in keys:
            keyboards.append(dev)
        else:
            dev.close()
    return keyboards


class EvdevCombinedHotkeyListener:
    """Hold + toggle hotkeys read straight from the kernel input layer.

    Interface-compatible with hotkey.CombinedHotkeyListener.
    """

    def __init__(
        self,
        hold_hotkey: str,
        hold_on_press: Callable,
        hold_on_release: Callable,
        toggle_hotkey: str | None = None,
        toggle_on_start: Callable | None = None,
        toggle_on_stop: Callable | None = None,
    ):
        self._hold_name = hold_hotkey
        self._hold_on_press = hold_on_press
        self._hold_on_release = hold_on_release
        self._hold_pressed = False

        self._toggle_name = toggle_hotkey
        self._toggle_on_start = toggle_on_start
        self._toggle_on_stop = toggle_on_stop
        self._toggle_recording = False

        self._devices: list = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Open every keyboard and watch them from a daemon thread."""
        self._devices = find_keyboards()
        if not self._devices:
            raise PermissionError(
                "No readable keyboard found in /dev/input. "
                "Add yourself to the 'input' group and re-login: "
                "sudo usermod -aG input $USER"
            )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        hold_code = _resolve_code(self._hold_name)
        toggle_code = _resolve_code(self._toggle_name) if self._toggle_name else None

        sel = selectors.DefaultSelector()
        for dev in self._devices:
            sel.register(dev, selectors.EVENT_READ)
        try:
            while not self._stop_event.is_set():
                # Timeout keeps stop() responsive even when no keys are pressed
                for key, _ in sel.select(timeout=0.2):
                    for event in key.fileobj.read():
                        self._handle(event, hold_code, toggle_code)
        finally:
            sel.close()
            for dev in self._devices:
                try:
                    dev.close()
                except OSError:
                    pass
            self._devices = []

    def _handle(self, event, hold_code: int, toggle_code: int | None) -> None:
        from evdev import ecodes

        if event.type != ecodes.EV_KEY or event.value not in (_KEY_PRESS, _KEY_RELEASE):
            return

        # Hold hotkey — suppressed while a toggle recording is in progress
        if event.code == hold_code and not self._toggle_recording:
            if event.value == _KEY_PRESS and not self._hold_pressed:
                self._hold_pressed = True
                self._hold_on_press()
            elif event.value == _KEY_RELEASE and self._hold_pressed:
                self._hold_pressed = False
                self._hold_on_release()

        # Toggle hotkey — suppressed while holding, and only on press
        if (
            toggle_code is not None
            and event.code == toggle_code
            and event.value == _KEY_PRESS
            and not self._hold_pressed
        ):
            self._do_toggle()

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
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def hold_pressed(self) -> bool:
        return self._hold_pressed

    @property
    def toggle_recording(self) -> bool:
        return self._toggle_recording
