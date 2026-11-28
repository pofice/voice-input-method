"""Tests for the Windows keyboard-hook backend without a real keyboard."""

import sys
import time
import types

import pytest

from voice_input_method.config import Config
from voice_input_method.factory import ConfigError, create_hotkey_listener
from voice_input_method.hotkey_windows import WindowsKeyboardHotkeyListener, _resolve_name


class FakeKeyboard:
    def __init__(self):
        self.hooks = []
        self.unhooked = []
        self.sent = []

    def press(self, name):
        self.sent.append(("press", name))

    def release(self, name):
        self.sent.append(("release", name))

    def hook_key(self, name, callback, suppress=False):
        hook = types.SimpleNamespace(name=name, callback=callback, suppress=suppress)
        self.hooks.append(hook)
        return hook

    def unhook(self, hook):
        self.unhooked.append(hook)


def install_fake_keyboard(monkeypatch):
    fake = FakeKeyboard()
    monkeypatch.setitem(sys.modules, "keyboard", fake)
    return fake


def event(event_type):
    return types.SimpleNamespace(event_type=event_type)


def test_key_names_map_to_keyboard_names():
    assert _resolve_name("caps_lock") == "caps lock"
    assert _resolve_name("alt_r") == "right alt"
    assert _resolve_name("f7") == "f7"


def test_caps_lock_hold_suppresses_and_deduplicates(monkeypatch):
    fake = install_fake_keyboard(monkeypatch)
    calls = []
    listener = WindowsKeyboardHotkeyListener(
        "caps_lock", lambda: calls.append("start"), lambda: calls.append("stop")
    )
    listener.start()
    assert fake.hooks[0].name == "caps lock"
    assert fake.hooks[0].suppress is True
    callback = fake.hooks[0].callback
    callback(event("down"))
    callback(event("down"))
    callback(event("up"))
    callback(event("up"))
    assert calls == ["start", "stop"]
    assert listener.hold_pressed is False


def test_short_caps_lock_press_is_re_emitted(monkeypatch):
    fake = install_fake_keyboard(monkeypatch)
    listener = WindowsKeyboardHotkeyListener("caps_lock", lambda: None, lambda: None)
    listener.start()
    callback = fake.hooks[0].callback
    callback(event("down"))
    time.sleep(0.01)
    callback(event("up"))
    assert fake.sent == [("press", "caps lock"), ("release", "caps lock")]


def test_stop_unhooks_all_registered_keys(monkeypatch):
    fake = install_fake_keyboard(monkeypatch)
    listener = WindowsKeyboardHotkeyListener("caps_lock", lambda: None, lambda: None)
    listener.start()
    listener.stop()
    assert fake.unhooked == fake.hooks
    listener.stop()


def test_toggle_and_hold_are_mutually_exclusive(monkeypatch):
    fake = install_fake_keyboard(monkeypatch)
    calls = []
    listener = WindowsKeyboardHotkeyListener(
        "caps_lock", lambda: calls.append("hold-start"), lambda: calls.append("hold-stop"),
        "f7", lambda: calls.append("toggle-start"), lambda: calls.append("toggle-stop"),
    )
    listener.start()
    hooks = {hook.name: hook.callback for hook in fake.hooks}
    hooks["f7"](event("down"))
    hooks["caps lock"](event("down"))
    hooks["caps lock"](event("up"))
    hooks["f7"](event("up"))
    hooks["f7"](event("down"))
    assert calls == ["toggle-start", "toggle-stop"]


def test_factory_selects_windows_backend(monkeypatch):
    install_fake_keyboard(monkeypatch)
    config = Config(platform="windows", hotkey_backend="keyboard", hotkey="caps_lock")
    listener = create_hotkey_listener(config, lambda: None, lambda: None)
    assert isinstance(listener, WindowsKeyboardHotkeyListener)


def test_factory_rejects_windows_backend_on_other_platform():
    config = Config(platform="x11", hotkey_backend="keyboard")
    with pytest.raises(ConfigError, match="only supported on Windows"):
        create_hotkey_listener(config, lambda: None, lambda: None)
