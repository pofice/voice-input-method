"""Unit tests for the evdev hotkey backend and its factory wiring.

No real /dev/input access — evdev is stubbed so these run in CI.
"""

import sys
import types

import pytest

from voice_input_method.config import Config
from voice_input_method.factory import ConfigError, create_hotkey_listener


def _install_fake_evdev(monkeypatch):
    """Provide a minimal fake `evdev` module so imports succeed headless."""
    ecodes = types.SimpleNamespace(
        EV_KEY=1, KEY_A=30, KEY_Z=44, KEY_F6=64, KEY_F7=65,
        KEY_SCROLLLOCK=70, KEY_LEFTALT=56,
    )
    fake = types.ModuleType("evdev")
    fake.ecodes = ecodes
    fake.list_devices = lambda: []
    fake.InputDevice = object
    monkeypatch.setitem(sys.modules, "evdev", fake)
    return fake


def test_auto_backend_picks_evdev_on_wayland(monkeypatch):
    _install_fake_evdev(monkeypatch)
    cfg = Config(platform="wayland", hotkey_backend="auto")
    listener = create_hotkey_listener(cfg, lambda: None, lambda: None)
    assert type(listener).__name__ == "EvdevCombinedHotkeyListener"


def test_auto_backend_picks_pynput_on_x11():
    cfg = Config(platform="x11", hotkey_backend="auto")
    listener = create_hotkey_listener(cfg, lambda: None, lambda: None)
    assert type(listener).__name__ == "CombinedHotkeyListener"


def test_explicit_pynput_on_wayland():
    cfg = Config(platform="wayland", hotkey_backend="pynput")
    listener = create_hotkey_listener(cfg, lambda: None, lambda: None)
    assert type(listener).__name__ == "CombinedHotkeyListener"


def test_invalid_backend_raises_config_error():
    cfg = Config(hotkey_backend="nope")
    with pytest.raises(ConfigError, match="hotkey_backend"):
        create_hotkey_listener(cfg, lambda: None, lambda: None)


def test_start_without_keyboard_raises_permission_error(monkeypatch):
    _install_fake_evdev(monkeypatch)  # list_devices() returns []
    from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

    listener = EvdevCombinedHotkeyListener("f6", lambda: None, lambda: None)
    with pytest.raises(PermissionError, match="input"):
        listener.start()


def test_key_name_resolution(monkeypatch):
    _install_fake_evdev(monkeypatch)
    from voice_input_method.hotkey_evdev import _resolve_code

    assert _resolve_code("f6") == 64
    assert _resolve_code("scroll_lock") == 70
    assert _resolve_code("alt") == 56
    assert _resolve_code("unknown_key") == 64  # falls back to F6
