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


def test_resolve_codes_accepts_single_name(monkeypatch):
    _install_fake_evdev(monkeypatch)
    from voice_input_method.hotkey_evdev import _resolve_code, _resolve_codes

    assert _resolve_codes("f6") == {_resolve_code("f6")}


def test_resolve_codes_accepts_list_of_names(monkeypatch):
    """Binding the same action to several physical keys — e.g. `scroll_lock`
    on a keyboard that has it, plus `f6` for a laptop keyboard that doesn't."""
    _install_fake_evdev(monkeypatch)
    from voice_input_method.hotkey_evdev import _resolve_code, _resolve_codes

    codes = _resolve_codes(["scroll_lock", "f6"])
    assert codes == {_resolve_code("scroll_lock"), _resolve_code("f6")}


def test_combined_listener_accepts_lists_for_both_hotkeys(monkeypatch):
    _install_fake_evdev(monkeypatch)
    from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

    listener = EvdevCombinedHotkeyListener(
        hold_hotkey=["scroll_lock", "f6"],
        hold_on_press=lambda: None,
        hold_on_release=lambda: None,
        toggle_hotkey=["alt", "f7"],
        toggle_on_start=lambda: None,
        toggle_on_stop=lambda: None,
    )
    assert listener.hold_pressed is False
    assert listener.toggle_recording is False


def test_handle_dispatches_from_either_bound_key(monkeypatch):
    """A press on *any* key in the hold list must trigger hold_on_press —
    this is the actual behavior the feature is for, not just construction."""
    fake = _install_fake_evdev(monkeypatch)
    from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener, _resolve_codes

    presses = []
    listener = EvdevCombinedHotkeyListener(
        hold_hotkey=["scroll_lock", "f6"],
        hold_on_press=lambda: presses.append("press"),
        hold_on_release=lambda: presses.append("release"),
    )
    hold_codes = _resolve_codes(["scroll_lock", "f6"])
    toggle_codes = set()

    press_event = types.SimpleNamespace(type=fake.ecodes.EV_KEY, code=fake.ecodes.KEY_F6, value=1)
    release_event = types.SimpleNamespace(type=fake.ecodes.EV_KEY, code=fake.ecodes.KEY_F6, value=0)
    listener._handle(press_event, hold_codes, toggle_codes)
    assert presses == ["press"]
    assert listener.hold_pressed is True
    listener._handle(release_event, hold_codes, toggle_codes)
    assert presses == ["press", "release"]
    assert listener.hold_pressed is False

    # Now trigger via the *other* bound key (scroll_lock) — same action.
    presses.clear()
    press_event2 = types.SimpleNamespace(
        type=fake.ecodes.EV_KEY, code=fake.ecodes.KEY_SCROLLLOCK, value=1
    )
    listener._handle(press_event2, hold_codes, toggle_codes)
    assert presses == ["press"]
    assert listener.hold_pressed is True
