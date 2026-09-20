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


class _FakeDevice:
    """Stand-in for evdev.InputDevice in hotplug tests — only .path and .close() matter."""

    def __init__(self, path):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


class _FakeSelector:
    """Stand-in for selectors.BaseSelector — records register/unregister calls
    instead of touching real file descriptors."""

    def __init__(self):
        self.registered = []

    def register(self, fileobj, events):
        self.registered.append(fileobj)

    def unregister(self, fileobj):
        if fileobj not in self.registered:
            raise KeyError(fileobj)
        self.registered.remove(fileobj)


class TestHotplug:
    """Keyboards plugged/unplugged after start() — e.g. switching between an
    external keyboard and a laptop's built-in one mid-session."""

    def test_drop_device_unregisters_closes_and_forgets(self, monkeypatch):
        _install_fake_evdev(monkeypatch)
        from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

        listener = EvdevCombinedHotkeyListener("f6", lambda: None, lambda: None)
        dev = _FakeDevice("/dev/input/event3")
        listener._devices = [dev]
        sel = _FakeSelector()
        sel.registered = [dev]

        listener._drop_device(sel, dev)

        assert dev.closed is True
        assert dev not in listener._devices
        assert dev not in sel.registered

    def test_drop_device_is_safe_when_already_unregistered(self, monkeypatch):
        """Must not raise even if the selector never had this device —
        an uncaught error here is exactly what used to kill the whole
        listener thread when a keyboard was unplugged mid-read."""
        _install_fake_evdev(monkeypatch)
        from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

        listener = EvdevCombinedHotkeyListener("f6", lambda: None, lambda: None)
        dev = _FakeDevice("/dev/input/event3")
        listener._devices = [dev]
        sel = _FakeSelector()  # empty — dev was never registered

        listener._drop_device(sel, dev)  # must not raise

        assert dev.closed is True
        assert dev not in listener._devices

    def test_rescan_registers_new_device_and_closes_duplicate_handles(self, monkeypatch):
        """A keyboard plugged in after start() gets picked up; a redundant
        fresh handle to one we already have open gets closed, not kept
        (find_keyboards() opens a new fd for every device on every call,
        including ones already tracked)."""
        _install_fake_evdev(monkeypatch)
        from voice_input_method import hotkey_evdev
        from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

        already_open = _FakeDevice("/dev/input/event3")
        redundant_handle_of_already_open = _FakeDevice("/dev/input/event3")
        newly_plugged_in = _FakeDevice("/dev/input/event7")

        listener = EvdevCombinedHotkeyListener("f6", lambda: None, lambda: None)
        listener._devices = [already_open]
        sel = _FakeSelector()
        sel.registered = [already_open]
        monkeypatch.setattr(
            hotkey_evdev,
            "find_keyboards",
            lambda: [redundant_handle_of_already_open, newly_plugged_in],
        )

        listener._rescan_keyboards(sel)

        assert newly_plugged_in in listener._devices
        assert newly_plugged_in in sel.registered
        assert redundant_handle_of_already_open.closed is True
        assert redundant_handle_of_already_open not in listener._devices

    def test_rescan_with_no_new_devices_is_a_no_op(self, monkeypatch):
        _install_fake_evdev(monkeypatch)
        from voice_input_method import hotkey_evdev
        from voice_input_method.hotkey_evdev import EvdevCombinedHotkeyListener

        already_open = _FakeDevice("/dev/input/event3")
        redundant_handle = _FakeDevice("/dev/input/event3")

        listener = EvdevCombinedHotkeyListener("f6", lambda: None, lambda: None)
        listener._devices = [already_open]
        sel = _FakeSelector()
        sel.registered = [already_open]
        monkeypatch.setattr(hotkey_evdev, "find_keyboards", lambda: [redundant_handle])

        listener._rescan_keyboards(sel)

        assert listener._devices == [already_open]
        assert sel.registered == [already_open]
