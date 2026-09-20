"""Tests for hotkey module — verifies the HotkeyListener interface without pynput."""

from voice_input_method.hotkey import (
    CombinedHotkeyListener,
    HotkeyListener,
    _any_release_only,
    _names_of,
    _resolve_key,
    _resolve_keys,
)


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

    def test_accepts_a_list_of_keys(self):
        """A HotkeyListener can be bound to several physical keys at once."""
        listener = HotkeyListener(
            hotkey=["scroll_lock", "f6"],
            on_press=lambda: None,
            on_release=lambda: None,
        )
        assert listener is not None


class TestMultiKeyResolution:
    """Binding the same action to several physical keys — e.g. `scroll_lock`
    on a keyboard that has it, plus `f6` for when only a laptop's built-in
    keyboard (which usually lacks Scroll Lock) is plugged in."""

    def test_names_of_normalizes_string_and_list(self):
        assert _names_of("f6") == ["f6"]
        assert _names_of(["f6", "scroll_lock"]) == ["f6", "scroll_lock"]

    def test_resolve_keys_single_name_matches_resolve_key(self):
        assert _resolve_keys("f6") == {_resolve_key("f6")}

    def test_resolve_keys_list_resolves_every_name(self):
        targets = _resolve_keys(["f6", "scroll_lock"])
        assert targets == {_resolve_key("f6"), _resolve_key("scroll_lock")}

    def test_any_release_only_true_for_single_release_only_key(self):
        assert _any_release_only("fn") is True

    def test_any_release_only_true_if_any_key_in_list_is_release_only(self):
        assert _any_release_only(["f6", "fn"]) is True

    def test_any_release_only_false_when_no_key_is_release_only(self):
        assert _any_release_only(["scroll_lock", "f6"]) is False

    def test_combined_listener_accepts_lists_for_both_hotkeys(self):
        """CombinedHotkeyListener (what factory.py actually wires up) accepts
        a list for hold_hotkey and toggle_hotkey independently."""
        listener = CombinedHotkeyListener(
            hold_hotkey=["scroll_lock", "f6"],
            hold_on_press=lambda: None,
            hold_on_release=lambda: None,
            toggle_hotkey=["alt", "f7"],
            toggle_on_start=lambda: None,
            toggle_on_stop=lambda: None,
        )
        assert listener.hold_pressed is False
        assert listener.toggle_recording is False
