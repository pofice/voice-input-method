"""Tests for the recording indicator module."""

from voice_input_method.indicator import NullIndicator
from voice_input_method.protocols import RecordingIndicator


class TestNullIndicator:
    """NullIndicator should be a silent no-op that satisfies the Protocol."""

    def test_satisfies_protocol(self):
        assert isinstance(NullIndicator(), RecordingIndicator)

    def test_show_is_noop(self):
        indicator = NullIndicator()
        indicator.show()  # should not raise

    def test_hide_is_noop(self):
        indicator = NullIndicator()
        indicator.hide()  # should not raise

    def test_shutdown_is_noop(self):
        indicator = NullIndicator()
        indicator.shutdown()  # should not raise

    def test_full_lifecycle(self):
        indicator = NullIndicator()
        indicator.show()
        indicator.hide()
        indicator.show()
        indicator.hide()
        indicator.shutdown()


class TestCreateIndicator:
    """Factory function should return NullIndicator on non-macOS platforms."""

    def test_non_macos_returns_null(self):
        from voice_input_method.factory import create_indicator

        indicator = create_indicator("x11")
        assert isinstance(indicator, NullIndicator)

    def test_linux_returns_null(self):
        from voice_input_method.factory import create_indicator

        indicator = create_indicator("wayland")
        assert isinstance(indicator, NullIndicator)

    def test_windows_returns_null(self):
        from voice_input_method.factory import create_indicator

        indicator = create_indicator("windows")
        assert isinstance(indicator, NullIndicator)

    def test_unknown_platform_returns_null(self):
        from voice_input_method.factory import create_indicator

        indicator = create_indicator("unknown")
        assert isinstance(indicator, NullIndicator)
