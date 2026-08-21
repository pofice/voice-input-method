"""Tests for the Wayland platform backend (wl-copy + ydotool)."""

import subprocess
from unittest.mock import patch

import pytest

from voice_input_method.platform.wayland import WaylandBackend


def _fake_runner(clipboard: dict, *, copy_fails=False, stale_reads=0):
    """Fake subprocess.run: wl-copy writes the dict, wl-paste reads it.

    stale_reads simulates the compositor taking a few polls to hand the
    selection over, which is exactly the race the read-back guards against.
    """
    state = {"pending": stale_reads}

    def run(cmd, **kwargs):
        if cmd[0] == "wl-copy":
            if copy_fails:
                raise subprocess.CalledProcessError(1, cmd)
            clipboard["next"] = kwargs["input"].decode("utf-8")
            if state["pending"] == 0:
                clipboard["current"] = clipboard["next"]
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "wl-paste":
            if state["pending"] > 0:
                state["pending"] -= 1
                if state["pending"] == 0:
                    clipboard["current"] = clipboard["next"]
            if "current" not in clipboard:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(
                cmd, 0, stdout=clipboard["current"].encode("utf-8")
            )
        clipboard.setdefault("keys", []).append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    return run


def test_paste_writes_clipboard_then_sends_ctrl_v():
    clipboard = {}
    with patch("subprocess.run", side_effect=_fake_runner(clipboard)):
        WaylandBackend().paste_text("你好世界")
    assert clipboard["current"] == "你好世界"
    assert clipboard["keys"] == [["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]]


def test_paste_waits_for_slow_clipboard_handover():
    """The keystroke must not fire until the read-back matches."""
    clipboard = {"current": "上一次的转录"}
    with patch("subprocess.run", side_effect=_fake_runner(clipboard, stale_reads=3)):
        WaylandBackend().paste_text("这一次的转录")
    assert clipboard["current"] == "这一次的转录"
    assert len(clipboard.get("keys", [])) == 1


def test_no_keystroke_when_clipboard_write_fails():
    """Pasting a stale clipboard is worse than pasting nothing."""
    clipboard = {"current": "上一次的转录"}
    with patch("subprocess.run", side_effect=_fake_runner(clipboard, copy_fails=True)):
        WaylandBackend().paste_text("这一次的转录")
    assert clipboard["current"] == "上一次的转录"
    assert clipboard.get("keys", []) == []


def test_no_keystroke_when_handover_never_completes():
    clipboard = {"current": "上一次的转录"}
    with (
        patch("subprocess.run", side_effect=_fake_runner(clipboard, stale_reads=10**6)),
        patch("voice_input_method.platform.wayland._CLIPBOARD_TIMEOUT", 0.05),
    ):
        WaylandBackend().paste_text("这一次的转录")
    assert clipboard.get("keys", []) == []


def test_check_permissions_reports_missing_wl_clipboard():
    with patch("shutil.which", return_value=None):
        missing = WaylandBackend().check_permissions()
    assert any("wl-copy" in m for m in missing)
    assert any("wl-paste" in m for m in missing)


@pytest.mark.parametrize("tool", ["wl-copy", "wl-paste"])
def test_check_permissions_quiet_when_tools_present(tool):
    with patch("shutil.which", return_value=f"/usr/bin/{tool}"):
        missing = WaylandBackend().check_permissions()
    assert not any("wl-clipboard" in m for m in missing)
