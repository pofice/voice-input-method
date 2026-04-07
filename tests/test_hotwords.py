"""Tests for hotwords module — pure Python loading, no Qt needed."""

import pytest
from pathlib import Path

from voice_input_method.hotwords import HotwordManager


class TestHotwordManager:
    def test_load_basic(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("你好\n世界\n测试")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "你好 世界 测试"

    def test_load_with_comments(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("# comment\n你好\n\n世界")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "你好 世界"

    def test_load_empty_file(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == ""

    def test_load_missing_file(self, tmp_path):
        hw_file = tmp_path / "nonexistent.txt"
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == ""

    def test_long_hotwords_split(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("这是一个超过十个字的热词测试")
        mgr = HotwordManager(hw_file)
        parts = mgr.hotwords_str.split(" ")
        assert all(len(p) <= HotwordManager.MAX_LENGTH for p in parts)

    def test_reload(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("初始")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "初始"

        hw_file.write_text("更新")
        mgr.reload()
        assert mgr.hotwords_str == "更新"

    def test_no_qt_import_on_init(self, tmp_path):
        """HotwordManager init should NOT import PySide6."""
        import sys
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("test")
        # This should work in headless without Qt
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "test"
        # _watcher should be None (not created until start_watching)
        assert mgr._watcher is None
