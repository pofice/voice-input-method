"""Tests for hotwords module — pure Python loading, no Qt needed."""


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
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("test")
        # This should work in headless without Qt
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "test"
        # _watcher should be None (not created until start_watching)
        assert mgr._watcher is None

    def test_corrections_basic(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("Cloud Code -> Claude Code\nGoole -> Google\n")
        mgr = HotwordManager(hw_file)
        assert mgr.corrections == {
            "Cloud Code": "Claude Code",
            "Goole": "Google",
        }
        # Corrections are not in hotwords list
        assert mgr.hotwords_str == ""

    def test_corrections_mixed_with_hotwords(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("Claude Code\nCloud Code -> Claude Code\n遍历\n")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "Claude Code 遍历"
        assert mgr.corrections == {"Cloud Code": "Claude Code"}

    def test_hotwords_csv_format(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("Claude Code\n遍历\n数组\n")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_csv == "Claude Code,遍历,数组"

    def test_hotwords_csv_empty(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_csv == ""

    def test_reload_updates_corrections(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("old -> new")
        mgr = HotwordManager(hw_file)
        assert mgr.corrections == {"old": "new"}

        hw_file.write_text("foo -> bar")
        mgr.reload()
        assert mgr.corrections == {"foo": "bar"}

    def test_on_reload_callback(self, tmp_path):
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("test")
        called = []
        HotwordManager(hw_file, on_reload=lambda: called.append(True))
        # on_reload is NOT called by __init__.reload
        # It's called by _on_file_changed
        assert called == []

    def test_file_error_keeps_empty(self, tmp_path):
        """If file can't be read (permission error etc), state stays empty."""
        hw_file = tmp_path / "hotwords.txt"
        hw_file.write_text("初始值")
        mgr = HotwordManager(hw_file)
        assert mgr.hotwords_str == "初始值"
        # Delete the file and reload
        hw_file.unlink()
        mgr.reload()
        assert mgr.hotwords_str == ""
        assert mgr.corrections == {}

    def test_long_hotword_split_precisely(self, tmp_path):
        """Hotwords longer than MAX_LENGTH are split into chunks."""
        hw_file = tmp_path / "hotwords.txt"
        # Create a hotword exactly 2x MAX_LENGTH + 5
        long_word = "a" * (HotwordManager.MAX_LENGTH * 2 + 5)
        hw_file.write_text(long_word)
        mgr = HotwordManager(hw_file)
        parts = mgr.hotwords_str.split(" ")
        assert len(parts) == 3  # two full chunks + remainder
        assert all(len(p) <= HotwordManager.MAX_LENGTH for p in parts)
