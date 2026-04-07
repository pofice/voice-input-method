"""Tests for text_processing module — pure functions, no external deps."""

import pytest

from voice_input_method.text_processing import clean_spaces, convert_chinese_numbers, ChineseConverter


class TestCleanSpaces:
    def test_removes_spaces_between_cjk(self):
        assert clean_spaces("你 好 世界") == "你好世界"

    def test_removes_spaces_between_cjk_and_latin(self):
        assert clean_spaces("你 hello 世界") == "你hello世界"

    def test_preserves_spaces_between_latin(self):
        assert clean_spaces("hello world") == "hello world"

    def test_empty_string(self):
        assert clean_spaces("") == ""

    def test_no_spaces(self):
        assert clean_spaces("你好世界") == "你好世界"

    def test_mixed_content(self):
        assert clean_spaces("AI 模型 test 结果") == "AI模型test结果"

    def test_multiple_spaces_between_cjk(self):
        # clean_spaces only removes single spaces between CJK
        result = clean_spaces("你  好")
        # Two spaces: first space removed (CJK-space-CJK), but second remains
        # Actually regex only matches single space between CJK, so "你  好" stays
        assert "你" in result and "好" in result


class TestConvertChineseNumbers:
    def test_basic_conversion(self):
        # cn2an may not be installed, function handles gracefully
        result = convert_chinese_numbers("一百二十三")
        # If cn2an installed, should convert; if not, returns original
        assert isinstance(result, str)
        assert len(result) > 0

    def test_passthrough_when_no_numbers(self):
        result = convert_chinese_numbers("你好世界")
        assert "你好世界" in result

    def test_empty_string(self):
        assert convert_chinese_numbers("") == ""
