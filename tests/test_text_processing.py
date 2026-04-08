"""Tests for text_processing module — pure functions, no external deps."""

import pytest

from voice_input_method.text_processing import (
    clean_spaces,
    strip_trailing_punctuation,
    ChineseConverter,
)


class TestStripTrailingPunctuation:
    def test_strip_chinese_period(self):
        assert strip_trailing_punctuation("你好。") == "你好"

    def test_strip_chinese_exclamation(self):
        assert strip_trailing_punctuation("你好！") == "你好"

    def test_strip_chinese_question(self):
        assert strip_trailing_punctuation("你好吗？") == "你好吗"

    def test_strip_english_period(self):
        assert strip_trailing_punctuation("hello.") == "hello"

    def test_strip_english_exclamation(self):
        assert strip_trailing_punctuation("hello!") == "hello"

    def test_strip_multiple_trailing(self):
        assert strip_trailing_punctuation("你好！？。") == "你好"

    def test_preserves_internal_punctuation(self):
        assert strip_trailing_punctuation("你好，世界。") == "你好，世界"

    def test_strip_trailing_whitespace_too(self):
        assert strip_trailing_punctuation("你好。  ") == "你好"

    def test_empty_string(self):
        assert strip_trailing_punctuation("") == ""

    def test_only_punctuation(self):
        assert strip_trailing_punctuation("。。。") == ""

    def test_no_trailing_punct(self):
        assert strip_trailing_punctuation("你好") == "你好"

    def test_strip_ellipsis(self):
        assert strip_trailing_punctuation("你好…") == "你好"

    def test_strip_comma(self):
        assert strip_trailing_punctuation("你好，") == "你好"


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
        result = clean_spaces("你  好")
        assert "你" in result and "好" in result

    def test_merge_single_letters(self):
        assert clean_spaces("A I") == "AI"
        assert clean_spaces("K F C") == "KFC"
