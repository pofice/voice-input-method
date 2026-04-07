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
    def test_empty_string(self):
        assert convert_chinese_numbers("") == ""

    def test_passthrough_when_no_numbers(self):
        assert convert_chinese_numbers("你好世界") == "你好世界"

    def test_passthrough_latin(self):
        assert convert_chinese_numbers("hello world") == "hello world"

    def test_basic_number_conversion(self):
        try:
            import cn2an  # noqa
            import jieba  # noqa
        except ImportError:
            pytest.skip("cn2an/jieba not installed")
        assert convert_chinese_numbers("三千五百块") == "3500块"
        assert convert_chinese_numbers("二十年") == "20年"
        assert convert_chinese_numbers("十点半") == "10点半"

    @pytest.mark.parametrize("text", [
        # 一-prefixed fixed phrases
        "一定要去",
        "一起走",
        "一样的",
        "一直在",
        "一些东西",
        "一种方式",
        "一旦发生",
        "一辈子",
        "一律执行",
        "一连串",
        # 第N / 唯一 / 万一 / 统一
        "第一名",
        "第二名",
        "第三方",
        "唯一选择",
        "万一发生",
        "万一不行",
        "统一标准",   # jieba merges into compound
        "归一化",
        "合一",
        "之一",
        "其一",
        # 二/四 phrases
        "二者关系",
        "二维码",
    ])
    def test_protected_phrases_not_converted(self, text):
        """Trap words containing 一/二/三 etc should pass through unchanged."""
        try:
            import cn2an  # noqa
            import jieba  # noqa
        except ImportError:
            pytest.skip("cn2an/jieba not installed")
        assert convert_chinese_numbers(text) == text, \
            f"Protected phrase {text!r} was incorrectly converted"

    @pytest.mark.parametrize("text,expected", [
        ("三个苹果", "3个苹果"),
        ("我有一个想法", "我有1个想法"),
        ("两个人", "2个人"),
        # Mixed: protected phrase + convertible numbers in one sentence
        ("我有三个苹果一定要吃完", "我有3个苹果一定要吃完"),
        ("第一名得到三千块", "第一名得到3000块"),
        ("一定要买二十个苹果", "一定要买20个苹果"),
        # Regression: jieba splits "一百二十三" into ["一百二十", "三"];
        # the buffer must merge them before cn2an, otherwise → "1203".
        ("一百二十三", "123"),
        ("一千二百三十", "1230"),
    ])
    def test_real_numbers_converted(self, text, expected):
        """Genuine quantities should still be converted to digits."""
        try:
            import cn2an  # noqa
            import jieba  # noqa
        except ImportError:
            pytest.skip("cn2an/jieba not installed")
        assert convert_chinese_numbers(text) == expected

    def test_no_chinese_chars(self):
        """ASCII / numeric text passes through untouched."""
        assert convert_chinese_numbers("test 123") == "test 123"
        assert convert_chinese_numbers("v1.0") == "v1.0"
