"""Tests for text_processing module — pure functions, no external deps."""

from unittest.mock import MagicMock, patch

from voice_input_method.text_processing import (
    _merge_single_letters,
    apply_corrections,
    clean_spaces,
    strip_trailing_punctuation,
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


class TestMergeSingleLetters:
    def test_two_letters(self):
        assert _merge_single_letters("A I") == "AI"

    def test_three_letters(self):
        assert _merge_single_letters("K F C") == "KFC"

    def test_preserves_words(self):
        assert _merge_single_letters("super powers") == "super powers"

    def test_mixed_letters_and_words(self):
        result = _merge_single_letters("hello A I world")
        assert "AI" in result

    def test_no_single_letters(self):
        assert _merge_single_letters("hello world") == "hello world"

    def test_empty_string(self):
        assert _merge_single_letters("") == ""


class TestApplyCorrections:
    def test_basic_correction(self):
        corrections = {"Cloud Code": "Claude Code"}
        assert apply_corrections("用Cloud Code写代码", corrections) == "用Claude Code写代码"

    def test_case_insensitive(self):
        corrections = {"cloud code": "Claude Code"}
        assert apply_corrections("用CLOUD CODE写代码", corrections) == "用Claude Code写代码"

    def test_multiple_corrections(self):
        corrections = {"Cloud Code": "Claude Code", "Goole": "Google"}
        text = "用Cloud Code搜Goole"
        result = apply_corrections(text, corrections)
        assert result == "用Claude Code搜Google"

    def test_empty_corrections(self):
        assert apply_corrections("hello", {}) == "hello"

    def test_no_match(self):
        corrections = {"xyz": "abc"}
        assert apply_corrections("hello world", corrections) == "hello world"

    def test_multiple_occurrences(self):
        corrections = {"AI": "人工智能"}
        assert apply_corrections("AI和AI", corrections) == "人工智能和人工智能"

    def test_special_regex_chars(self):
        corrections = {"c++": "C Plus Plus"}
        assert apply_corrections("学c++", corrections) == "学C Plus Plus"


class TestChineseConverter:
    def test_converter_init_with_mock_opencc(self):
        with patch.dict("sys.modules", {"opencc": MagicMock()}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter()
            assert hasattr(cc, "cc_s2t")
            assert hasattr(cc, "cc_t2s")

    def test_converter_with_library_file(self, tmp_path):
        lib_file = tmp_path / "trad_chars.txt"
        lib_file.write_text("繁體字", encoding="utf-8")
        with patch.dict("sys.modules", {"opencc": MagicMock()}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter(library_path=lib_file)
            assert len(cc.traditional_chars) > 0

    def test_is_traditional_true(self):
        with patch.dict("sys.modules", {"opencc": MagicMock()}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter()
            cc.traditional_chars = {"繁", "體"}
            assert cc.is_traditional("這是繁體") is True

    def test_is_traditional_false(self):
        with patch.dict("sys.modules", {"opencc": MagicMock()}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter()
            cc.traditional_chars = {"繁", "體"}
            assert cc.is_traditional("这是简体") is False

    def test_convert_simplified_to_traditional(self):
        mock_opencc = MagicMock()
        mock_s2t = MagicMock()
        mock_t2s = MagicMock()
        mock_opencc.OpenCC.side_effect = [mock_s2t, mock_t2s]
        mock_s2t.convert.return_value = "轉換結果"

        with patch.dict("sys.modules", {"opencc": mock_opencc}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter()
            cc.traditional_chars = set()
            cc.cc_s2t = mock_s2t
            cc.cc_t2s = mock_t2s
            cc.convert("转换结果")
            mock_s2t.convert.assert_called_once_with("转换结果")

    def test_convert_traditional_to_simplified(self):
        mock_opencc = MagicMock()
        mock_s2t = MagicMock()
        mock_t2s = MagicMock()
        mock_opencc.OpenCC.side_effect = [mock_s2t, mock_t2s]
        mock_t2s.convert.return_value = "转换结果"

        with patch.dict("sys.modules", {"opencc": mock_opencc}):
            from voice_input_method.text_processing import ChineseConverter
            cc = ChineseConverter()
            cc.traditional_chars = {"繁", "轉"}
            cc.cc_s2t = mock_s2t
            cc.cc_t2s = mock_t2s
            cc.convert("轉換結果")
            mock_t2s.convert.assert_called_once_with("轉換結果")
