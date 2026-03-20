"""Text post-processing: space cleanup, number conversion, traditional/simplified Chinese."""

import re
from pathlib import Path


def clean_spaces(text: str) -> str:
    """Remove unnecessary spaces between CJK characters and between CJK and Latin."""
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[a-zA-Z])", "", text)
    text = re.sub(r"(?<=[a-zA-Z]) (?=[\u4e00-\u9fff])", "", text)
    return text


def convert_chinese_numbers(text: str) -> str:
    """Convert Chinese number words to Arabic numerals."""
    try:
        import cn2an
        return cn2an.transform(text, "cn2an")
    except ImportError:
        print("Warning: cn2an not installed, skipping number conversion")
        return text
    except Exception as e:
        print(f"Error converting Chinese numbers: {e}")
        return text


class ChineseConverter:
    """Traditional/Simplified Chinese converter."""

    def __init__(self, library_path: Path | None = None):
        from opencc import OpenCC
        self.cc_s2t = OpenCC("s2twp")
        self.cc_t2s = OpenCC("t2s")
        self.traditional_chars: set[str] = set()
        if library_path and library_path.exists():
            self.traditional_chars = set(library_path.read_text(encoding="utf-8"))

    def is_traditional(self, text: str) -> bool:
        return any(char in self.traditional_chars for char in text)

    def convert(self, text: str) -> str:
        """Auto-detect direction and convert."""
        if self.is_traditional(text):
            return self.cc_t2s.convert(text)
        return self.cc_s2t.convert(text)
