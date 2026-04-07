"""Text post-processing: space cleanup, letter merging, traditional/simplified Chinese."""

import re
from pathlib import Path


def _merge_single_letters(text: str) -> str:
    """Merge sequences of space-separated single letters: 'A I' → 'AI', 'K F C' → 'KFC'.
    Preserves multi-char words like 'super powers'.
    """
    def _replace(m: re.Match) -> str:
        return m.group(0).replace(" ", "")
    # Match 2+ single letters separated by spaces (e.g. "A I", "K F C")
    return re.sub(r"(?<![a-zA-Z])([a-zA-Z] ){1,}[a-zA-Z](?![a-zA-Z])", _replace, text)


def clean_spaces(text: str) -> str:
    """Remove unnecessary spaces between CJK characters and between CJK and Latin.
    Also merge isolated single letters like 'A I' → 'AI', 'K F C' → 'KFC'.
    """
    # First merge single letters (before CJK-space removal changes boundaries)
    text = _merge_single_letters(text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[a-zA-Z])", "", text)
    text = re.sub(r"(?<=[a-zA-Z]) (?=[\u4e00-\u9fff])", "", text)
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
