"""Text post-processing: space cleanup, number conversion, traditional/simplified Chinese."""

import re
from pathlib import Path


# Chinese numeral characters used to detect number-bearing tokens.
_CHINESE_NUMBER_CHARS = frozenset("零一二两三四五六七八九十百千万亿点")

# Phrases that contain Chinese number characters but should NEVER be
# converted to digits — they are fixed expressions where the "number"
# has lost its quantitative meaning.  Sorted by length (longest-first
# applied at runtime) so multi-char phrases are checked before single
# characters.
#
# Add new entries here when users report false-positives.
_PROTECTED_PHRASES = frozenset({
    # 一-prefixed
    "一定", "一起", "一样", "一直", "一些", "一下", "一点", "一边",
    "一切", "一面", "一时", "一旦", "一种", "一阵", "一共", "一般",
    "一向", "一致", "一连", "一律", "一同", "一道", "一行", "一会儿",
    "一会", "一辈子", "一辈",
    # 二/两-prefixed
    "二维", "二者", "二话", "两者", "两边", "两旁",
    # 三/四/五等
    "三角", "四方", "五官", "六合", "七夕", "八卦", "九霄", "十足",
    # ...-prefixed (顺序词)
    "第一", "第二", "第三", "第一名", "第二名", "第三名", "唯一",
    "万一", "统一", "归一", "合一", "之一", "其一", "百分之一",
    # 数字+方向词（非量化）
    "百年", "千年", "万年", "千古", "百姓",
})


def clean_spaces(text: str) -> str:
    """Remove unnecessary spaces between CJK characters and between CJK and Latin."""
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[a-zA-Z])", "", text)
    text = re.sub(r"(?<=[a-zA-Z]) (?=[\u4e00-\u9fff])", "", text)
    return text


def _has_chinese_number(token: str) -> bool:
    """Check if a token contains any Chinese numeral character."""
    return any(c in _CHINESE_NUMBER_CHARS for c in token)


def _convert_token(token: str, cn2an_module) -> str:
    """Convert a single jieba token containing Chinese numerals.

    If the token equals a protected phrase, it is returned unchanged.
    If the token *contains* one or more protected phrases as substrings
    (e.g. jieba merged "统一" into the compound "统一标准"), each protected
    phrase is masked with a placeholder before cn2an runs and restored
    afterwards, so the protected phrase's number characters are never
    converted.
    """
    if token in _PROTECTED_PHRASES:
        return token

    # Mask any protected phrase that appears as a substring.
    # Longest-first to avoid partial overlaps.
    masked = token
    placeholders: dict[str, str] = {}
    for i, phrase in enumerate(sorted(_PROTECTED_PHRASES, key=len, reverse=True)):
        if phrase in masked:
            placeholder = f"\x00{i}\x00"
            placeholders[placeholder] = phrase
            masked = masked.replace(phrase, placeholder)

    try:
        converted = cn2an_module.transform(masked, "cn2an")
    except Exception:
        # cn2an raises on tokens it cannot parse — keep original.
        return token

    # Restore protected phrases.
    for placeholder, phrase in placeholders.items():
        converted = converted.replace(placeholder, phrase)
    return converted


def _contains_protected_substring(token: str) -> bool:
    """True if any protected phrase appears as a substring of *token*."""
    return any(p in token for p in _PROTECTED_PHRASES)


def convert_chinese_numbers(text: str) -> str:
    """Convert Chinese number words in *text* to Arabic numerals.

    Algorithm:
    1. Tokenize with jieba (HMM=False for stable segmentation).
    2. Walk tokens. Adjacent "plain number tokens" (containing only number
       characters and not protected) are buffered and converted as one
       merged string — this handles "一百二十" + "三" → "一百二十三" → "123".
    3. A token equal to a protected phrase passes through unchanged.
    4. A token containing a protected phrase as substring (e.g. jieba
       merged "统一" into "统一标准") flushes the buffer, then masks the
       protected phrase before running cn2an, then restores.
    5. Tokens with no number characters flush the buffer and pass through.

    Examples::

        "一定要去公园"          → "一定要去公园"     (一定 protected)
        "三个苹果二十年"        → "3个苹果20年"
        "我有一个想法"          → "我有1个想法"
        "第一名得到三千块"      → "第一名得到3000块"
        "统一标准"              → "统一标准"        (jieba 合并的复合词)
        "一百二十三"            → "123"             (跨 token 合并)

    Falls back to passthrough if jieba or cn2an are not installed.
    """
    if not text:
        return text

    try:
        import cn2an
        import jieba
    except ImportError:
        return text

    result_parts: list[str] = []
    buffer: list[str] = []  # accumulates contiguous plain number tokens

    def flush_buffer() -> None:
        """Convert the buffered number tokens as a single merged string."""
        if not buffer:
            return
        merged = "".join(buffer)
        try:
            result_parts.append(cn2an.transform(merged, "cn2an"))
        except Exception:
            result_parts.append(merged)
        buffer.clear()

    for token in jieba.cut(text, HMM=False):
        if not _has_chinese_number(token):
            flush_buffer()
            result_parts.append(token)
            continue

        if token in _PROTECTED_PHRASES:
            flush_buffer()
            result_parts.append(token)
            continue

        if _contains_protected_substring(token):
            # Protected phrase is embedded in a longer compound — handle in
            # isolation so the buffer's accumulated context isn't polluted.
            flush_buffer()
            result_parts.append(_convert_token(token, cn2an))
            continue

        # Plain number-bearing token — accumulate.
        buffer.append(token)

    flush_buffer()
    return "".join(result_parts)


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
