#!/usr/bin/env python3
"""Extract hotwords from rime-ice user dictionary for use with voice-input-method.

Usage:
    python tools/rime_ice2hotwords.py /path/to/rime_ice.userdb.txt -o hotwords.txt
"""

import argparse
import re


def extract_chinese_words(file_path: str, keep_single_char: bool = False) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    words = re.findall(r"\t([\u4e00-\u9fff]+)", content)

    if not keep_single_char:
        words = [w for w in words if len(w) > 1]

    return words


def main():
    parser = argparse.ArgumentParser(description="Extract hotwords from rime-ice user dictionary")
    parser.add_argument("input", help="Path to rime_ice.userdb.txt")
    parser.add_argument("-o", "--output", default="hotwords.txt", help="Output hotwords file")
    parser.add_argument("--keep-single", action="store_true", help="Keep single-character words")
    args = parser.parse_args()

    words = extract_chinese_words(args.input, keep_single_char=args.keep_single)

    with open(args.output, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

    print(f"Extracted {len(words)} hotwords to {args.output}")


if __name__ == "__main__":
    main()
