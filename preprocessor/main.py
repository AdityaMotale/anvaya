"""Preprocessor to process raw Sanskrit text to prepare for BPE tokenization."""

import argparse
import sys
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import IO

from common import (
    SPECIAL_TOKENS_SET,
    insert_special_tokens,
    normalize_verse,
    sanitize_verse,
    split_verse_by_special_tokens,
)


def open_file(path: str, mode: str = "r") -> IO:
    """Open a file with the given path and mode.

    Args:
        path: The file path
        mode: The mode in which to open the file (default is "r").

    Returns:
        A file object opened with the specified mode.

    """
    return Path(path).open(mode, encoding="utf-8")


def read_lines(path: str) -> Generator[str, None, None]:
    """Yield lines from a UTF-8 text file lazily, one at a time.

    Args:
        path: The file path

    Yields:
        A line from the file

    """
    file = open_file(path)

    for line in file:
        yield line.rstrip("\n")


def write_lines(path: str, lines: Iterable[str]) -> None:
    """Write lines to a UTF-8 text file.

    Args:
        path: The file path
        lines: An iterable of strings to write to the file

    """
    file = open_file(path, "w")

    for line in lines:
        file.write(line + "\n")


def main() -> None:
    """Preprocess raw Sanskrit text files (utf-8) encoded."""
    parser = argparse.ArgumentParser(
        description="Pre-process raw Sanskrit text for tokenization process"
    )
    parser.add_argument("input", nargs="?", help="Path to input file (UTF-8)")
    parser.add_argument("output", nargs="?", help="Path to output file (UTF-8)")

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    # exit(1) if invalid args
    if input_file is None or output_file is None:
        parser.print_help()
        sys.exit(1)

    lines: Iterable[str] = []

    for verse in read_lines(input_file):
        norm_verse = normalize_verse(verse)
        sanitized_verse = sanitize_verse(norm_verse)
        tokenized_verse = insert_special_tokens(sanitized_verse)
        split_verse = split_verse_by_special_tokens(tokenized_verse, SPECIAL_TOKENS_SET)

        lines.extend(split_verse)

    write_lines(output_file, lines)


if __name__ == "__main__":
    main()


# - Read line-by-line
# - Normalize text using NFC
# - Validate sanskrit bytes
# - Special tokens (| and ||)
# - Apply Sandhi-Splitting rules
# - Split lines by special tokens
# - Write back to txt file
