"""Preprocessor for raw Sanskrit text to prepare it for BPE tokenization."""

import argparse
import sys
from collections.abc import Generator, Iterable
from itertools import product
from pathlib import Path
from typing import IO

from common import (
    SPECIAL_TOKENS_SET,
    get_morphemes_from_verse,
    insert_special_tokens,
    normalize_verse,
    sanitize_verse,
    split_verse_by_special_tokens,
)
from sandhi_split import generate_split_candidates


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
    """Preprocess raw Sanskrit text (utf-8 encoded) files."""
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
    processed_count: int = 0

    for verse in read_lines(input_file):
        norm_verse = normalize_verse(verse)
        sanitized_verse = sanitize_verse(norm_verse)
        tokenized_verse = insert_special_tokens(sanitized_verse)
        split_verses = split_verse_by_special_tokens(
            tokenized_verse, SPECIAL_TOKENS_SET
        )

        for v in split_verses:
            morphemes = get_morphemes_from_verse(v, SPECIAL_TOKENS_SET)

            # collect possible splits for each morpheme
            all_split_options = []

            for morpheme in morphemes:
                splits = generate_split_candidates(morpheme)

                # no split possible, we keep original word
                if len(splits) == 0:
                    all_split_options.append([(morpheme, "")])
                else:
                    all_split_options.append(splits)

            # Cartesian product => all combinations of morpheme splits
            for combo in product(*all_split_options):
                new_verse = v

                for original, (left, right) in zip(morphemes, combo):
                    # only replace if a real split
                    if right:
                        new_verse = new_verse.replace(original, f"{left} {right}", 1)

                lines.append(new_verse.strip())

        processed_count += 1

    print(f'Processed {len(lines)} lines from "{input_file}"')

    write_lines(output_file, lines)
    print(f'Wrote {len(lines)} verse\'s to "{output_file}"')


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nPreprocessing completed in {end_time - start_time:.3f} seconds")
