# ruff: noqa

"""Train a BPE Tokenizer from preprocessed Sanskrit corpus."""

import argparse
import sys
from pathlib import Path
from typing import IO, Iterable
from collections import Counter, defaultdict

import regex

EOM_MARKER: str = "</m>"
NUM_MERGES: int = 10_000
MIN_FREQ: int = 1

DANDA_TOKEN = "<DANDA>"
DOUBLE_DANDA_TOKEN = "<DANDA2>"

SPECIAL_TOKENS_SET: list[str] = [DANDA_TOKEN, DOUBLE_DANDA_TOKEN]


def open_file(path: str, mode: str = "r") -> IO:
    """Open a file with the given path and mode.

    Args:
        path: The file path
        mode: The mode in which to open the file (default is "r").

    Returns:
        A file object opened with the specified mode.

    """
    return Path(path).open(mode, encoding="utf-8")


def read_file(path: str) -> list[str]:
    """Read entire input file into a large list of strings.

    Args:
        path: The file path

    Returns:
        List of lines read from the input file

    """
    file = open_file(path)

    return file.readlines()


def grapheme_split(m: str) -> list[str]:
    """Split Sanskrit morpheme into Unicode grapheme clusters.

    Args:
        m: Input morpheme

    Returns:
        List of Unicode symbols

    """
    return regex.findall(r"\X", m)


def main() -> None:
    """Train a BPE Tokenizer from raw Sanskrit text."""
    parser = argparse.ArgumentParser(
        description="Train a BPE Tokenizer from raw Sanskrit text."
    )
    parser.add_argument(
        "input", nargs="?", help="Path to preprocessed input file (UTF-8)"
    )

    args = parser.parse_args()
    input_file = args.input

    # exit(1) if invalid args
    if input_file is None:
        parser.print_help()
        sys.exit(1)

    lines = read_file(input_file)

    graphemes_list = []
    vocab = defaultdict(int)

    for line in lines[:10]:
        for word in line.strip().split(" "):
            _list = []
            word = word.strip()

            if word in SPECIAL_TOKENS_SET:
                _list.append(word)
            else:
                _list.extend(grapheme_split(word))

            _list.append(EOM_MARKER)
            graphemes_list.append(_list)

    for v in graphemes_list:
        for i in range(len(v) - 1):
            pair = (v[i], v[i + 1])

            vocab[pair] += 1

    sorted_vocab = dict(
        sorted(
            vocab.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    for k, v in sorted_vocab.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nBPE training completed in {end_time - start_time:.3f} seconds")
