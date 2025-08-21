"""Train a BPE Tokenizer from preprocessed Sanskrit corpus."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO

from grampheme import grapheme_clusters

NUM_MERGES: int = 10
MIN_FREQ: int = 1

DANDA_TOKEN = "<DANDA>"  # noqa: S105
DOUBLE_DANDA_TOKEN = "<DANDA2>"  # noqa: S105
EOM_TOKEN = "</M>"  # noqa: S105
PAD_TOKEN = "<PAD>"  # noqa: S105
UNK_TOKEN = "<UNK>"  # noqa: S105

SPECIAL_TOKENS_SET: list[str] = [
    DANDA_TOKEN,
    DOUBLE_DANDA_TOKEN,
    EOM_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
]


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


def _initial_vocab(corpus: list[str]) -> list[list[str]]:
    vocab = []

    for verse in corpus:
        words = verse.split(" ")

        for word in words:
            word = word.strip()
            clusters = None

            if word in SPECIAL_TOKENS_SET:
                clusters = [word]
            else:
                clusters = grapheme_clusters(word)

            clusters.append(EOM_TOKEN)

            if len(clusters) > 1:
                vocab.append(clusters)

    return vocab


def count_frequencies(vocab: list[list[str]]) -> dict[tuple[str, str], int]:
    """Count frequencies of all adjacent grapheme pairs in the vocab.

    Args:
        vocab: list of words, each as list of graphemes.

    Returns:
        dict mapping (grapheme1, grapheme2) -> frequency

    """
    pair_freqs: dict[tuple[str, str], int] = defaultdict(int)

    for word in vocab:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += 1

    return dict(pair_freqs)


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

    corpus = read_file(input_file)
    print(f"Training on {len(corpus)} lines of corpus")

    initial_vocab = _initial_vocab(corpus[:10])
    pair_freq = count_frequencies(initial_vocab)

    print(pair_freq)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nBPE training completed in {end_time - start_time:.3f} seconds")
