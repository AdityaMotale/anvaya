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


def merge_pair(pair: tuple[str, str], vocab: list[list[str]]) -> list[list[str]]:
    """Merge all occurrences of the adjacent pair `pair` in the vocab.

    Args:
        pair: a tuple of two symbols to merge, e.g. ("रा", "म")
        vocab: list of words, each word is a list of grapheme tokens (strings)

    Returns:
        new_vocab: new vocab w/ merged pairs

    """
    a, b = pair
    bigram = f"{a} {b}"
    merged_symbol = a + b

    new_vocab: list[list[str]] = []

    for word in vocab:
        wstr = " ".join(word)

        if bigram in wstr:
            wstr = wstr.replace(bigram, merged_symbol)
            new_word = wstr.split(" ")
            new_vocab.append(new_word)
        else:
            new_vocab.append(list(word))

    return new_vocab


def _training_loop(corpus: list[str]) -> None:
    vocab = _initial_vocab(corpus)
    merges: list[tuple[str, str]] = []

    while True:
        pair_freq = count_frequencies(vocab)

        if not pair_freq:
            break

        best_pair = max(pair_freq.items(), key=lambda kv: kv[1])[0]

        if pair_freq[best_pair] < MIN_FREQ:
            break

        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

    print(vocab)
    print(merges)


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

    _training_loop(corpus[:10])


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nBPE training completed in {end_time - start_time:.3f} seconds")
