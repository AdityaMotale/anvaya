# ruff: noqa

"""Train a BPE Tokenizer from raw Sanskrit text."""

import argparse
import sys
from pathlib import Path
from typing import IO, Iterable
from collections import Counter, defaultdict

import regex

EOM_MARKER: str = "</w>"
NUM_MERGES: int = 10_000
MIN_FREQ: int = 1


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


def default_tokenize_word(word: str) -> list[str]:
    parts = grapheme_split(word)
    parts.append(EOM_MARKER)

    return parts


def get_initial_vocab(corpus: Iterable[str]) -> Counter:
    vocab = Counter()

    for line in corpus:
        line = line.strip()

        if not line:
            continue

        for raw_word in line.split():
            word = raw_word.strip()
            symbols = tuple(default_tokenize_word(word))

            vocab[symbols] += 1

    return vocab


def merge_vocab_once(pair: tuple[str, str], vocab: Counter) -> Counter:
    merged = {}

    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word, freq in vocab.items():
        word_str = " ".join(word)
        new_word_str = word_str.replace(bigram, replacement)
        new_word = tuple(new_word_str.split(" "))
        merged[new_word] = merged.get(new_word, 0) + freq

    return Counter(merged)


def get_pair_stats(vocab: Counter) -> dict[tuple[str, str], int]:
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq

    return pairs


def train(corpus: Iterable[str]) -> Counter:
    vocab = get_initial_vocab(corpus)

    for _ in range(NUM_MERGES):
        pairs = get_pair_stats(vocab)

        if not pairs:
            break

        best_pair, best_count = max(pairs.items(), key=lambda kv: kv[1])

        if best_count < MIN_FREQ:
            break

        vocab = merge_vocab_once(best_pair, vocab)

    return vocab


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

    l1 = lines[:10]
    vocab = train(l1)

    for pair in vocab:
        print(pair)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nBPE training completed in {end_time - start_time:.3f} seconds")
