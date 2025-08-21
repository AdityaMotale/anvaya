"""Train a BPE Tokenizer from preprocessed Sanskrit corpus."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO, TYPE_CHECKING

import regex

if TYPE_CHECKING:
    from collections.abc import Iterable

#
# Default Configs
#
DEFAULT_TARGET_VOCAB = 28_000
DEFAULT_MIN_FREQ = 2
DEFAULT_EOS = "</M>"
DEFAULT_SPECIAL_TOKENS = ["<DANDA>", "<DANDA2>", "</M>", "<PAD>", "<UNK>"]
LOG_INTERVAL_MERGES = 50

# ---------------------------
# Logger
# ---------------------------
logger = logging.getLogger("BPE Training")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def open_file(path: str, mode: str = "r") -> IO:
    """Open a file with the given path and mode.

    Args:
        path: The file path
        mode: The mode in which to open the file (default is "r").

    Returns:
        A file object opened with the specified mode.

    """
    return Path(path).open(mode, encoding="utf-8")


def stream_lines(path: str) -> Iterable[str]:
    """Yield lines from a UTF-8 text file lazily, one at a time.

    Args:
        path: The file path

    Yields:
        A line from the file

    """
    file = open_file(path)

    for line in file:
        yield line.rstrip("\n")


def _grapheme_clusters(s: str) -> list[str]:
    return regex.findall(r"\X", s)


def _build_initial_vocab(
    lines: Iterable[str],
    eos_marker: str,
    special_tokens: list[str],
) -> Counter:
    vocab: Counter = Counter()

    for line in lines:
        if not line:
            continue

        for raw_word in line.split():
            word = raw_word.strip()

            if not word:
                continue
            if word in special_tokens:
                symbols = (word, eos_marker)
            else:
                symbols = (*tuple(_grapheme_clusters(word)), eos_marker)

            vocab[symbols] += 1

    return vocab


def get_pair_frequencies(vocab: Counter) -> dict[tuple[str, str], int]:
    """Count frequencies of all adjacent grapheme pairs in the vocab.

    Args:
        vocab: list of words, each as list of graphemes.

    Returns:
        dict mapping (grapheme1, grapheme2) -> frequency

    """
    pairs: dict[tuple[str, str], int] = defaultdict(int)

    for word, freq in vocab.items():
        if len(word) < 2:
            continue

        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq  # noqa

    return pairs


def merge_vocab_once(pair: tuple[str, str], vocab: Counter) -> Counter:
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
    new_vocab: Counter = Counter()

    for word, freq in vocab.items():
        s = " ".join(word)

        if bigram in s:
            s_new = s.replace(bigram, merged_symbol)
            new_word = tuple(s_new.split(" "))
            new_vocab[new_word] += freq
        else:
            new_vocab[word] += freq

    return new_vocab


def extract_token_set(vocab: Counter) -> set:
    """Get the set of all token symbols present in vocab.

    Returns:
        set of tokens

    """
    tokens = set()

    for word in vocab:
        for tok in word:
            tokens.add(tok)

    return tokens


def build_token2id(
    vocab: Counter, special_tokens: list[str], eos_marker: str
) -> dict[str, int]:
    """Build token2id mappings.

    > Special tokens are placed first in the given order.

    Returns:
        token2id dictioanry

    """
    token_freq: Counter = Counter()

    for word, cnt in vocab.items():
        for tok in word:
            token_freq[tok] += cnt

    token_list: list[str] = []
    seen = set()

    for st in special_tokens:
        if st not in seen:
            token_list.append(st)
            seen.add(st)

    sorted_tokens = sorted(
        ((tok, f) for tok, f in token_freq.items() if tok not in seen),
        key=lambda kv: (-kv[1], kv[0]),
    )

    for tok, _ in sorted_tokens:
        token_list.append(tok)
        seen.add(tok)

    if eos_marker not in seen:
        token_list.append(eos_marker)

    return {tok: idx for idx, tok in enumerate(token_list)}


def train_bpe(
    input_path: str,
) -> dict:
    """Train a BPE Tokenizer on preprocessed Sanskrit text corpus.

    Returns:
        model as dict object

    """
    target_vocab: int = DEFAULT_TARGET_VOCAB
    min_freq: int = DEFAULT_MIN_FREQ
    max_merges: int | None = None
    eos_marker: str = DEFAULT_EOS
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS

    logger.info("Building initial vocab from %s", input_path)

    lines_iter = stream_lines(input_path)
    vocab = _build_initial_vocab(
        lines_iter,
        eos_marker=eos_marker,
        special_tokens=special_tokens,
    )

    logger.info("Initial vocab types: %d  (unique word forms)", len(vocab))

    token_set = extract_token_set(vocab)
    logger.info("Initial token set size: %d", len(token_set))

    merges: list[tuple[str, str]] = []
    merge_count = 0

    last_merged_pair: tuple[str, str] | None = None
    last_merged_freq: int | None = None

    # safety cap
    max_merges = max_merges if max_merges is not None else (target_vocab * 10)

    while True:
        if len(token_set) >= target_vocab:
            logger.info(
                "Reached target token set size: %d >= %d", len(token_set), target_vocab
            )
            break

        if merge_count >= max_merges:
            logger.warning("Reached max_merges cap: %d", merge_count)
            break

        pair_freqs = get_pair_frequencies(vocab)
        if not pair_freqs:
            logger.info("No adjacent pairs left to merge.")
            break

        best_freq = max(pair_freqs.values())
        best_candidates = [p for p, f in pair_freqs.items() if f == best_freq]
        best_pair = sorted(best_candidates)[0]

        if best_freq < min_freq:
            logger.info(
                "Best pair freq %d is below min_freq %d; stopping.", best_freq, min_freq
            )
            break

        last_merged_pair = best_pair
        last_merged_freq = best_freq

        vocab = merge_vocab_once(best_pair, vocab)
        merges.append(best_pair)
        merge_count += 1

        # log periodically
        if merge_count % LOG_INTERVAL_MERGES == 0 or merge_count < 10:
            logger.info(
                "Merge #%d: %s (freq=%d)",
                merge_count,
                best_pair,
                best_freq,
            )

        token_set = extract_token_set(vocab)

    token2id = build_token2id(
        vocab, special_tokens=special_tokens, eos_marker=eos_marker
    )
    final_merges = [[a, b] for (a, b) in merges]

    logger.info(
        "Final token set size: %d; final vocabulary tokens in token2id: %d",
        len(token_set),
        len(token2id),
    )
    logger.info(
        "Final size of merge list: %d",
        len(final_merges),
    )
    logger.info(
        "Last merged pair: %s with frequency %d", last_merged_pair, last_merged_freq
    )

    return {
        "merges": final_merges,
        "token2id": token2id,
        "config": {
            "target_vocab": target_vocab,
            "min_freq": min_freq,
            "eos_marker": eos_marker,
            "special_tokens": special_tokens,
            "max_merges_cap": max_merges,
        },
    }


def save_model(path: str, model: dict) -> None:
    """Write lines to a UTF-8 text file.

    Args:
        path: The file path
        model: Trained BPE Model

    """
    file = Path(path)
    file.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Train a BPE Tokenizer from raw Sanskrit text."""
    parser = argparse.ArgumentParser(
        description="Train a BPE Tokenizer from raw Sanskrit text."
    )
    parser.add_argument(
        "input", nargs="?", help="Path to preprocessed input file (UTF-8)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Path to output file (UTF-8)",
    )

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    # exit(1) if invalid args
    if input_file is None or output_file is None:
        parser.print_help()
        sys.exit(1)

    # training loop
    model = train_bpe(input_file)
    save_model(output_file, model)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\nBPE training completed in {end_time - start_time:.3f} seconds")
