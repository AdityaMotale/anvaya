# ruff: noqa

from __future__ import annotations
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Optional

import regex  # pip install regex


def _grapheme_clusters(s: str) -> list[str]:
    return regex.findall(r"\X", s)


class BPETokenizerProd:
    def __init__(self) -> None:
        self.bpe_merges: list[tuple[str, str]] = []
        self.bpe_ranks: dict[tuple, int] = {}
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.eos_marker: str = "</M>"
        self.special_tokens: list[str] = []
        self._encode_cache: dict[str, list[str]] = {}
        self.unk_token: str = "<UNK>"

    @classmethod
    def load(cls, model_path: str | Path) -> "BPETokenizerProd":
        p = Path(model_path)

        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))

        tok = cls()
        merges = data.get("merges", [])
        tok.bpe_merges = [tuple(pair) for pair in merges]
        tok.bpe_ranks = {tuple(pair): i for i, pair in enumerate(tok.bpe_merges)}

        tok.token2id = {k: int(v) for k, v in data.get("token2id", {}).items()}
        tok.id2token = {int(v): k for k, v in tok.token2id.items()}

        cfg = data.get("config", {})
        tok.eos_marker = cfg.get("eos_marker", tok.eos_marker)
        tok.special_tokens = cfg.get("special_tokens", [])
        tok.unk_token = "<UNK>"

        return tok

    def token_to_id(self, token: str) -> Optional[int]:
        return self.token2id.get(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self.id2token.get(idx)

    def _initial_symbols(self, word: str) -> list[str]:
        if word in self.special_tokens:
            return [word, self.eos_marker]

        parts = list(word)
        parts.append(self.eos_marker)

        return parts

    def encode_word(self, word: str) -> list[str]:
        if word in self._encode_cache:
            return list(self._encode_cache[word])

        symbols = self._initial_symbols(word)

        while True:
            if len(symbols) < 2:
                break

            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_ranks = {
                pair: self.bpe_ranks[pair] for pair in pairs if pair in self.bpe_ranks
            }

            if not candidate_ranks:
                break

            best_pair = min(candidate_ranks, key=candidate_ranks.get)
            bigram = " ".join(best_pair)
            merged_symbol = best_pair[0] + best_pair[1]
            sym_str = " ".join(symbols)
            sym_str = sym_str.replace(bigram, merged_symbol)
            symbols = sym_str.split(" ")

        if symbols and symbols[-1] == self.eos_marker:
            symbols = symbols[:-1]

        self._encode_cache[word] = list(symbols)
        return list(symbols)

    def encode(
        self, text: str, return_words: bool = True
    ) -> list[list[str]] | list[str]:
        words = [w for w in text.strip().split() if w]
        encoded_words = [self.encode_word(w) for w in words]

        if return_words:
            return encoded_words

        flat = [t for sub in encoded_words for t in sub]
        return flat

    def encode_to_ids(
        self, text: str, flatten: bool = True, unk_as_id: Optional[int] = None
    ) -> list[int] | list[list[int]]:
        if unk_as_id is None:
            unk_as_id = self.token2id.get("<UNK>", None)

        encoded = (
            self.encode(text, return_words=not flatten)
            if not flatten
            else self.encode(text, return_words=True)
        )

        def tokens_to_ids(tok_list: list[str]) -> list[int]:
            ids = []

            for t in tok_list:
                tid = self.token2id.get(t)

                if tid is None:
                    if unk_as_id is None:
                        ids.append(None)
                    else:
                        ids.append(unk_as_id)
                else:
                    ids.append(tid)

            return ids

        if flatten:
            flat_ids = []

            for w in encoded:
                flat_ids.extend(tokens_to_ids(w))

            return flat_ids
        else:
            return [tokens_to_ids(w) for w in encoded]

    def decode_ids(self, ids: Iterable[int], flatten: bool = True) -> str | list[str]:
        if flatten:
            tokens = []

            for i in ids:
                tok = self.id2token.get(int(i), None)

                if tok is None:
                    tok = "<UNK>" if "<UNK>" in self.token2id else ""

                tokens.append(tok)

            words: list[str] = []
            current: list[str] = []

            for t in tokens:
                if t in self.special_tokens:
                    if current:
                        words.append("".join(current))
                        current = []

                    words.append(t)
                    continue

                if t == self.eos_marker:
                    if current:
                        words.append("".join(current))
                        current = []
                    else:
                        words.append("")
                    continue

                current.append(t)

            if current:
                words.append("".join(current))

            return " ".join(words)
        else:
            words = []

            for word_ids in ids:
                toks = [self.id2token.get(int(i), "<UNK>") for i in word_ids]
                words.append("".join([t for t in toks if t != self.eos_marker]))

            return words

    def vocab_size(self) -> int:
        return len(self.token2id)

    def clear_cache(self) -> None:
        self._encode_cache.clear()


def _smoke_test(model_path: str) -> None:
    tok = BPETokenizerProd.load(model_path)
    examples = [
        "तस्यां चीरं वसानायां नाथवत्यामनाथवत् <DANDA>",
        "प्रचुक्रोश जनः सर्वो धिक् त्वां दशरथं त्विति  <DANDA2>",
    ]

    for line in examples:
        toks = tok.encode(line, return_words=True)
        ids = tok.encode_to_ids(line, flatten=True)
        decoded = tok.decode_ids(ids, flatten=True)

        print("LINE: ", line)
        print("TOKS: ", toks)
        print("IDS : ", ids[:60], "..." if len(ids) > 60 else "")
        print("DECD: ", decoded)
        print("-" * 60)


def main():
    _smoke_test("../datasets/bpe.json")


if __name__ == "__main__":
    main()
