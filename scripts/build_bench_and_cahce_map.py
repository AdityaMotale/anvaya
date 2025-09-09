"""Build a sandhi split dataset for benching."""

from pathlib import Path
from unicodedata import normalize

import pandas as pd

VISARGA_LATIN = ":"
VISARGA_UNICODE = "U+0903"
ANUSVARA_UNICODE = "U+0902"
VIRAMA_UNICODE = "U+094D"

OUT_CANDI = "candi.txt"
OUT_CACHE = "cache.txt"
FREQ_FILE = "freq.txt"

# list of tuple (file [url, path], word column index, split column index (must be seperated by "+"))
INPUT_URLS = [
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Bhagvad_Gita%20Corpus.xls",
        1,
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Astaadhyaayii%20Corpus.xls",
        1,
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Rule-based%20Corpus%20and%20Literature%20Corpus.xls",
        1,
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/UoH_Corpus.xls",
        1,
        2,
    ),
]

# dict of words and split candidate
CANDIDATE_MAP = {}

# dict of words to be cached
CACHE_MAP = {}


def nfc(s: str) -> str:
    """Normalize word to NFC.

    Parameters
    ----------
    s : str
        Input word.

    Returns
    -------
    str
       Normalized word in NFC.

    """
    return normalize("NFC", s)


def get_unicode(c: str) -> str:
    """Return the Unicode of a single character as a string ('U+XXXX').

    Parameters
    ----------
    c : str
        A single Unicode character string.

    Returns
    -------
    str
        The Unicode string in format 'U+XXXX'.

    Raises
    ------
    ValueError
        If the input string length is not exactly one character.

    """
    if len(c) != 1:
        raise ValueError(
            f"Input string must be exactly one character long, got length {len(c)}"
        )

    code = ord(c)
    return f"U+{code:04X}"


def sanitize(word: str) -> str:
    """Remove trailing visarga or anusvara characters from a word if present.

    Parameters
    ----------
    word : str
        The input word to sanitize. Must be a string.

    Returns
    -------
    str
        The sanitized word with trailing visarga or anusvara removed.

    """
    if len(word) < 2:
        return word

    last = word[-1]
    last_unicode = get_unicode(last)

    if (
        last == VISARGA_LATIN
        or last_unicode == VISARGA_UNICODE
        or last_unicode == ANUSVARA_UNICODE
        or last_unicode == VIRAMA_UNICODE
    ):
        word = word[:-1]

    return word


freq_table = {}

with open(FREQ_FILE, "r") as f:
    for line in f:
        word, freq = line.strip().split(",")
        freq_table[word] = int(freq)

print(f"\n[DEBUG] Freq Table contains {len(freq_table.items())} entries")

# extract data
for url, word_col, splits_col in INPUT_URLS:
    df = pd.read_excel(url)
    word_rows = df.iloc[:, word_col]
    splits_rows = df.iloc[:, splits_col]
    print(
        f"\n[TRACE] {df.shape} / ({len(splits_rows)}, {len(word_rows)}) <=> {url}\n{df.head()}"
    )

    for i in range(len(word_rows)):
        w = word_rows[i]
        s = splits_rows[i]

        if type(s) is str and type(w) is str:
            word = sanitize(w.strip())
            splits = "+".join(nfc(sanitize(s)) for s in s.split("+"))
            candis = splits.split("+")

            if len(word.split(" ")) > 1:
                continue

            if len(candis) > 2:
                CACHE_MAP[nfc(word)] = splits
                continue

            if len(candis) == 2:
                for c in candis:
                    if c.strip() in freq_table and freq_table[c.strip()] > 2500:
                        CACHE_MAP[nfc(word)] = splits
                        break

                CANDIDATE_MAP[nfc(word)] = splits

print(f"\n[DEBUG] Added {len(CANDIDATE_MAP)} entries to Candidate Map")
print(f"\n[DEBUG] Added {len(CACHE_MAP)} entries to Cache Map")

# candidate map
out_candi = Path(OUT_CANDI).open(mode="w")

for key, value in CANDIDATE_MAP.items():
    out_candi.write(f"{key},{value}\n")

out_candi.close()

print(f"[DEBUG] Candidate Map written to '{OUT_CANDI}'")

# cache map
out_cache = Path(OUT_CACHE).open(mode="w")

for key, value in CACHE_MAP.items():
    out_cache.write(f"{key},{value}\n")

out_cache.close()

print(f"[DEBUG] Candidate Map written to '{OUT_CACHE}'")
