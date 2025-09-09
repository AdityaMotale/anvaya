"""Build a frequency table from xls datasets."""

from collections import Counter

import pandas as pd

VISARGA_UNICODE = "U+0903"
ANUSVARA_UNICODE = "U+0902"
VISARGA_LATIN = ":"

# list of tuple (file_url, split column index (must be seperated by "+"))
INPUT_URLS = [
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Bhagvad_Gita%20Corpus.xls",
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Astaadhyaayii%20Corpus.xls",
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/Rule-based%20Corpus%20and%20Literature%20Corpus.xls",
        2,
    ),
    (
        "https://raw.githubusercontent.com/sanskrit-sandhi/SandhiKosh/refs/heads/master/UoH_Corpus.xls",
        2,
    ),
]

# list of words extracted from split columns
SPLITS = []


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
    TypeError
        If the input is not a string.

    ValueError
        If the input string length is not exactly one character.

    Examples
    --------
    >>> get_unicode('A')
    'U+0041'

    >>> get_unicode('€')
    'U+20AC'

    """
    # sanity check
    if not isinstance(c, str):
        raise TypeError(f"Expected input type str, got {type(c).__name__!r}")

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

    Raises
    ------
    TypeError
        If the input is not a string.

    """
    if not isinstance(word, str):
        raise TypeError(f"Expected input type str, got {type(word).__name__!r}")

    if len(word) < 2:
        return word

    last = word[-1]

    if (
        last == VISARGA_LATIN
        or get_unicode(last) == VISARGA_UNICODE
        or get_unicode(last) == ANUSVARA_UNICODE
    ):
        word = word[:-1]

    return word


def is_sanskrit_word(word: str) -> bool:
    """Check if a word is a Sanskrit word.

    NOTE: We simply check if all chars falls under the Devanagari Unicode
    range (U+0900 to U+097F).

    Parameters
    ----------
    word : str
        The input word to check.

    Returns
    -------
    bool
        True if all characters are in Devanagari Unicode range

    Examples
    --------
    >>> is_sanskrit_word("राम")
    True

    >>> is_sanskrit_word("rama")
    False

    """
    for char in word:
        code = ord(char)
        if not (0x0900 <= code <= 0x097F):
            return False
    return True


# extract data
for url, splits_col in INPUT_URLS:
    df = pd.read_excel(url)
    splits_rows = df.iloc[:, splits_col]
    print(f"\n[TRACE] {df.shape} / {len(splits_rows)} <=> {url}\n{df.head()}")

    for i in range(len(splits_rows)):
        s = splits_rows[i]

        if type(s) is str:
            sp = s.split("+")

            if len(sp) > 0:
                SPLITS.extend(sanitize(s) for s in sp if is_sanskrit_word(s))

print(f"\n[DEBUG] Found {len(SPLITS)} splits")
print(f"[DEBUG] Data sample => {SPLITS[-10:-5]}")

freq_table = Counter(SPLITS)
print(f"[DEBUG] Unique Words => {len(freq_table)}")
print(f"[DEBUG] T10 in freq table => {freq_table.most_common(10)}")
