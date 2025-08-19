# ruff: noqa: S105, RUF001

"""Common utils and constants."""

import re
import unicodedata

#
# Sanskrit Unicodes
#

VERTICAL_BAR = "|"
DOUBLE_VERTICAL_BAR = "||"

ANUSVARA = "ं"  # U+0902
VISARGA = "ः"  # U+0903
VIRAMA = "्"  # U+094D
DANDA = "।"  # U+0964
DOUBLE_DANDA = "॥"  # U+0965

VOWELS_INDEP = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"]

MATRA_TO_INDEP = {
    "ा": "आ",
    "ि": "इ",
    "ी": "ई",
    "ु": "उ",
    "ू": "ऊ",
    "ृ": "ऋ",
    "ॄ": "ॠ",
    "े": "ए",
    "ै": "ऐ",
    "ो": "ओ",
    "ौ": "औ",
}

# Consonant classes
GUTTURALS = set(list("कखगघङ"))
PALATALS = set(list("चछजझञ"))
RETROFLEX = set(list("टठडढण"))
DENTALS = set(list("तथदधन"))
LABIALS = set(list("पफबभम"))

NASAL_FOR_CLASS = {
    "guttural": "ङ" + VIRAMA,
    "palatal": "ञ" + VIRAMA,
    "retroflex": "ण" + VIRAMA,
    "dental": "न" + VIRAMA,
    "labial": "म" + VIRAMA,
}

# Vowel reverse mapping (common classical reverse-sandhi candidates)
VOWEL_REVERSE_RULES = {
    "ौ": ("अ", "ऊ"),  # au -> a + ū
    "ै": ("अ", "ई"),  # ai -> a + ī
    "ो": ("अ", "उ"),  # o  -> a + u
    "े": ("अ", "इ"),  # e  -> a + i
    "ा": ("अ", "अ"),  # ā  -> a + a
    "औ": ("अ", "ऊ"),
    "ऐ": ("अ", "ई"),
    "ओ": ("अ", "उ"),
    "ए": ("अ", "इ"),
    "आ": ("अ", "अ"),
}

# Specific consonant cluster reversals
SPECIFIC_CONSONANT_REVERSES = {
    "च्च": ("त" + VIRAMA, "च"),
    "च्छ": ("त" + VIRAMA, "छ"),
    "त्त": ("त" + VIRAMA, "त"),
    "द्ध": ("द" + VIRAMA, "ध"),
}


#
# Special Tokens
#

DANDA_TOKEN = "<DANDA>"
DOUBLE_DANDA_TOKEN = "<DANDA2>"

SPECIAL_TOKENS_SET: list[str] = [DANDA_TOKEN, DOUBLE_DANDA_TOKEN]


#
# Common utilities
#


def get_morphemes_from_verse(verse: str, exclude_tokens: list[str]) -> list[str]:
    """Extract morphemes (Sanskrit Word) from the given verse, excluding special tokens.

    Args:
        verse: preprocessed Sanskrit verse
        exclude_tokens: list of special tokens to exclude

    Returns:
        list of morphemes w/o the special tokens

    """
    token_pattern = "|".join(map(re.escape, exclude_tokens))
    pattern = f"(?:{token_pattern}|\\s+)"

    parts = re.split(pattern, verse)

    return [p for p in parts if p and p not in exclude_tokens]


def split_verse_by_special_tokens(verse: str, tokens: list[str]) -> list[str]:
    """Split given Sanskrit verse by special tokens.

    Args:
        verse: Preprocessed Sanskrit verse
        tokens: List of special tokens

    Returns:
        Returns list of Sanskrit verse's

    """
    pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
    parts = re.split(pattern, verse)

    result = []

    for i in range(0, len(parts), 2):
        segment = parts[i]

        if i + 1 < len(parts):
            segment += parts[i + 1]

        if segment:
            result.append(segment)

    return result


def insert_special_tokens(verse: str) -> str:
    """Insert special tokens into Sanskrit Verse.

    Args:
        verse: Normalized Sanskrit verse

    Returns:
        Sanskrit verse w/ special tokens

    """
    # special token for (।), used as line break in verse.
    verse = verse.replace(DANDA, DANDA_TOKEN)
    verse = verse.replace(VERTICAL_BAR, DANDA_TOKEN)

    # special token for (॥ ), indicates verse end.
    verse = verse.replace(DOUBLE_DANDA, DOUBLE_DANDA_TOKEN)
    return verse.replace(DOUBLE_VERTICAL_BAR, DOUBLE_DANDA_TOKEN)


def is_sanskrit_char(ch: str) -> bool:
    """Validate a given character.

    Args:
        ch: Input text character (must be of len == 1)

    Returns:
        boolean indicating validity of input

    Raises:
        ValueError: If given input is not a single character

    """
    if not (isinstance(ch, str) and len(ch) == 1):
        raise ValueError("Input must be a single character string")

    code_point = ord(ch)

    return (
        (0x0900 <= code_point <= 0x097F)
        or (0xA8E0 <= code_point <= 0xA8FF)
        or (0x11B00 <= code_point <= 0x11B5F)
        or (0x1CD0 <= code_point <= 0x1CFF)
    )


def sanitize_verse(verse: str) -> str:
    """Sanitize given Sanskrit verse by removing non-sanskrit (non-devanagari) script characters.

    Args:
        verse: Normalized Sanskrit verse

    Returns:
        Sanitized Sanskrit verse

    """
    return "".join(ch for ch in verse if is_sanskrit_char(ch) or ch.isspace())


def normalize_verse(verse: str) -> str:
    """Normalize (NFC) a given Sanskrit verse.

    Args:
        verse: Raw Sanskrit verse

    Returns:
        NFC Normalized Sanskrit verse

    """
    return unicodedata.normalize("NFC", verse)
