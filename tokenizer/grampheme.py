"""Create grapheme clusters for morphemes in Sanskrit.

NOTE: This is only suitable for Sanskrit in Devanāgarī, it may
break otherwise.
"""

# grapheme_clusters.py
import unicodedata

VIRAMA = "\u094d"
ZWJ = "\u200d"  # zero width joiner
ZWNJ = "\u200c"  # zero width non-joiner (treated like normal char)


def _is_combining(ch: str) -> bool:
    return unicodedata.combining(ch) != 0


def _is_zwj(ch: str) -> bool:
    return ch == ZWJ


def _is_virama(ch: str) -> bool:
    return ch == VIRAMA


def grapheme_clusters(text: str) -> list[str]:
    """Split text into grapheme clusters.

    Heuristics:
      - combining marks (unicodedata.combining != 0) attach to current cluster
      - ZWJ attaches to current cluster
      - if the last codepoint of the current cluster is virama or ZWJ,
        the next character attaches (forming conjuncts like प्र).
      - otherwise start a new cluster.

    Returns:
        list of graphemes (split characters)

    """
    if not text:
        return []

    clusters: list[str] = []
    current = text[0]

    for ch in text[1:]:
        if _is_combining(ch) or _is_zwj(ch):
            current += ch
            continue

        last_cp = current[-1]

        if _is_virama(last_cp) or _is_zwj(last_cp):
            current += ch

            continue

        clusters.append(current)
        current = ch

    clusters.append(current)

    return clusters
