"""Module to aid with Sandhi splitting for Sanskrit Verses.

## What is `Sandhi`?

The word sandhi itself means joining or junction.

In Sanskrit, sandhi is a phonetic change that occur at the junction where two
sounds meet, typically at word or morpheme boundaries.

For example, many native English speakers, do not pronounce the final "g" sound
for words like "going", "walking", etc. They are pronounced as "goin" & "walkin",
respectively. These kinds of phonetic changes occur in many languages, including
Sanskrit.

e.g. `सीता + अश्वम् → सीताश्वम्`

Here the words "सीता" and "अश्व" are combined in a single morpheme.

So, to define, sandhi refer to the phonetic change or tranformation that occur at
the junction where two sounds or morpheme meets, either within or at edges.

## Why is `Sandhi`?

Sandhi is a regular sound-change that make connected speech smoother. As Sanskrit
is spoken rythmatically, the flow of punctionations should be smooth for Shlokas to
be easily remembered.

When we speak quickly, it is difficult to pause after the `ā` in `sītā` and start
again with the first `a` of `aśvam`. So by combining these two we get `sītāśvam`,
which is much easier to speek/chant in rythm, due to lack of pause in-between.

So by using sandhi, the pronounciation preserves its rythm and makes speech fluent.

## Why we must split sandhis?

Sandhi's in Sanskrit are unusual, becuase they are often written down.

For example, the Sanskrit words gajo (गजो) and gajas (गजस्) have exactly the same
meaning, the elephant, but has no effect on grammer. Yet they are written at different
places, depending on the phonetic rythm of the verse.

As sandhi's are written, the same underlying morpheme form can occur in different
surface forms, e.g "Gaj (गज)" can be written as "gajo (गजो)" and "gajas (गजस्)".

For the process of tokenization, this inflates the vocabulary and scatters contexts for
the same lemma (base morpheme), not ideal when we need to learn from the surface forms
of the text.

By reversing sandhi's (where appropriate) we reduce vocabulary sparsity and improve
semantic/contextual consistency.

## How sandhi is split?

Following are major sandhi categories,

▶ Vowel sandhi (स्वर सन्धि) — changes when vowels meet (e.g., a + a → ā,
  i + a → ya-like sounds).
▶ Consonant sandhi (व्यञ्जन सन्धि) — assimilation/gemination when consonants meet.
▶ Visarga sandhi (विसर्ग सन्धि) — changes involving visarga.
▶ Anusvāra sandhi (अनुस्वार सन्धि) — nasalization mapped to a homorganic nasal
  (ṃ → ṅ/ñ/ṇ/n/m depending on followed consonant).
▶ Special marks — avagraha (ऽ) indicates elision of initial 'a' and is phonologically
  meaningful.

So by the process of reverse-engineering, we try to split sandhi's where appropriate.
"""

import re
import unicodedata

from common import (
    ANUSVARA,
    DANDA,
    DENTALS,
    DOUBLE_DANDA,
    DOUBLE_VERTICAL_BAR,
    GUTTURALS,
    LABIALS,
    MATRA_TO_INDEP,
    NASAL_FOR_CLASS,
    PALATALS,
    RETROFLEX,
    SPECIFIC_CONSONANT_REVERSES,
    VERTICAL_BAR,
    VIRAMA,
    VISARGA,
    VOWEL_REVERSE_RULES,
    VOWELS_INDEP,
    normalize_verse,
)


def grapheme_list(verse: str) -> list[str]:
    """Split a Sanskrit (Devanāgarī) verse into grapheme clusters.

    > Cluster here is defined conservatively as a base character (typically a
    consonant or independent vowel) grouped with any following unicode combining
    marks (matrās, virāma, anusvāra, visarga, etc.)

    Args:
        verse: NFC normalized Sanskrit verse

    Returns:
        A list of grapheme clusters

    Example:
    >>> grapheme_list("कमल")
    ['क', 'म', 'ल']

    >>> grapheme_list("किं")
    ['किं']

    >>> grapheme_list("क् त")
    ['क्', 'त']

    """
    clusters = []
    buf = ""

    for ch in verse:
        # combining unicode cluster mark
        if unicodedata.category(ch).startswith("M"):
            buf += ch
        else:
            if buf:
                clusters.append(buf)

            buf = ch

    if buf:
        clusters.append(buf)

    return clusters


def is_consonant(ch: str) -> bool:
    """Check if given Sanskrit character is a consonent.

    > Basic Devanāgarī consonant range is (क..ह)
    > `क्ष` (kṣa) and `ज्ञ` (jña) are conjunct consonants or ligatures
      formed by combining two consonants (क् + ष and ज् + ञ respectively)
    > `ळ` is a retroflex lateral consonant used mainly in Marathi

    Arg:
        ch: Input Sanskrit character

    Returns:
        boolean indicating if a given character is a consonent

    """
    cp = ord(ch) if ch else 0
    return 0x0915 <= cp <= 0x0939


def is_vowel_indep(ch: str) -> bool:
    """Check if given character exists in list of independent vowels.

    Arg:
        ch: Input Sanskrit character

    Returns:
        boolean indicating if a given character exists in independent
        vowels list

    """
    return ch in VOWELS_INDEP


def is_matras(ch: str) -> bool:
    """Check if given character exists in list of independent matras.

    Arg:
        ch: Input Sanskrit character

    Returns:
        boolean indicating if a given character exists in independent
        matra's list

    """
    return ch in MATRA_TO_INDEP.keys()


def anusvara_replacements(next_ch: str) -> list[str]:
    """Return replacement `nasal + virama` suggestions for `anusvāra` before next char.

    Arg:
        next_ch: Next character from the stream of Sanskrit verse

    Returns:
        list of suggestions for replacement for `anusvāra`

    """
    # sanity check
    if not next_ch:
        return []

    if next_ch in GUTTURALS:
        return [NASAL_FOR_CLASS["guttural"]]

    if next_ch in PALATALS:
        return [NASAL_FOR_CLASS["palatal"]]

    if next_ch in RETROFLEX:
        return [NASAL_FOR_CLASS["retroflex"]]

    if next_ch in DENTALS:
        return [NASAL_FOR_CLASS["dental"]]

    if next_ch in LABIALS:
        return [NASAL_FOR_CLASS["labial"]]

    # fallback
    return [NASAL_FOR_CLASS["dental"]]


def visarga_replacements(next_ch: str) -> list[str]:
    """Visarga reversals by remove or replace with s/ś.

    Arg:
        next_ch: Next character from the stream of Sanskrit verse

    Returns:
        list of suggestions for visarga reversals

    """
    if not next_ch:
        return [""]

    return ["", "स" + VIRAMA, "श" + VIRAMA]


def gemination_reversal(left: str, right: str) -> list[tuple[str, str]]:
    """If right starts with doubled consonant (same consonant twice), move one consonant to left.

    Args:
        left: Character to the left
        right: Character to the right

    Returns:
        list of tuples with performed reversals

    """
    res = []

    if len(right) >= 2 and right[0] == right[1] and is_consonant(right[0]):
        c = right[0]

        new_left = left + c
        new_right = right[1:]

        res.append((new_left, new_right))

    return res


def reverse_at_boundary(left: str, right: str) -> list[tuple[str, str]]:
    """Get list of reversals candidate for given left and right strings.

    Args:
        left: Text tp the left
        right: Text to the right

    Returns:
        list of reversals candidates

    """
    left = normalize_verse(left)
    right = normalize_verse(right)
    results = []

    # sanity check
    if not left or not right:
        return results

    lg = grapheme_list(left)
    rg = grapheme_list(right)

    last = lg[-1] if lg else ""
    first = rg[0] if rg else ""

    # ▶ Vowel-matra independent reverse rules
    if last:
        last_cp = last[-1]

        if last_cp in VOWEL_REVERSE_RULES and (
            is_vowel_indep(first) or is_matras(first) or first in VOWELS_INDEP
        ):
            a, b = VOWEL_REVERSE_RULES[last_cp]

            new_left = "".join(lg[:-1]) + a
            new_right = b + right

            results.append((new_left, new_right))

    # ▶ Anusvara reversal
    if left.endswith(ANUSVARA):
        next_c = right[0] if right else ""

        for rep in anusvara_replacements(next_c):
            new_left = left[:-1] + rep
            results.append((new_left, right))

    # ▶ Visarga reversal
    if left.endswith(VISARGA):
        for rep in visarga_replacements(first[0] if first else ""):
            new_left = left[:-1] + rep
            results.append((new_left, right))

    # ▶ Gemination generalized reversal
    for nl, nr in gemination_reversal(left, right):
        results.append((nl, nr))

    # ▶ Specific consonant cluster reversals
    for key, (l_rep, r_rep) in SPECIFIC_CONSONANT_REVERSES.items():
        if right.startswith(key):
            new_left = left + l_rep
            new_right = r_rep + right[len(key) :]
            results.append((new_left, new_right))

    # ▶ Deduplicate by (l, r), keep first rule label encountered
    seen = set()
    uniq = []

    for nl, nr in results:
        if (nl, nr) not in seen:
            seen.add((nl, nr))
            uniq.append((nl, nr))

    return uniq


def all_splits_for_token(token: str) -> list[tuple[str, str]]:
    """For a given token, try every grapheme boundary and return candidate splits.

    Args:
        token: Input Sanskrit morpheme

    Returns:
        list of split candidates containing tuple as (left_candidate, right_candidate)

    """
    token = normalize_verse(token)
    g = grapheme_list(token)
    results = []

    for i in range(1, len(g)):
        left = "".join(g[:i])
        right = "".join(g[i:])

        candidates = reverse_at_boundary(left, right)

        for nl, nr in candidates:
            results.append((nl, nr))

    return results


def generate_split_candidates(text: str) -> list[tuple[str, str]]:
    """Process and get a list of candidate splits for NFC normalized Sanskrit morpheme.

    Args:
        text: Normalized Sanskrit morpheme

    Returns:
        list of split candidate as tuple: (nl, nr)

    """
    out = []

    for line in text.splitlines():
        tokens = re.split(r"(\s+)", line)

        for t in tokens:
            if not t.strip():
                continue

            # skip special tokens
            if any(
                ch in f"{DANDA} {DOUBLE_DANDA} {VERTICAL_BAR} {DOUBLE_VERTICAL_BAR}"
                for ch in t
            ):
                continue

            if not any("\u0900" <= ch <= "\u097f" for ch in t):
                continue

            splits = all_splits_for_token(t)

            for nl, nr in splits:
                out.append((nl, nr))

    return out
