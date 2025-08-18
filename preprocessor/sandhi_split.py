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
