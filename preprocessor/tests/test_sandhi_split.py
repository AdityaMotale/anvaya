# ruff: noqa

import unicodedata
from sandhi_split import grapheme_list


class TestGraphemeList:
    def test_single_consonants(self):
        assert grapheme_list("कमल") == ["क", "म", "ल"]

    def test_consonant_with_matra(self):
        # क + ि (matra) should be one cluster
        assert grapheme_list("कि") == ["कि"]

    def test_consonant_with_virama_cluster(self):
        # क + ् + त = cluster split as ["क्","त"]
        word = "क्त"
        assert grapheme_list(word) == ["क्", "त"]

    def test_independent_vowels(self):
        # independent vowels should remain separate clusters
        assert grapheme_list("अइउ") == ["अ", "इ", "उ"]

    def test_consonant_with_anusvara(self):
        # क + ि + ं (anusvāra) stays together
        assert grapheme_list("किं") == ["किं"]

    def test_consonant_with_visarga(self):
        # क + ः (visarga) should be one cluster
        assert grapheme_list("कः") == ["कः"]

    def test_mixed_word(self):
        # नमः (न + म + ः)
        assert grapheme_list("नमः") == ["न", "मः"]

    def test_trailing_combining_mark_weird_case(self):
        # edge case: string starts with a combining mark (not common but should still cluster)
        s = "\u093f"  # DEVANAGARI VOWEL SIGN I
        out = grapheme_list(s)
        # It should treat it as one cluster
        assert out == [s]

    def test_normalization_nfc(self):
        # ensure normalization doesn’t split graphemes incorrectly
        s = unicodedata.normalize("NFC", "कि")  # क + ि
        assert grapheme_list(s) == ["कि"]
