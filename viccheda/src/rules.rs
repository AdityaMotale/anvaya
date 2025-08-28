use std::{collections::HashMap, sync::LazyLock};

struct PhoneticAdjuncts;

impl PhoneticAdjuncts {
    pub const ANUSVARA: &'static str = "ं"; // U+0902
    pub const VISARGA: &'static str = "ः"; // U+0903
    pub const VIRAMA: &'static str = "्"; // U+094D

    pub const GUTTURALS: [&'static str; 5] = ["क", "ख", "ग", "घ", "ङ"];
    pub const PALATALS: [&'static str; 5] = ["च", "छ", "ज", "झ", "ञ"];
    pub const RETROFLEX: [&'static str; 5] = ["ट", "ठ", "ड", "ढ", "ण"];
    pub const DENTALS: [&'static str; 5] = ["त", "थ", "द", "ध", "न"];
    pub const LABIALS: [&'static str; 5] = ["प", "फ", "ब", "भ", "म"];
}

struct Orthography;

impl Orthography {
    pub const MATRA_TO_INDEP_VOWEL_MAP: LazyLock<HashMap<&'static str, &'static str>> =
        LazyLock::new(|| {
            let mut m = HashMap::new();

            m.insert("ा", "आ");
            m.insert("ि", "इ");
            m.insert("ी", "ई");
            m.insert("ु", "उ");
            m.insert("ू", "ऊ");
            m.insert("ृ", "ऋ");
            m.insert("ॄ", "ॠ");
            m.insert("े", "ए");
            m.insert("ै", "ऐ");
            m.insert("ो", "ओ");
            m.insert("ौ", "औ");

            m
        });

    pub const INDEP_VOWELS: [&'static str; 14] = [
        "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ",
    ];
}
