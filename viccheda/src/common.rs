use std::{collections::HashMap, sync::LazyLock};

trait AsStr {
    fn as_str(&self) -> &'static str;
}

#[derive(Debug)]
pub(crate) enum Adjuncts {
    ANUSVARA,
    VISARGA,
    VIRAMA,
}

impl AsStr for Adjuncts {
    fn as_str(&self) -> &'static str {
        match self {
            Adjuncts::ANUSVARA => "ं", // U+0902
            Adjuncts::VISARGA => "ः", // U+0903
            Adjuncts::VIRAMA => "्",   // U+094D
        }
    }
}

#[derive(Debug)]
pub(crate) enum Vowel {
    A,
    AA,
    I,
    II,
    U,
    UU,
    R,
    RR,
    E,
    AI,
    O,
    AU,
}

impl AsStr for Vowel {
    fn as_str(&self) -> &'static str {
        match self {
            Vowel::A => "",
            Vowel::AA => "ा",
            Vowel::I => "ि",
            Vowel::II => "ी",
            Vowel::U => "ु",
            Vowel::UU => "ू",
            Vowel::R => "ृ",
            Vowel::RR => "ॄ",
            Vowel::E => "े",
            Vowel::AI => "ै",
            Vowel::O => "ो",
            Vowel::AU => "ौ",
        }
    }
}

#[derive(Debug)]
pub(crate) enum Consonant {
    // Gutturals (velars)
    Ka,
    Kha,
    Ga,
    Gha,
    Nga,

    // Palatals
    Cha,
    Chha,
    Ja,
    Jha,
    Nya,

    // Retroflex
    Tta,
    Ttha,
    Dda,
    Ddha,
    Nna,

    // Dentals
    Ta,
    Tha,
    Da,
    Dha,
    Na,

    // Labials
    Pa,
    Pha,
    Ba,
    Bha,
    Ma,

    // Semi-vowels
    Ya,
    Ra,
    La,
    Va,

    // Sibilants + Aspirate
    Sha,
    Ssa,
    Sa,
    Ha,
}

impl AsStr for Consonant {
    fn as_str(&self) -> &'static str {
        match self {
            // Gutturals
            Consonant::Ka => "क",
            Consonant::Kha => "ख",
            Consonant::Ga => "ग",
            Consonant::Gha => "घ",
            Consonant::Nga => "ङ",

            // Palatals
            Consonant::Cha => "च",
            Consonant::Chha => "छ",
            Consonant::Ja => "ज",
            Consonant::Jha => "झ",
            Consonant::Nya => "ञ",

            // Retroflex
            Consonant::Tta => "ट",
            Consonant::Ttha => "ठ",
            Consonant::Dda => "ड",
            Consonant::Ddha => "ढ",
            Consonant::Nna => "ण",

            // Dentals
            Consonant::Ta => "त",
            Consonant::Tha => "थ",
            Consonant::Da => "द",
            Consonant::Dha => "ध",
            Consonant::Na => "न",

            // Labials
            Consonant::Pa => "प",
            Consonant::Pha => "फ",
            Consonant::Ba => "ब",
            Consonant::Bha => "भ",
            Consonant::Ma => "म",

            // Semi-vowels
            Consonant::Ya => "य",
            Consonant::Ra => "र",
            Consonant::La => "ल",
            Consonant::Va => "व",

            // Sibilants + Aspirate
            Consonant::Sha => "श",
            Consonant::Ssa => "ष",
            Consonant::Sa => "स",
            Consonant::Ha => "ह",
        }
    }
}

pub(crate) struct Orthography;

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

    pub const GUTTURALS: [Consonant; 5] = [
        Consonant::Ka,
        Consonant::Kha,
        Consonant::Ga,
        Consonant::Gha,
        Consonant::Nga,
    ];

    pub const PALATALS: [Consonant; 5] = [
        Consonant::Cha,
        Consonant::Chha,
        Consonant::Ja,
        Consonant::Jha,
        Consonant::Nya,
    ];

    pub const RETROFLEX: [Consonant; 5] = [
        Consonant::Tta,
        Consonant::Ttha,
        Consonant::Dda,
        Consonant::Ddha,
        Consonant::Nna,
    ];

    pub const DENTALS: [Consonant; 5] = [
        Consonant::Ta,
        Consonant::Tha,
        Consonant::Da,
        Consonant::Dha,
        Consonant::Na,
    ];

    pub const LABIALS: [Consonant; 5] = [
        Consonant::Pa,
        Consonant::Pha,
        Consonant::Ba,
        Consonant::Bha,
        Consonant::Ma,
    ];

    pub const SEMIVOWELS: [Consonant; 4] =
        [Consonant::Ya, Consonant::Ra, Consonant::La, Consonant::Va];

    pub const SIBILANTS: [Consonant; 3] = [Consonant::Sha, Consonant::Ssa, Consonant::Sa];

    pub const ASPIRATE: [Consonant; 1] = [Consonant::Ha];
}
