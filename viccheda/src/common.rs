use std::{collections::HashMap, sync::LazyLock};

pub(crate) trait AsStr {
    fn as_str(&self) -> &'static str;
}

pub(crate) trait AsChar {
    fn as_char(&self) -> char;
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
            Adjuncts::ANUSVARA => "ं",
            Adjuncts::VISARGA => "ः",
            Adjuncts::VIRAMA => "्",
        }
    }
}

impl AsChar for Adjuncts {
    fn as_char(&self) -> char {
        match self {
            Adjuncts::ANUSVARA => '\u{0902}',
            Adjuncts::VISARGA => '\u{0903}',
            Adjuncts::VIRAMA => '\u{094D}',
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
            Vowel::A => unimplemented!(),
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

impl AsChar for Vowel {
    fn as_char(&self) -> char {
        match self {
            Vowel::A => '\u{0905}',
            Vowel::AA => '\u{093E}',
            Vowel::I => '\u{093F}',
            Vowel::II => '\u{0940}',
            Vowel::U => '\u{0941}',
            Vowel::UU => '\u{0942}',
            Vowel::R => '\u{0943}',
            Vowel::RR => '\u{0944}',
            Vowel::E => '\u{0947}',
            Vowel::AI => '\u{0948}',
            Vowel::O => '\u{094B}',
            Vowel::AU => '\u{094C}',
        }
    }
}

impl Vowel {
    fn to_indep(&self) -> IndepVowel {
        match self {
            Vowel::A => IndepVowel::A,
            Vowel::AA => IndepVowel::AA,
            Vowel::I => IndepVowel::I,
            Vowel::II => IndepVowel::II,
            Vowel::U => IndepVowel::U,
            Vowel::UU => IndepVowel::UU,
            Vowel::R => IndepVowel::R,
            Vowel::RR => IndepVowel::RR,
            Vowel::E => IndepVowel::E,
            Vowel::AI => IndepVowel::AI,
            Vowel::O => IndepVowel::O,
            Vowel::AU => IndepVowel::AU,
        }
    }
}

#[derive(Debug)]
pub(crate) enum IndepVowel {
    A,
    AA,
    I,
    II,
    U,
    UU,
    R,
    RR,
    L,
    LL,
    E,
    AI,
    O,
    AU,
}

impl AsStr for IndepVowel {
    fn as_str(&self) -> &'static str {
        match self {
            IndepVowel::A => "अ",
            IndepVowel::AA => "आ",
            IndepVowel::I => "इ",
            IndepVowel::II => "ई",
            IndepVowel::U => "उ",
            IndepVowel::UU => "ऊ",
            IndepVowel::R => "ऋ",
            IndepVowel::RR => "ॠ",
            IndepVowel::L => "ऌ",
            IndepVowel::LL => "ॡ",
            IndepVowel::E => "ए",
            IndepVowel::AI => "ऐ",
            IndepVowel::O => "ओ",
            IndepVowel::AU => "औ",
        }
    }
}

impl AsChar for IndepVowel {
    fn as_char(&self) -> char {
        match self {
            IndepVowel::A => '\u{0905}',
            IndepVowel::AA => '\u{0906}',
            IndepVowel::I => '\u{0907}',
            IndepVowel::II => '\u{0908}',
            IndepVowel::U => '\u{0909}',
            IndepVowel::UU => '\u{090A}',
            IndepVowel::R => '\u{090B}',
            IndepVowel::RR => '\u{0960}',
            IndepVowel::L => '\u{090C}',
            IndepVowel::LL => '\u{0961}',
            IndepVowel::E => '\u{090F}',
            IndepVowel::AI => '\u{0910}',
            IndepVowel::O => '\u{0913}',
            IndepVowel::AU => '\u{0914}',
        }
    }
}

impl IndepVowel {
    fn to_vowel(&self) -> Vowel {
        match self {
            IndepVowel::A => Vowel::A,
            IndepVowel::AA => Vowel::AA,
            IndepVowel::I => Vowel::I,
            IndepVowel::II => Vowel::II,
            IndepVowel::U => Vowel::U,
            IndepVowel::UU => Vowel::UU,
            IndepVowel::R => Vowel::R,
            IndepVowel::RR => Vowel::RR,
            IndepVowel::E => Vowel::E,
            IndepVowel::AI => Vowel::AI,
            IndepVowel::O => Vowel::O,
            IndepVowel::AU => Vowel::AU,
            IndepVowel::L => unimplemented!(),
            IndepVowel::LL => unimplemented!(),
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

#[derive(Debug)]
pub(crate) enum SoundClass {
    Vowel(Vowel),
    IndepVowel(IndepVowel),
    Consonent(Consonant),
}

impl AsStr for SoundClass {
    fn as_str(&self) -> &'static str {
        match self {
            SoundClass::Vowel(v) => v.as_str(),
            SoundClass::IndepVowel(v) => v.as_str(),
            SoundClass::Consonent(c) => c.as_str(),
        }
    }
}

impl AsChar for SoundClass {
    fn as_char(&self) -> char {
        match self {
            SoundClass::Vowel(v) => v.as_char(),
            SoundClass::IndepVowel(v) => v.as_char(),
            SoundClass::Consonent(c) => unimplemented!(),
        }
    }
}
