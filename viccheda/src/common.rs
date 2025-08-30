use std::{collections::HashMap, sync::LazyLock};

pub(crate) trait AsStr {
    fn as_str(&self) -> Option<&'static str>;
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
    fn as_str(&self) -> Option<&'static str> {
        match self {
            Adjuncts::ANUSVARA => Some("ं"),
            Adjuncts::VISARGA => Some("ः"),
            Adjuncts::VIRAMA => Some("्"),
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
    fn as_str(&self) -> Option<&'static str> {
        match self {
            Vowel::A => None,
            Vowel::AA => Some("ा"),
            Vowel::I => Some("ि"),
            Vowel::II => Some("ी"),
            Vowel::U => Some("ु"),
            Vowel::UU => Some("ू"),
            Vowel::R => Some("ृ"),
            Vowel::RR => Some("ॄ"),
            Vowel::E => Some("े"),
            Vowel::AI => Some("ै"),
            Vowel::O => Some("ो"),
            Vowel::AU => Some("ौ"),
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
    fn as_str(&self) -> Option<&'static str> {
        match self {
            IndepVowel::A => Some("अ"),
            IndepVowel::AA => Some("आ"),
            IndepVowel::I => Some("इ"),
            IndepVowel::II => Some("ई"),
            IndepVowel::U => Some("उ"),
            IndepVowel::UU => Some("ऊ"),
            IndepVowel::R => Some("ऋ"),
            IndepVowel::RR => Some("ॠ"),
            IndepVowel::L => Some("ऌ"),
            IndepVowel::LL => Some("ॡ"),
            IndepVowel::E => Some("ए"),
            IndepVowel::AI => Some("ऐ"),
            IndepVowel::O => Some("ओ"),
            IndepVowel::AU => Some("औ"),
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
    fn to_vowel(&self) -> Option<Vowel> {
        match self {
            IndepVowel::A => Some(Vowel::A),
            IndepVowel::AA => Some(Vowel::AA),
            IndepVowel::I => Some(Vowel::I),
            IndepVowel::II => Some(Vowel::II),
            IndepVowel::U => Some(Vowel::U),
            IndepVowel::UU => Some(Vowel::UU),
            IndepVowel::R => Some(Vowel::R),
            IndepVowel::RR => Some(Vowel::RR),
            IndepVowel::E => Some(Vowel::E),
            IndepVowel::AI => Some(Vowel::AI),
            IndepVowel::O => Some(Vowel::O),
            IndepVowel::AU => Some(Vowel::AU),
            IndepVowel::L => None,
            IndepVowel::LL => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
    fn as_str(&self) -> Option<&'static str> {
        match self {
            // Gutturals
            Consonant::Ka => Some("क"),
            Consonant::Kha => Some("ख"),
            Consonant::Ga => Some("ग"),
            Consonant::Gha => Some("घ"),
            Consonant::Nga => Some("ङ"),

            // Palatals
            Consonant::Cha => Some("च"),
            Consonant::Chha => Some("छ"),
            Consonant::Ja => Some("ज"),
            Consonant::Jha => Some("झ"),
            Consonant::Nya => Some("ञ"),

            // Retroflex
            Consonant::Tta => Some("ट"),
            Consonant::Ttha => Some("ठ"),
            Consonant::Dda => Some("ड"),
            Consonant::Ddha => Some("ढ"),
            Consonant::Nna => Some("ण"),

            // Dentals
            Consonant::Ta => Some("त"),
            Consonant::Tha => Some("थ"),
            Consonant::Da => Some("द"),
            Consonant::Dha => Some("ध"),
            Consonant::Na => Some("न"),

            // Labials
            Consonant::Pa => Some("प"),
            Consonant::Pha => Some("फ"),
            Consonant::Ba => Some("ब"),
            Consonant::Bha => Some("भ"),
            Consonant::Ma => Some("म"),

            // Semi-vowels
            Consonant::Ya => Some("य"),
            Consonant::Ra => Some("र"),
            Consonant::La => Some("ल"),
            Consonant::Va => Some("व"),

            // Sibilants + Aspirate
            Consonant::Sha => Some("श"),
            Consonant::Ssa => Some("ष"),
            Consonant::Sa => Some("स"),
            Consonant::Ha => Some("ह"),
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum SoundClass {
    Vowel(Vowel),
    IndepVowel(IndepVowel),
    Consonent(Consonant),
}

impl AsStr for SoundClass {
    fn as_str(&self) -> Option<&'static str> {
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
