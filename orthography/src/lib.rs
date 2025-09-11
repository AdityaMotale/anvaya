use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub trait AsStr {
    fn as_str(&self) -> Option<&'static str>;
}

pub trait AsChar {
    fn as_char(&self) -> char;
}

pub trait AsIter {
    fn as_iter() -> impl Iterator<Item = Self>;
}

pub trait FromStr: Sized + AsIter + AsStr + AsChar {
    fn from_str(inp: &str) -> Option<Self> {
        for item in Self::as_iter() {
            if let Some(str) = item.as_str() {
                if str == inp {
                    return Some(item);
                }
            }

            if item.as_char().to_string() == inp {
                return Some(item);
            }
        }

        None
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, EnumIter)]
pub enum Adjuncts {
    ANUSVARA,
    VISARGA,
    VIRAMA,
    AVAGRAHA,
    CHANDRABINDU,
}

impl AsStr for Adjuncts {
    #[inline]
    fn as_str(&self) -> Option<&'static str> {
        match self {
            Adjuncts::ANUSVARA => Some("ं"),
            Adjuncts::VISARGA => Some("ः"),
            Adjuncts::VIRAMA => Some("्"),
            Adjuncts::AVAGRAHA => Some("ऽ"),
            Adjuncts::CHANDRABINDU => Some("ँ"),
        }
    }
}

impl AsChar for Adjuncts {
    #[inline]
    fn as_char(&self) -> char {
        match self {
            Adjuncts::ANUSVARA => '\u{0902}',
            Adjuncts::VISARGA => '\u{0903}',
            Adjuncts::VIRAMA => '\u{094D}',
            Adjuncts::AVAGRAHA => '\u{093D}',
            Adjuncts::CHANDRABINDU => '\u{0901}',
        }
    }
}

impl AsIter for Adjuncts {
    #[inline]
    fn as_iter() -> impl Iterator<Item = Self> {
        Self::iter()
    }
}

impl FromStr for Adjuncts {}

#[test]
fn adjuncts_as_char_matches_as_str() {
    for a in Adjuncts::as_iter() {
        if let Some(s) = a.as_str() {
            assert_eq!(a.as_char(), s.chars().next().unwrap());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum Vowel {
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
    #[inline]
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
    #[inline]
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

impl AsIter for Vowel {
    #[inline]
    fn as_iter() -> impl Iterator<Item = Vowel> {
        Vowel::iter()
    }
}

impl FromStr for Vowel {}

impl Vowel {
    #[inline]
    pub const fn to_independent(&self) -> IndependentVowel {
        match self {
            Vowel::A => IndependentVowel::A,
            Vowel::AA => IndependentVowel::AA,
            Vowel::I => IndependentVowel::I,
            Vowel::II => IndependentVowel::II,
            Vowel::U => IndependentVowel::U,
            Vowel::UU => IndependentVowel::UU,
            Vowel::R => IndependentVowel::R,
            Vowel::RR => IndependentVowel::RR,
            Vowel::E => IndependentVowel::E,
            Vowel::AI => IndependentVowel::AI,
            Vowel::O => IndependentVowel::O,
            Vowel::AU => IndependentVowel::AU,
        }
    }
}

#[test]
fn vowels_as_char_matches_as_str() {
    for v in Vowel::as_iter() {
        if let Some(s) = v.as_str() {
            assert_eq!(v.as_char(), s.chars().next().unwrap());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum IndependentVowel {
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

impl AsStr for IndependentVowel {
    #[inline]
    fn as_str(&self) -> Option<&'static str> {
        match self {
            IndependentVowel::A => Some("अ"),
            IndependentVowel::AA => Some("आ"),
            IndependentVowel::I => Some("इ"),
            IndependentVowel::II => Some("ई"),
            IndependentVowel::U => Some("उ"),
            IndependentVowel::UU => Some("ऊ"),
            IndependentVowel::R => Some("ऋ"),
            IndependentVowel::RR => Some("ॠ"),
            IndependentVowel::L => Some("ऌ"),
            IndependentVowel::LL => Some("ॡ"),
            IndependentVowel::E => Some("ए"),
            IndependentVowel::AI => Some("ऐ"),
            IndependentVowel::O => Some("ओ"),
            IndependentVowel::AU => Some("औ"),
        }
    }
}

impl AsChar for IndependentVowel {
    #[inline]
    fn as_char(&self) -> char {
        match self {
            IndependentVowel::A => '\u{0905}',
            IndependentVowel::AA => '\u{0906}',
            IndependentVowel::I => '\u{0907}',
            IndependentVowel::II => '\u{0908}',
            IndependentVowel::U => '\u{0909}',
            IndependentVowel::UU => '\u{090A}',
            IndependentVowel::R => '\u{090B}',
            IndependentVowel::RR => '\u{0960}',
            IndependentVowel::L => '\u{090C}',
            IndependentVowel::LL => '\u{0961}',
            IndependentVowel::E => '\u{090F}',
            IndependentVowel::AI => '\u{0910}',
            IndependentVowel::O => '\u{0913}',
            IndependentVowel::AU => '\u{0914}',
        }
    }
}

impl AsIter for IndependentVowel {
    #[inline]
    fn as_iter() -> impl Iterator<Item = Self> {
        Self::iter()
    }
}

impl FromStr for IndependentVowel {}

impl IndependentVowel {
    #[inline]
    pub const fn to_vowel(&self) -> Option<Vowel> {
        match self {
            IndependentVowel::A => Some(Vowel::A),
            IndependentVowel::AA => Some(Vowel::AA),
            IndependentVowel::I => Some(Vowel::I),
            IndependentVowel::II => Some(Vowel::II),
            IndependentVowel::U => Some(Vowel::U),
            IndependentVowel::UU => Some(Vowel::UU),
            IndependentVowel::R => Some(Vowel::R),
            IndependentVowel::RR => Some(Vowel::RR),
            IndependentVowel::E => Some(Vowel::E),
            IndependentVowel::AI => Some(Vowel::AI),
            IndependentVowel::O => Some(Vowel::O),
            IndependentVowel::AU => Some(Vowel::AU),
            IndependentVowel::L => None,
            IndependentVowel::LL => None,
        }
    }
}

#[test]
fn indep_vowels_as_char_matches_as_str() {
    for v in IndependentVowel::as_iter() {
        if let Some(s) = v.as_str() {
            assert_eq!(v.as_char(), s.chars().next().unwrap());
        }
    }
}

#[test]
fn vowel_indep_roundtrip() {
    for v in Vowel::as_iter() {
        let indep = v.to_independent();
        assert_eq!(indep.to_vowel().unwrap(), v);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum Consonant {
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
    #[inline]
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

impl AsChar for Consonant {
    #[inline]
    fn as_char(&self) -> char {
        match self {
            // Gutturals
            Consonant::Ka => '\u{0915}',
            Consonant::Kha => '\u{0916}',
            Consonant::Ga => '\u{0917}',
            Consonant::Gha => '\u{0918}',
            Consonant::Nga => '\u{0919}',

            // Palatals
            Consonant::Cha => '\u{091A}',
            Consonant::Chha => '\u{091B}',
            Consonant::Ja => '\u{091C}',
            Consonant::Jha => '\u{091D}',
            Consonant::Nya => '\u{091E}',

            // Retroflex (cerebral)
            Consonant::Tta => '\u{091F}',
            Consonant::Ttha => '\u{0920}',
            Consonant::Dda => '\u{0921}',
            Consonant::Ddha => '\u{0922}',
            Consonant::Nna => '\u{0923}',

            // Dentals
            Consonant::Ta => '\u{0924}',
            Consonant::Tha => '\u{0925}',
            Consonant::Da => '\u{0926}',
            Consonant::Dha => '\u{0927}',
            Consonant::Na => '\u{0928}',

            // Labials
            Consonant::Pa => '\u{092A}',
            Consonant::Pha => '\u{092B}',
            Consonant::Ba => '\u{092C}',
            Consonant::Bha => '\u{092D}',
            Consonant::Ma => '\u{092E}',

            // Semivowels
            Consonant::Ya => '\u{092F}',
            Consonant::Ra => '\u{0930}',
            Consonant::La => '\u{0932}',
            Consonant::Va => '\u{0935}',

            // Sibilants
            Consonant::Sha => '\u{0936}',
            Consonant::Ssa => '\u{0937}',
            Consonant::Sa => '\u{0938}',

            // Aspirate
            Consonant::Ha => '\u{0939}',
        }
    }
}

impl AsIter for Consonant {
    #[inline]
    fn as_iter() -> impl Iterator<Item = Self> {
        Self::iter()
    }
}

impl FromStr for Consonant {}

#[test]
fn consonant_as_char_matches_as_str() {
    for c in Consonant::as_iter() {
        let s = c.as_str().expect("consonants must have as_str");
        assert_eq!(c.as_char(), s.chars().next().unwrap());
    }
}

#[test]
fn no_duplicate_consonant_chars() {
    let mut seen = std::collections::HashSet::new();

    for c in Consonant::as_iter() {
        assert!(seen.insert(c.as_char()), "Duplicate char: {:?}", c);
    }
}

pub const ASPIRATE: [Consonant; 1] = [Consonant::Ha];

pub const SIBILANTS: [Consonant; 3] = [Consonant::Sha, Consonant::Ssa, Consonant::Sa];

pub const SEMIVOWELS: [Consonant; 4] = [Consonant::Ya, Consonant::Ra, Consonant::La, Consonant::Va];

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

#[test]
fn all_chars_in_devanagari_block() {
    fn check(c: char) {
        let cp = c as u32;

        assert!(
            (0x0900..=0x097F).contains(&cp),
            "Non-Devanagari char: U+{:04X}",
            cp
        );
    }

    for v in Vowel::iter() {
        check(v.as_char());
    }

    for v in IndependentVowel::iter() {
        check(v.as_char());
    }

    for c in Consonant::iter() {
        check(c.as_char());
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SoundClass {
    Vowel(Vowel),
    IndependentVowel(IndependentVowel),
    Consonant(Consonant),
    Adjuncts(Adjuncts),
    AllVowel,
    AllConsonant,
}

impl AsStr for SoundClass {
    #[inline]
    fn as_str(&self) -> Option<&'static str> {
        match self {
            SoundClass::Vowel(v) => v.as_str(),
            SoundClass::IndependentVowel(v) => v.as_str(),
            SoundClass::Consonant(c) => c.as_str(),
            SoundClass::Adjuncts(a) => a.as_str(),
            _ => None,
        }
    }
}

impl AsChar for SoundClass {
    #[inline]
    fn as_char(&self) -> char {
        match self {
            SoundClass::Vowel(v) => v.as_char(),
            SoundClass::IndependentVowel(v) => v.as_char(),
            SoundClass::Consonant(c) => c.as_char(),
            SoundClass::Adjuncts(a) => a.as_char(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Akshara(pub Vec<SoundClass>);

impl std::fmt::Display for Akshara {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let chrs: String = self.0.iter().map(|s| s.as_char()).collect();
        let strs: String = self
            .0
            .iter()
            .map(|s| s.as_str().unwrap_or("<UNK>"))
            .collect();

        write!(f, " {strs} ( {chrs} )")
    }
}

impl Akshara {
    pub fn as_str(&self) -> Option<String> {
        let mut out = String::new();

        for c in &self.0 {
            match c.as_str() {
                Some(s) => out.push_str(s),
                None => return None,
            }
        }

        Some(out)
    }
}

pub type SpecialAkshara = (Akshara, bool);

/// get the NFC form of the word
pub fn to_nfc(word: &str) -> String {
    word.nfc().collect()
}

/// get char clusters w/ unicode segmentation by grouping chrs together
pub fn to_unicode_segmentation(word: &str) -> Vec<String> {
    UnicodeSegmentation::graphemes(word, true)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Returns true if the given string ends with exactly म्
fn ends_with_standalone_ma_virama(input: &str) -> bool {
    let segs = to_unicode_segmentation(input);

    if let Some(last) = segs.last() {
        last == "म्"
    } else {
        false
    }
}

/// Sanitize a Sanskrit word to Devnagari Sanskrit
///
/// ## Sanitization Process,
///
///   ▶ Normalize chars to NFC
///   ▶ Replace [Anusvara] w/ [Indep Vowel A + Anusvara] iff only if at start
///   ▶ Replace [Vowel] with [Indep Vowel] if any, iff only at the start
///   ▶ Remove any dangeling [Vowels] and [Adjuncts] from the starting sequence
///   ▶ Remove [Visarga] or [Consonant Ma] followed by [Virama] if present at the end
///   ▶ Build back the string w/ Unicode Segmentation
///
/// ## Purpose
///
/// The purpose of this function to get the word into
/// its base form, and nothing else. If we do not, the word
/// dictionary is highly inflated and are many potential
/// representation possible for a single word. To create
/// accurate sandhi splits and avoid dictionary enflation for
/// the tokenizer, we just chop of the word to try to get
/// its base form!
pub fn sanitize(word: &str) -> String {
    // sanity check
    if word.is_empty() || word.len() < 2 {
        return word.to_string();
    }

    let input = to_nfc(word);
    let mut chrs: Vec<String> = input.chars().map(|c| c.to_string()).collect();

    // sanitize start
    loop {
        let ch = match chrs.first() {
            Some(c) => &c.clone(),
            None => break,
        };

        if let Some(_) = Consonant::from_str(ch) {
            break;
        }

        if let Some(_) = IndependentVowel::from_str(ch) {
            break;
        }

        if let Some(ad) = Adjuncts::from_str(ch) {
            // if we found anusvara, we add Independent A
            // otherwise we remove the Adjunct
            if ad == Adjuncts::ANUSVARA {
                chrs.insert(0, IndependentVowel::A.as_char().to_string());
            } else {
                chrs.remove(0);
            }

            break;
        }

        // NOTE: We must only repalce vowel to indep at the start, not at
        // end or middle
        //
        // NOTE: This change is protected by above checks, which breaks
        // the iteration if a valid character is found in the sequence
        if let Some(v) = Vowel::from_str(ch) {
            let indep = v.to_independent();
            chrs[0] = indep.as_char().to_string();

            break;
        }
    }

    // sanitize end
    //
    // ▶ remove anusvara if is at end
    // ▶ remove visarga if is at end
    //
    // NOTE: In sandhi this is replaced with [Visarga],
    // but we normalize words (remove visarga at end)
    if let Some(last) = chrs.last() {
        if let Some(adj) = Adjuncts::from_str(last) {
            match adj {
                Adjuncts::ANUSVARA | Adjuncts::VISARGA => {
                    chrs.pop();
                }

                // remove `म्` from the end if present
                //
                // NOTE: `म्` is written at the end of the word as a
                // replacement for [Anusvara], so to get the word
                // into the base form, we need to remove it
                Adjuncts::VIRAMA => {
                    if ends_with_standalone_ma_virama(&chrs.join("")) {
                        // remove [Virama]
                        chrs.pop();

                        // remove [Consonant Ma]
                        chrs.pop();
                    }
                }
                _ => {}
            }
        }
    }

    to_unicode_segmentation(&chrs.join("")).join("")
}

#[cfg(test)]
mod sanitize_tests {
    use super::*;

    /// Utility: easier to spot issues in diffs
    fn s(input: &str, expected: &str) {
        assert_eq!(sanitize(input), expected, "input: {}", input);
    }

    #[test]
    fn empty_input() {
        s("", "");
    }

    #[test]
    fn normal_word_unchanged() {
        s("प्रथम", "प्रथम");
    }

    #[test]
    fn leading_indep_vowel_unchanged() {
        s("अग्नि", "अग्नि");
    }

    #[test]
    fn leading_dependent_vowel_promoted() {
        // starting with ि + क should be promoted to इ + क
        s("िक", "इक");
    }

    #[test]
    fn leading_non_anusvara_adjunct_removed() {
        // starting with visarga before a consonant should just be dropped
        s("ःकर्म", "कर्म");
    }

    #[test]
    fn trailing_anusvara_removed() {
        s("कर्मं", "कर्म");
    }

    #[test]
    fn trailing_visarga_removed() {
        s("कर्मः", "कर्म");
    }

    #[test]
    fn mid_word_anusvara_not_removed() {
        // "संपूर्ण" should remain unchanged; anusvara only dropped at end
        s("संपूर्ण", "संपूर्ण");
    }

    #[test]
    fn mid_word_visarga_not_removed() {
        // "अःपुरुषः" should keep internal visarga, drop only trailing
        s("अःपुरुषः", "अःपुरुष");
    }
}
