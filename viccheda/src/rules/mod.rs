use crate::{
    common::{AsChar, AsStr, SoundClass},
    split::{Candidate, Sandhi},
};
use unicode_normalization::UnicodeNormalization;

pub(crate) mod dirgha;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RuleData {
    pub name: &'static str,
    pub desc: &'static str,
    pub tag: &'static str,
    pub left: SoundClass,
    pub right: SoundClass,
    pub merged: SoundClass,
}

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;
    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Candidate>>;
}

pub(crate) fn ends_with(s: &str, candidate: &SoundClass) -> bool {
    if let Some(str) = candidate.as_str() {
        if s.ends_with(str) {
            return true;
        }
    }

    s.chars().last() == Some(candidate.as_char())
}

pub(crate) fn nfc<S: AsRef<str>>(s: S) -> String {
    s.as_ref().nfc().collect()
}
