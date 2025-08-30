use crate::{
    common::{AsChar, AsStr, SoundClass},
    split::Sandhi,
};

pub(crate) mod dirgha;

#[derive(Debug)]
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
    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Vec<String>>>;
}

pub(crate) fn ends_with(s: &str, candidate: &SoundClass) -> bool {
    if let Some(str) = candidate.as_str() {
        if s.ends_with(str) {
            return true;
        }
    }

    s.chars().last() == Some(candidate.as_char())
}
