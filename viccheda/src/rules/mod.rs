use crate::{
    common::{AsChar, AsStr, SoundClass},
    split::{Candidate, Sandhi},
};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

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

pub(crate) fn nfc<S: AsRef<str>>(s: S) -> String {
    s.as_ref().nfc().collect()
}

pub(crate) fn ends_with_soundclass(s: &str, sc: &SoundClass) -> bool {
    let s_n = nfc(s);
    let grs: Vec<&str> = UnicodeSegmentation::graphemes(s_n.as_str(), true).collect();

    if grs.is_empty() {
        return false;
    }

    let last = grs.last().unwrap();

    if let Some(sc_str) = sc.as_str() {
        if nfc(last) == nfc(sc_str) {
            return true;
        }
    }

    if last.chars().last() == Some(sc.as_char()) {
        return true;
    }

    false
}

pub(crate) fn trim_end_soundclass(s: &str, sc: &SoundClass) -> String {
    let mut s_n = nfc(s);
    let mut grs: Vec<&str> = UnicodeSegmentation::graphemes(s_n.as_str(), true).collect();

    if grs.is_empty() {
        return s_n;
    }

    if let Some(sc_str) = sc.as_str() {
        let sc_n = nfc(sc_str);

        if grs.last().map(|g| nfc(*g)) == Some(sc_n.clone()) {
            grs.pop();

            return grs.join("");
        }

        let sc_grs: Vec<&str> = UnicodeSegmentation::graphemes(sc_n.as_str(), true).collect();

        if sc_grs.len() > 1 && sc_grs.len() <= grs.len() {
            let tail = &grs[grs.len() - sc_grs.len()..];
            let tail_joined = tail.join("");

            if nfc(tail_joined) == sc_n {
                grs.truncate(grs.len() - sc_grs.len());

                return grs.join("");
            }
        }
    }

    if let Some(last) = grs.last() {
        if last.chars().last() == Some(sc.as_char()) {
            grs.pop();

            return grs.join("");
        }
    }

    grs.join("")
}
