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

pub(crate) fn trim_end_soundclass(s: &str, sc: &SoundClass) -> Option<String> {
    let s_n = nfc(s);
    let mut grs: Vec<String> = UnicodeSegmentation::graphemes(s_n.as_str(), true)
        .map(|g| g.to_string())
        .collect();

    if grs.is_empty() {
        return None;
    }

    eprintln!("trim_debug: input='{}' graphemes={:?}", s_n, grs);

    // If sc has a string representation (could be multi-grapheme OR single matra)
    if let Some(sc_str) = sc.as_str() {
        let sc_n = nfc(sc_str);
        let sc_grs: Vec<String> = UnicodeSegmentation::graphemes(sc_n.as_str(), true)
            .map(|g| g.to_string())
            .collect();

        eprintln!(
            "trim_debug: trying sc.as_str()='{}' -> sc_grs={:?}",
            sc_n, sc_grs
        );

        // 1) Exact tail (multi-grapheme) match
        if sc_grs.len() <= grs.len() {
            let tail = &grs[grs.len() - sc_grs.len()..];
            let tail_joined = tail.join("");
            if tail_joined == sc_n {
                grs.truncate(grs.len() - sc_grs.len());
                let base = grs.join("");
                eprintln!("trim_debug: matched exact string-form; base='{}'", base);
                return Some(base);
            }
        }

        // 2) If sc_n is a single codepoint (matra) — check if it's the final codepoint of the last grapheme
        if sc_grs.len() == 1 && sc_grs[0].chars().count() == 1 {
            let sc_ch = sc_grs[0].chars().next().unwrap();
            eprintln!(
                "trim_debug: trying single-codepoint sc '{}' inside last grapheme",
                sc_ch
            );

            // pop last grapheme and inspect
            let last = grs.pop().unwrap();
            eprintln!("trim_debug: last_grapheme='{}'", last);

            if last.chars().last() == Some(sc_ch) {
                // remove just that final char
                let mut last_mod = last.clone();
                last_mod.pop();
                if !last_mod.is_empty() {
                    grs.push(last_mod.clone());
                }
                let base = grs.join("");
                eprintln!(
                    "trim_debug: matched matra inside last grapheme; new_last='{}'; base='{}'",
                    last_mod, base
                );
                return Some(base);
            }

            // no match: push it back and continue to other fallbacks
            grs.push(last);
            eprintln!("trim_debug: single-codepoint sc not present as final codepoint");
        }

        // don't return None here — allow fallback to as_char() below
        eprintln!("trim_debug: sc.as_str() did not match tail; trying as_char fallback");
    }

    // pop last grapheme and inspect
    let last = grs.pop().unwrap();
    eprintln!(
        "trim_debug: trying sc.as_char()='{}' against last_grapheme='{}'",
        sc.as_char(),
        last
    );

    if last.chars().last() == Some(sc.as_char()) {
        let mut last_mod = last.clone();
        last_mod.pop();
        if !last_mod.is_empty() {
            grs.push(last_mod.clone());
        }
        let base = grs.join("");
        eprintln!(
            "trim_debug: matched char-form; new_last='{}'; base='{}'",
            last_mod, base
        );
        return Some(base);
    } else {
        // restore and fail
        grs.push(last);
        eprintln!("trim_debug: as_char() did not match last grapheme");
    }

    None
}
