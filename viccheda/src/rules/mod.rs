use crate::{
    common::{AsChar, AsStr, SoundClass},
    logger::Logger,
    split::{Candidate, Splitter},
    tracef,
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

impl std::fmt::Display for RuleData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}]: {}", self.name, self.tag, self.desc)
    }
}

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;
    fn apply(&self, splitter: &Splitter, left: &str, right: &str) -> Option<Vec<Candidate>>;
}

pub(crate) struct RuleUtils;

impl RuleUtils {
    pub fn nfc<S: AsRef<str>>(s: S) -> String {
        s.as_ref().nfc().collect()
    }

    pub fn ends_with_soundclass(s: &str, sc: &SoundClass, logger: &Logger) -> bool {
        let s_n = Self::nfc(s);
        let grs: Vec<&str> = UnicodeSegmentation::graphemes(s_n.as_str(), true).collect();

        if grs.is_empty() {
            return false;
        }

        let last = grs.last().unwrap();

        if let Some(sc_str) = sc.as_str() {
            if Self::nfc(last) == Self::nfc(sc_str) {
                tracef!(logger, "{s} matched (as_str) w/ {:?}", sc);
                return true;
            }
        }

        if last.chars().last() == Some(sc.as_char()) {
            tracef!(logger, "{s} matched (as_char) w/ {:?}", sc);
            return true;
        }

        false
    }

    pub fn trim_end_soundclass(s: &str, sc: &SoundClass, logger: &Logger) -> Option<String> {
        let s_n = Self::nfc(s);

        let mut graphemes: Vec<String> = UnicodeSegmentation::graphemes(s_n.as_str(), true)
            .map(|g| g.to_string())
            .collect();

        if graphemes.is_empty() {
            return None;
        }

        if let Some(sc_str) = sc.as_str() {
            let sc_n = Self::nfc(sc_str);
            let sc_grs: Vec<String> = UnicodeSegmentation::graphemes(sc_n.as_str(), true)
                .map(|g| g.to_string())
                .collect();

            // ▶ Exact tail (multi-grapheme) match
            if sc_grs.len() <= graphemes.len() {
                let tail = &graphemes[graphemes.len() - sc_grs.len()..];
                let tail_joined = tail.join("");

                if tail_joined == sc_n {
                    graphemes.truncate(graphemes.len() - sc_grs.len());
                    tracef!(
                        logger,
                        "[trim_debug]: Removed following => {}:{}",
                        tail_joined,
                        sc_n
                    );

                    return Some(graphemes.join(""));
                }
            }

            // ▶ If sc_n is a single codepoint (matra), we check if it's the final
            // codepoint of the last grapheme
            if sc_grs.len() == 1 && sc_grs[0].chars().count() == 1 {
                let sc_ch = sc_grs[0].chars().next().unwrap();

                // pop last grapheme and inspect
                let last = graphemes.pop().unwrap();
                tracef!(logger, "[trim_debug:] last_grapheme='{}'", last);

                if last.chars().last() == Some(sc_ch) {
                    // remove just that final char
                    let mut last_mod = last.clone();
                    last_mod.pop();

                    if !last_mod.is_empty() {
                        graphemes.push(last_mod.clone());
                    }

                    let base = graphemes.join("");
                    tracef!(
                        logger, "[trim_debug]: matched matra inside last grapheme; new_last='{}'; base='{}'",
                        last_mod, base
                    );

                    return Some(base);
                }

                // no match, so continue w/ other fallbacks
                graphemes.push(last);
            }

            // don't return, let flow falldown
        }

        // pop last grapheme and inspect
        let last = graphemes.pop().unwrap();

        if last.chars().last() == Some(sc.as_char()) {
            let mut last_mod = last.clone();
            last_mod.pop();

            if !last_mod.is_empty() {
                graphemes.push(last_mod.clone());
            }

            let base = graphemes.join("");
            tracef!(
                logger,
                "[trim_debug]: matched char-form; new_last='{}'; base='{}'",
                last_mod,
                base
            );

            return Some(base);
        } else {
            // restore and fail
            graphemes.push(last);
        }

        None
    }
}
