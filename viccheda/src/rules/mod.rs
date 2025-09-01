use crate::split::{Candidate, Splitter};
use logger::{debugf, tracef, Logger};
use orthography::{Akshara, AsChar, AsStr};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) mod dirgha;

#[derive(Debug, Clone)]
pub(crate) struct RuleData {
    pub name: &'static str,
    pub desc: &'static str,
    pub tag: &'static str,
    pub left: Akshara,
    pub right: Akshara,
    pub merged: Akshara,
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
    pub fn if_sound_ends_with_akshara(sound: &str, akshara: &Akshara, logger: &Logger) -> bool {
        assert!(akshara.0.len() != 0);

        // fast fail for invalid input
        if sound.is_empty() {
            return false;
        }

        tracef!(
            logger,
            "[Soundclass match] matching sound=`{}` with akshara {}",
            sound,
            akshara
        );

        // Work on a mutable tail (bytes) and consume suffixes/last-chars from right→left.
        // We keep it as a String so we can truncate by byte index safely.
        let mut tail = sound.to_string();

        for sc in akshara.0.iter().rev() {
            if tail.is_empty() {
                tracef!(
                    logger,
                    "[Soundclass Match (FAIL)] tail empty but still have {:?} to match",
                    sc.as_str()
                );

                return false;
            }

            if let Some(expected) = sc.as_str() {
                if tail.ends_with(expected) {
                    let new_len = tail.len() - expected.len();
                    tail.truncate(new_len);

                    tracef!(
                        logger,
                        "[Soundclass match] matched as_str `{}` for {:?}; new tail=`{}`",
                        expected,
                        sc.as_str(),
                        tail
                    );

                    continue;
                } else {
                    tracef!(
                        logger,
                        "[Soundclass match FAIL] tail `{}` does not end with `{}` for {:?}",
                        tail,
                        expected,
                        sc.as_str(),
                    );

                    return false;
                }
            } else {
                // no as_str -> match single char (as_char)
                let last_char = tail.chars().rev().next();

                if last_char == Some(sc.as_char()) {
                    // remove last char safely by truncating at its starting byte index
                    let cut_at = tail
                        .char_indices()
                        .rev()
                        .next()
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    tail.truncate(cut_at);

                    tracef!(
                        logger,
                        "[Soundclass match] matched as_char U+{:04X} for {:?}; new tail=`{}`",
                        sc.as_char() as u32,
                        sc.as_str(),
                        tail
                    );

                    continue;
                } else {
                    tracef!(
                    logger,
                    "[Soundclass match FAIL] tail last char ({:?}) != as_char U+{:04X} for {:?}",
                    last_char,
                    sc.as_char() as u32,
                    sc.as_str(),
                );

                    return false;
                }
            }
        }

        tracef!(
            logger,
            "[Soundclass Match (OK)] all soundclasses matched for `{}`",
            sound
        );

        true
    }

    pub fn trim_end_soundclass(sound: &str, akshara: &Akshara, logger: &Logger) -> Option<String> {
        let mut graphemes: Vec<String> = UnicodeSegmentation::graphemes(sound, true)
            .map(|g| g.to_string())
            .collect();

        if graphemes.is_empty() {
            return None;
        }

        // if let Some(sc_str) = sc.as_str() {
        //     let sc_grs: Vec<String> = UnicodeSegmentation::graphemes(sc_str, true)
        //         .map(|g| g.to_string())
        //         .collect();

        //     // ▶ Exact tail (multi-grapheme) match
        //     if sc_grs.len() <= graphemes.len() {
        //         let tail = &graphemes[graphemes.len() - sc_grs.len()..];
        //         let tail_joined = tail.join("");

        //         if tail_joined == sc_str {
        //             graphemes.truncate(graphemes.len() - sc_grs.len());
        //             tracef!(
        //                 logger,
        //                 "[trim_debug]: Removed following => {}:{}",
        //                 tail_joined,
        //                 sc_str
        //             );

        //             return Some(graphemes.join(""));
        //         }
        //     }

        //     // ▶ If sc_n is a single codepoint (matra), we check if it's the final
        //     // codepoint of the last grapheme
        //     if sc_grs.len() == 1 && sc_grs[0].chars().count() == 1 {
        //         let sc_ch = sc_grs[0].chars().next().unwrap();

        //         // pop last grapheme and inspect
        //         let last = graphemes.pop().unwrap();
        //         tracef!(logger, "[trim_debug:] last_grapheme='{}'", last);

        //         if last.chars().last() == Some(sc_ch) {
        //             // remove just that final char
        //             let mut last_mod = last.clone();
        //             last_mod.pop();

        //             if !last_mod.is_empty() {
        //                 graphemes.push(last_mod.clone());
        //             }

        //             let base = graphemes.join("");
        //             tracef!(
        //                 logger, "[trim_debug]: matched matra inside last grapheme; new_last='{}'; base='{}'",
        //                 last_mod, base
        //             );

        //             return Some(base);
        //         }

        //         // no match, so continue w/ other fallbacks
        //         graphemes.push(last);
        //     }

        //     // don't return, let the flow falldown
        // }

        // // pop last grapheme and inspect
        // let last = graphemes.pop().unwrap();

        // if last.chars().last() == Some(sc.as_char()) {
        //     let mut last_mod = last.clone();
        //     last_mod.pop();

        //     if !last_mod.is_empty() {
        //         graphemes.push(last_mod.clone());
        //     }

        //     let base = graphemes.join("");
        //     tracef!(
        //         logger,
        //         "[trim_debug]: matched char-form; new_last='{}'; base='{}'",
        //         last_mod,
        //         base
        //     );

        //     return Some(base);
        // } else {
        //     // restore and fail
        //     graphemes.push(last);
        // }

        None
    }
}
