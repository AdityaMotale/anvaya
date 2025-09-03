use crate::split::{Candidate, Splitter};
use logger::{debugf, errorf, tracef, Logger};
use orthography::{Akshara, AsChar, AsStr};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

mod svar;

pub(crate) fn get_all_rules() -> Vec<Box<dyn Rule>> {
    let mut all_rules = Vec::new();

    all_rules.extend(svar::dirgha::SvarDirgha::rules());
    all_rules.extend(svar::guna::SvarGuna::rules());

    all_rules
}

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;

    fn apply(&self, splitter: &Splitter, left: &str, right: &str) -> Option<Vec<Candidate>>;
}

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

pub(crate) trait RuleGroup {
    fn rules() -> Vec<Box<dyn Rule>>;
}

struct BaseRule(pub RuleData);

impl Rule for BaseRule {
    #[inline]
    fn data(&self) -> &RuleData {
        &self.0
    }

    fn apply(&self, splitter: &Splitter, left: &str, right: &str) -> Option<Vec<Candidate>> {
        let mut out = Vec::new();
        let rule_data = self.data();

        let left_base =
            match RuleUtils::trim_sound_with_akshara(&left, &rule_data.merged, &splitter.logger) {
                Some(b) => b,
                None => return None,
            };

        // sanity check
        if left_base.is_empty() {
            debugf!(
                &splitter.logger,
                "[(BASE) Rule Apply] left_base {left_base} is empty after trimming"
            );

            return None;
        };

        let right_candidate = match rule_data.right.as_str() {
            Some(s) => format!("{s}{right}"),
            None => right.to_string(),
        };

        // first candidate (left + right_candidate)
        out.push(Candidate::new(
            vec![left.to_string(), right_candidate.clone()],
            rule_data.clone(),
        ));

        // second candidate (left_trimmed + right_candidate),
        // only if left_base != left
        if left_base != left {
            let left_candidate = match rule_data.left.as_str() {
                Some(s) => format!("{left_base}{s}"),
                None => left_base.clone(),
            };

            out.push(Candidate::new(
                vec![left_candidate, right_candidate],
                rule_data.clone(),
            ));
        }

        // now we recursively generate candidates for the right side
        if let Some(candidates) = splitter.candidates(right) {
            for candi in candidates {
                if candi.splits.len() > 1 {
                    let first_combined = match &rule_data.left.as_str() {
                        Some(s) => format!("{s}{}", candi.splits[0]),
                        None => candi.splits[0].clone(),
                    };

                    let mut cand: Candidate = Candidate::new(
                        Vec::with_capacity(1 + candi.splits.len()),
                        rule_data.clone(),
                    );

                    cand.splits.push(left_base.clone());
                    cand.splits.push(first_combined);
                    cand.splits.extend(candi.splits.clone().into_iter().skip(1));

                    out.push(cand);
                }
            }
        }

        Some(out)
    }
}

pub(crate) struct RuleUtils;

impl RuleUtils {
    // trim input sound from right with the [Akshara], returns `None`
    // if [Akshara] does not matches w/ input sound
    pub fn trim_sound_with_akshara(
        sound: &str,
        akshara: &Akshara,
        logger: &Logger,
    ) -> Option<String> {
        // sanity check
        assert!(akshara.0.len() != 0);

        // fast fail for invalid input
        if sound.is_empty() {
            return None;
        }

        tracef!(
            logger,
            "----\n[Sound Trim] matching sound=`{}` with akshara {}",
            sound,
            akshara
        );

        // in the loop we consume suffixes/last-chars (bytes) from right -> left,
        // this way we get our trimmed sound
        let mut tail = sound.to_string();

        // we iter w/ reverse sequence cause we are matching from right -> left,
        // or end -> start
        for soundclass in akshara.0.iter().rev() {
            if tail.is_empty() {
                tracef!(
                    logger,
                    "[Sound Trim] (ERROR) tail empty but still have {} to match",
                    soundclass.as_char()
                );

                return None;
            }

            if let Some(expected_str) = soundclass.as_str() {
                if tail.ends_with(expected_str) {
                    let new_len = tail.len() - expected_str.len();
                    tail.truncate(new_len);

                    tracef!(
                        logger,
                        "[Sound Trim] (MATCH) `{}` for {} ;; new tail=`{}`",
                        expected_str,
                        soundclass.as_char(),
                        tail
                    );

                    continue;
                } else {
                    tracef!(
                        logger,
                        "[Sound Trim] (SKIP) tail `{}` does not end with `{}` for {}",
                        tail,
                        expected_str,
                        soundclass.as_char(),
                    );

                    return None;
                }
            } else {
                // NOTE: Except [Vowel::A], all sound class have string representation (as_str)
                // And [Vowel::A] must never be used in merged sequence, cause we never split on
                // single [Vowel::A] or [IndependentVowel::A]. So, we're mostly safe here ;)

                errorf!(
                    logger,
                    "[Sound Trim] {akshara} contains empty string sequence {}",
                    soundclass.as_char(),
                );

                return None;
            }
        }

        Some(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_logger;
    use orthography::{Adjuncts, Vowel};
    use orthography::{AsStr, SoundClass};

    #[test]
    fn test_trim_sound_with_sequence() {
        let log_cell = init_logger("RuleUtils (Test)");
        let logger = log_cell.get().expect("Custom Logger for test");

        let candidates: Vec<(String, &'static str, Akshara, &'static str)> = vec![
            (
                String::from("पित"),
                "पितृृ",
                Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                "[Vowel::R, Vowel::R] matching with Vowel::RR",
            ),
            (
                String::from("सर्व"),
                "सर्वां",
                Akshara(vec![
                    SoundClass::Vowel(Vowel::AA),
                    SoundClass::Adjuncts(Adjuncts::ANUSVARA),
                ]),
                "[Vowel::AA, Adjuncts::ANUSVARA] matching with [Vowel::AA, Adjuncts::ANUSVARA]",
            ),
        ];

        for (out, inp, aksh, desc) in candidates {
            let sound = RuleUtils::trim_sound_with_akshara(inp, &aksh, logger);

            assert_eq!(Some(out), sound, "Failed for {}", desc);
        }
    }
}
