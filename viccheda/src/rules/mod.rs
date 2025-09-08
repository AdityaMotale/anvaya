pub mod rule;
mod svar;
mod visarg;
mod vyanjan;

use std::char;

use crate::rules::rule::{Candidate, Rule, RuleData, RuleGroup};
use logger::{errorf, tracef, Logger};
use orthography::{
    Adjuncts, Akshara, AsChar, AsIter, AsStr, Consonant, FromStr, IndependentVowel, SpecialAkshara,
    Vowel,
};

pub(crate) fn get_all_rules() -> Vec<Box<dyn Rule>> {
    let mut all_rules = Vec::new();

    // svar rules
    all_rules.extend(svar::dirgha::SvarDirgha::rules());
    all_rules.extend(svar::guna::SvarGuna::rules());
    all_rules.extend(svar::vriddhi::SvarVriddhi::rules());
    all_rules.extend(svar::yan::SvarYan::rules());
    all_rules.extend(svar::ayadi::SvarAyadi::rules());
    all_rules.extend(svar::poorvaroop::SvarPoorvaroop::rules());
    all_rules.extend(svar::pararupam::SvarPararupam::rules());

    // visarga rules
    all_rules.extend(visarg::satva::VisargSatva::rules());
    all_rules.extend(visarg::shatva::VisargShatva::rules());
    all_rules.extend(visarg::rutva::VisargRutva::rules());

    // vynjan rules
    all_rules.extend(vyanjan::chhatva::VynjanChhatva::rules());
    all_rules.extend(vyanjan::shchutva::VynjanShchutva::rules());
    all_rules.extend(vyanjan::jashtva::VynjanJashtva::rules());
    all_rules.extend(vyanjan::chatrva::VynjanChatrva::rules());
    all_rules.extend(vyanjan::latva::VynjanLatva::rules());
    all_rules.extend(vyanjan::paraswarn::VynjanParaswarn::rules());
    all_rules.extend(vyanjan::anunasik::VynjanAnunasik::rules());
    all_rules.extend(vyanjan::shtutva::VynjanShtuvta::rules());

    all_rules
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
        debug_assert!(akshara.0.len() != 0, "akshara must not be empty");

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

    // trim the left base with merge candidate, in case special sequence
    // is also removed,
    //
    // NOTE: Returns `true` if special seq is removed
    pub fn trim_left_base(
        left: &str,
        merged: &Akshara,
        sp: Option<&SpecialAkshara>,
        logger: &Logger,
    ) -> Option<(String, bool)> {
        // a kind of priority list for possibel merges
        let mut merge_candidates: Vec<Akshara> = Vec::with_capacity(2);
        let mut special_merged_opt: Option<Akshara> = None;

        // first merge_candidate
        if let Some((aksh, continue_search)) = sp {
            if !continue_search {
                let mut combined_vec = merged.0.clone();
                combined_vec.extend(aksh.0.clone());

                let special_merged = Akshara(combined_vec);

                if &special_merged != merged {
                    special_merged_opt = Some(special_merged.clone());
                    merge_candidates.push(special_merged);
                }
            }
        }

        // second merge_candidate
        merge_candidates.push(merged.clone());

        let mut left_base_opt: Option<String> = None;
        let mut special_removed = false;

        for sound in &merge_candidates {
            if let Some(base) = Self::trim_sound_with_akshara(&left, sound, logger) {
                left_base_opt = Some(base);

                if let Some(ref spec) = special_merged_opt {
                    if sound == spec {
                        special_removed = true;
                    }
                }

                break;
            }
        }

        let left_base = match left_base_opt {
            Some(b) => b,
            None => return None,
        };

        Some((left_base, special_removed))
    }

    pub fn create_right_candi(
        sp: Option<&SpecialAkshara>,
        left: &str,
        right: &str,
        add_sp: bool,
    ) -> String {
        if add_sp {
            if let Some((aksh, to_add)) = sp {
                if *to_add && aksh.as_str().is_some() {
                    return format!("{left}{}{right}", aksh.as_str().unwrap());
                }
            }
        }

        format!("{left}{right}")
    }

    fn trim_sound_from_left(sound: &str) -> String {
        let chrs: Vec<char> = sound.chars().collect();

        let mut valid_chars: Vec<char> = IndependentVowel::as_iter().map(|v| v.as_char()).collect();
        valid_chars.extend(Consonant::as_iter().map(|c| c.as_char()));
        valid_chars.extend(Adjuncts::as_iter().map(|c| c.as_char()));

        let mut index = 0usize;

        for c in &chrs {
            if valid_chars.contains(c) {
                break;
            }

            index += 1;
        }

        chrs[index..].iter().collect()
    }

    pub fn sanitize_sound(sound: &str) -> String {
        // sanity check
        if sound.is_empty() {
            return String::new();
        }

        let mut chrs: Vec<String> = sound.chars().map(|c| c.to_string()).collect();
        let first_char = chrs[0].clone().to_string();

        // sanitize start
        for (i, ch) in chrs.clone().iter().enumerate() {
            if let Some(c) = Consonant::from_str(ch) {
                break;
            }

            if let Some(iv) = IndependentVowel::from_str(ch) {
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
            }

            // NOTE: We must only repalce vowel to indep at the start, not at
            // end or middle
            if let Some(v) = Vowel::from_str(ch) {
                let indep = v.to_independent();
                chrs[i] = indep.as_char().to_string();
            }
        }

        // sanitize end (remove anusvara if is at end)
        //
        // NOTE: In sandhi this is replaced with [Visarga],
        // but we normalize words (remove visarga at end)
        if let Some(last) = chrs.last() {
            if let Some(adj) = Adjuncts::from_str(last) {
                if adj == Adjuncts::ANUSVARA {
                    // removes the last element
                    chrs.pop();
                }
            }
        }

        chrs.join("")
    }

    /// A generic apply function for the base logic for applying rules
    pub fn generic_apply(
        rule_data: &RuleData,
        left: &str,
        right: &str,
        sp: Option<&SpecialAkshara>,
        logger: &Logger,
    ) -> Option<Candidate> {
        let (left_base, special_removed) =
            match Self::trim_left_base(left, &rule_data.merged, sp, logger) {
                Some((lb, sr)) => (lb, sr),
                None => return None,
            };

        let left_candidate = match rule_data.left.as_str() {
            Some(s) => format!("{left_base}{s}"),
            None => left_base,
        };

        let right_candidate = match rule_data.right.as_str() {
            Some(s) => Self::create_right_candi(sp, &s, right, special_removed),
            None => right.to_string(),
        };

        let sanitized_right = Self::sanitize_sound(&right_candidate);

        Some(Candidate::new(
            vec![left_candidate, sanitized_right],
            rule_data.to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_logger;
    use orthography::{Adjuncts, Vowel};
    use orthography::{AsStr, SoundClass};

    #[test]
    fn test_trim_sound_with_akshara() {
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

    #[test]
    fn trim_sound_with_akshara_failure() {
        let log_cell = init_logger("RuleUtils (Test)");
        let logger = log_cell.get().expect("Custom Logger for test");

        let inp = "राम";
        let aksh = Akshara(vec![SoundClass::Vowel(Vowel::II)]);
        let out = RuleUtils::trim_sound_with_akshara(inp, &aksh, logger);

        assert_eq!(out, None, "राम does not end with ई");
    }

    #[test]
    fn create_right_candi_without_special() {
        let out = RuleUtils::create_right_candi(None, "ग", "आ", false);
        assert_eq!(out, "गआ");
    }

    #[test]
    fn create_right_candi_with_special_added() {
        let sp = (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]), true);
        let out = RuleUtils::create_right_candi(Some(&sp), "राम", "आगच्छति", true);

        assert_eq!(out, "रामःआगच्छति");
    }

    #[test]
    fn trim_sound_from_left_drops_nonvalid() {
        // add garbage chars at start
        let input = "123राम";
        let trimmed = RuleUtils::trim_sound_from_left(input);

        assert_eq!(trimmed, "राम");
    }

    #[test]
    fn generic_apply_basic() {
        let log_cell = init_logger("RuleUtils (Test)");
        let logger = log_cell.get().expect("Custom Logger for test");

        let rd = RuleData {
            name: "",
            desc: "",
            tag: "",
            left: Akshara(vec![SoundClass::Consonant(Consonant::Ka)]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![SoundClass::Consonant(Consonant::Ka)]),
            special_sequence: None,
        };

        let cand = RuleUtils::generic_apply(&rd, "क", "अगच्छति", None, logger);

        assert!(cand.is_some());

        let c = cand.unwrap();
        assert_eq!(c.splits[0], "क");
        assert_eq!(c.splits[1], "अगच्छति");
    }
}
