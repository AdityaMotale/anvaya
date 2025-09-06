use crate::{rules::trim_sound_with_akshara, split::Splitter};
use logger::{debugf, errorf, Logger, PrettyVec};
use orthography::{Akshara, AsChar, AsIter, AsStr, Consonant, IndependentVowel, SoundClass, Vowel};
use std::collections::HashSet;

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;

    fn apply(
        &self,
        left: &str,
        right: &str,
        logger: &Logger,
        sp: Option<&(Akshara, bool)>,
    ) -> Option<Vec<Candidate>> {
        if let Some(candi) = generic_apply(&self.data(), left, right, sp, logger) {
            return Some(vec![candi]);
        }

        None
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuleData {
    pub name: &'static str,
    pub desc: &'static str,
    pub tag: &'static str,
    pub left: Akshara,
    pub right: Akshara,
    pub merged: Akshara,
    pub special_sequence: Option<Vec<(Akshara, bool)>>,
}

impl std::fmt::Display for RuleData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}]: {}", self.name, self.tag, self.desc)
    }
}

pub(crate) trait RuleGroup {
    fn rules() -> Vec<Box<dyn Rule>>;
}

/// A generic apply function for the base logic for applying rules
fn generic_apply(
    rule_data: &RuleData,
    left: &str,
    right: &str,
    sp: Option<&(Akshara, bool)>,
    logger: &Logger,
) -> Option<Candidate> {
    // a kind of priority list for possibel merges
    let mut merge_candidates: Vec<Akshara> = Vec::with_capacity(2);
    let mut special_merged_opt: Option<Akshara> = None;

    // first merge_candidate
    if let Some((aksh, _)) = sp {
        let mut combined_vec = rule_data.merged.0.clone();
        combined_vec.extend(aksh.0.clone());

        let special_merged = Akshara(combined_vec);

        if special_merged != rule_data.merged {
            special_merged_opt = Some(special_merged.clone());
            merge_candidates.push(special_merged);
        }
    }

    // second merge_candidate
    merge_candidates.push(rule_data.merged.clone());

    let mut left_base_opt: Option<String> = None;
    let mut special_removed = false;

    for sound in &merge_candidates {
        if let Some(base) = trim_sound_with_akshara(&left, sound, logger) {
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

    let left_candidate = match rule_data.left.as_str() {
        Some(s) => format!("{left_base}{s}"),
        None => left_base,
    };

    let right_candidate = match rule_data.right.as_str() {
        Some(s) => {
            if let Some((aksh, to_add)) = sp {
                if *to_add && aksh.as_str().is_some() && special_removed {
                    format!("{s}{}{right}", aksh.as_str().unwrap())
                } else {
                    format!("{s}{right}")
                }
            } else {
                format!("{s}{right}")
            }
        }

        None => right.to_string(),
    };

    Some(Candidate::new(
        vec![left_candidate, right_candidate],
        rule_data.to_owned(),
    ))
}

fn trim_sound_from_left(sound: &str) -> String {
    let chrs: Vec<char> = sound.chars().collect();

    let mut valid_chars: Vec<char> = IndependentVowel::as_iter().map(|v| v.as_char()).collect();
    valid_chars.extend(Consonant::as_iter().map(|c| c.as_char()));

    let mut index = 0usize;

    for c in &chrs {
        if valid_chars.contains(c) {
            break;
        }

        index += 1;
    }

    chrs[index..].iter().collect()
}

pub(crate) struct BaseRule(pub RuleData);

impl Rule for BaseRule {
    #[inline]
    fn data(&self) -> &RuleData {
        &self.0
    }
}

pub(crate) struct AllKindRule {
    pub kind: SoundClass,
    pub data: RuleData,
}

impl Rule for AllKindRule {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(
        &self,
        left: &str,
        right: &str,
        logger: &Logger,
        sp: Option<&(Akshara, bool)>,
    ) -> Option<Vec<Candidate>> {
        let mut candidates = Vec::new();

        let right_candi_list: Vec<Option<&'static str>> = match self.kind {
            SoundClass::AllVowel => IndependentVowel::as_iter()
                .into_iter()
                .map(|v| v.as_str())
                .collect(),
            SoundClass::AllConsonant => Consonant::as_iter()
                .into_iter()
                .map(|c| c.as_str())
                .collect(),
            // sanity check
            _ => {
                let msg = format!(
                    "SoundClass {:?} is not allowed for `AllKindRule`",
                    &self.kind
                );

                debug_assert!(false, "{msg}");

                // safety for prod
                errorf!(logger, "{msg}");
                return None;
            }
        };

        // a kind of priority list for possibel merges
        let mut merge_candidates: Vec<Akshara> = Vec::with_capacity(2);
        let mut special_merged_opt: Option<Akshara> = None;

        // first merge_candidate
        if let Some((aksh, _)) = sp {
            let mut combined_vec = self.data.merged.0.clone();
            combined_vec.extend(aksh.0.clone());

            let special_merged = Akshara(combined_vec);

            if special_merged != self.data.merged {
                special_merged_opt = Some(special_merged.clone());
                merge_candidates.push(special_merged);
            }
        }

        // second merge_candidate
        merge_candidates.push(self.data.merged.clone());

        let mut left_base_opt: Option<String> = None;
        let mut special_removed = false;

        for sound in &merge_candidates {
            if let Some(base) = trim_sound_with_akshara(&left, sound, logger) {
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

        let left_candidate = match self.data.left.as_str() {
            Some(s) => format!("{left_base}{s}"),
            None => left_base,
        };

        for rc in right_candi_list {
            if let Some(right_candi) = rc {
                let trimmed_right = trim_sound_from_left(right);

                let right_candidate = if let Some((aksh, to_add)) = sp {
                    if *to_add && aksh.as_str().is_some() && special_removed {
                        format!("{right_candi}{}{trimmed_right}", aksh.as_str().unwrap())
                    } else {
                        format!("{right_candi}{trimmed_right}")
                    }
                } else {
                    format!("{right_candi}{trimmed_right}")
                };

                candidates.push(Candidate::new(
                    vec![left_candidate.clone(), right_candidate],
                    self.data.clone(),
                ));
            }
        }

        if candidates.is_empty() {
            None
        } else {
            Some(candidates)
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub splits: Vec<String>,
    pub rule: RuleData,
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateList<'a>(pub &'a [Candidate]);

impl Candidate {
    pub fn new(splits: Vec<String>, rule: RuleData) -> Self {
        Self { splits, rule }
    }
}

impl std::fmt::Display for Candidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs: Vec<&str> = self.splits.iter().map(String::as_str).collect();
        let splits = PrettyVec(strs);

        write!(f, "{:?} -> {}", splits, self.rule)
    }
}

impl<'a> std::fmt::Display for CandidateList<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n");

        for c in self.0 {
            writeln!(f, "{}", c)?;
        }

        Ok(())
    }
}
