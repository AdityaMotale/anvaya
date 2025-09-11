use crate::rules::RuleUtils;
use logger::{errorf, Logger, PrettyVec};
use orthography::{
    sanitize, Akshara, AsIter, AsStr, Consonant, IndependentVowel, SoundClass, SpecialAkshara,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Candidate {
    pub splits: Vec<String>,
    pub rule: RuleData,
}

#[derive(Debug, Clone)]
#[allow(unused)]
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
        writeln!(f, "\n").expect("Unable to write to stdout");

        for c in self.0 {
            writeln!(f, "{}", c)?;
        }

        Ok(())
    }
}

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;

    fn apply(
        &self,
        left: &str,
        right: &str,
        logger: &Logger,
        sp: Option<&SpecialAkshara>,
    ) -> Option<Vec<Candidate>> {
        if let Some(candi) = RuleUtils::generic_apply(&self.data(), left, right, sp, logger) {
            return Some(vec![candi]);
        }

        None
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RuleData {
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
        sp: Option<&SpecialAkshara>,
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

        let (left_base, special_removed) =
            match RuleUtils::trim_left_base(left, &self.data.merged, sp, logger) {
                Some((lb, sr)) => (lb, sr),
                None => return None,
            };

        let left_candidate = match self.data.left.as_str() {
            Some(s) => format!("{left_base}{s}"),
            None => left_base,
        };
        let sanitized_left = sanitize(&left_candidate);

        for rc in right_candi_list {
            if let Some(right_candi) = rc {
                let trimmed_right = RuleUtils::trim_sound_from_left(right);
                let right_candidate =
                    RuleUtils::create_right_candi(sp, right_candi, &trimmed_right, special_removed);

                let sanitized_right = sanitize(&right_candidate);

                candidates.push(Candidate::new(
                    vec![sanitized_left.clone(), sanitized_right],
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

pub(crate) struct MultiOptRule {
    pub merged_list: Vec<Akshara>,
    pub swap_list: Vec<Akshara>,
    pub data: RuleData,
}

impl Rule for MultiOptRule {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(
        &self,
        left: &str,
        right: &str,
        logger: &Logger,
        sp: Option<&SpecialAkshara>,
    ) -> Option<Vec<Candidate>> {
        // sanity check
        assert!(self.merged_list.len() == self.swap_list.len());

        let mut candidates = Vec::new();

        for (merged, left_candi) in self.merged_list.iter().zip(&self.swap_list) {
            let (left_base, special_removed) =
                match RuleUtils::trim_left_base(left, merged, sp, logger) {
                    Some((lb, sr)) => (lb, sr),
                    None => continue,
                };

            let left_candidate = match left_candi.as_str() {
                Some(s) => format!("{left_base}{s}"),
                None => left_base,
            };

            let right_candidate = match self.data.right.as_str() {
                Some(s) => RuleUtils::create_right_candi(sp, &s, right, special_removed),
                None => right.to_string(),
            };

            let sanitized_left = sanitize(&left_candidate);
            let sanitized_right = sanitize(&right_candidate);

            candidates.push(Candidate::new(
                vec![sanitized_left, sanitized_right],
                self.data.clone(),
            ));
        }

        Some(candidates)
    }
}
