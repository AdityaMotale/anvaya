use crate::{rules::trim_sound_with_akshara, split::Splitter};
use logger::PrettyVec;
use orthography::Akshara;

pub(crate) trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;

    fn apply(
        &self,
        splitter: &Splitter,
        left: &str,
        right: &str,
        sp: Option<&(Akshara, bool)>,
    ) -> Option<Vec<Candidate>>;
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

pub(crate) struct BaseRule(pub RuleData);

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
impl Rule for BaseRule {
    #[inline]
    fn data(&self) -> &RuleData {
        &self.0
    }

    fn apply(
        &self,
        splitter: &Splitter,
        left: &str,
        right: &str,
        sp: Option<&(Akshara, bool)>,
    ) -> Option<Vec<Candidate>> {
        let mut out = Vec::new();
        let rule_data = self.data();

        // a kind of priority list for possibel merges
        let mut merge_candidates: Vec<Akshara> = Vec::with_capacity(2);

        // first merge_candidate
        if let Some((aksh, _)) = sp {
            let mut combined_vec = rule_data.merged.0.clone();
            combined_vec.extend(aksh.0.clone());

            let special_merged = Akshara(combined_vec);

            if special_merged != rule_data.merged {
                merge_candidates.push(special_merged);
            }
        }

        // second merge_candidate
        merge_candidates.push(rule_data.merged.clone());

        let left_base_opt = merge_candidates
            .iter()
            .find_map(|sound| trim_sound_with_akshara(&left, &sound, &splitter.logger));

        let left_base = match left_base_opt {
            Some(b) => b,
            None => return None,
        };

        let right_candidate = match rule_data.right.as_str() {
            Some(s) => {
                if let Some((aksh, to_add)) = sp {
                    if *to_add && aksh.as_str().is_some() {
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

        // first candidate (left + right_candidate)
        out.push(Candidate::new(
            vec![left.to_string(), right_candidate.clone()],
            rule_data.clone(),
        ));

        // second candidate (left_trimmed + right_candidate),
        let left_candidate = match rule_data.left.as_str() {
            Some(s) => format!("{left_base}{s}"),
            None => left_base.clone(),
        };

        if !left_candidate.is_empty() {
            out.push(Candidate::new(
                vec![left_candidate, right_candidate.clone()],
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
