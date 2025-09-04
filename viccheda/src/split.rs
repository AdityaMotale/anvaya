use crate::rules::{get_all_rules, Rule, RuleData};
use logger::{debugf, tracef, Logger, PrettyVec};
use orthography::Akshara;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

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

pub(crate) struct Splitter {
    pub logger: Logger,
    rules: Vec<Box<dyn Rule>>,
}

impl Splitter {
    pub fn new(debug: bool) -> Self {
        Self {
            rules: get_all_rules(),
            logger: Logger::new(debug, "Viccheda::Splitter"),
        }
    }

    fn nfc<S: AsRef<str>>(s: S) -> String {
        s.as_ref().nfc().collect()
    }

    pub fn candidates(&self, input: &str) -> Option<Vec<Candidate>> {
        let morpheme = Self::nfc(input);
        let mut results: Vec<Candidate> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let graphemes: Vec<&str> =
            UnicodeSegmentation::graphemes(morpheme.as_str(), true).collect();

        debugf!(
            self.logger,
            "{morpheme} => \n{:?}\n{:?}",
            PrettyVec(graphemes.clone()),
            PrettyVec(
                morpheme
                    .chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
            ),
        );

        // sanity check
        if graphemes.len() < 2 {
            return None;
        }

        let mut seen: HashSet<String> = HashSet::new();

        for i in 1..graphemes.len() {
            let left = graphemes[..i].join("");
            let right = graphemes[i..].join("");

            for rule in &self.rules {
                let special_sequence = &rule.data().special_sequence;

                let custom_iter: Box<dyn Iterator<Item = Option<&(Akshara, bool)>>> =
                    match special_sequence {
                        Some(seq) => Box::new(seq.iter().map(Some)),
                        None => Box::new(std::iter::once(None)),
                    };

                for sp in custom_iter {
                    if let Some(candidates) = rule.apply(self, &left, &right, sp) {
                        tracef!(
                            &self.logger,
                            "Rule '{}' applied to {morpheme}",
                            &rule.data().name
                        );

                        for cand in candidates {
                            let key = cand.splits.join("|");

                            if seen.insert(key) {
                                results.push(cand);
                            }
                        }
                    }
                }
            }
        }

        tracef!(
            self.logger,
            "Generate candidates, {morpheme} => {}",
            CandidateList(&results)
        );

        Some(results)
    }
}

#[cfg(test)]
pub(crate) fn test_sandhi_cases(cases: Vec<(&str, Vec<Vec<&str>>)>, debug: bool) {
    use orthography::AsStr;

    let splitter = Splitter::new(debug);

    for (morpheme, expected_list) in cases {
        let candidates = match splitter.candidates(morpheme) {
            Some(c) => c,
            None => {
                panic!("split returned None for morpheme '{}'", morpheme);
            }
        };

        // normalized keys for contains-check
        let cand_keys: Vec<String> = candidates
            .iter()
            .map(|cand| cand.splits.join("|"))
            .collect();

        for expected in expected_list {
            let expected_key = expected.join("|");

            if !cand_keys.contains(&expected_key) {
                let mut debug = String::new();

                for (i, cand) in candidates.iter().enumerate() {
                    let joined = cand.splits.join(" | ");

                    let rule = &cand.rule;

                    let rule_name = rule.name;
                    let rule_tag = rule.tag;

                    let left_sc: String = rule
                        .left
                        .0
                        .iter()
                        .map(|sc| sc.as_str().unwrap_or("<UNK>"))
                        .collect();

                    let right_sc: String = rule
                        .right
                        .0
                        .iter()
                        .map(|sc| sc.as_str().unwrap_or("<UNK>"))
                        .collect();

                    let merged_sc: String = rule
                        .right
                        .0
                        .iter()
                        .map(|sc| sc.as_str().unwrap_or("<UNK>"))
                        .collect();

                    debug.push_str(&format!(
                             "candidate {}: {}\n  rule: {} (tag {})\n  left/right/merged: {} / {} / {}\n\n",
                             i, joined, rule_name, rule_tag, left_sc, right_sc, merged_sc
                        ));
                }

                panic!(
                        "morpheme '{}' missing expected split [{}]\nExpected key: '{}'\nActual candidates (normalized):\n{}\n",
                        morpheme,
                        expected.join(", "),
                        expected_key,
                        debug,
                    );
            }
        }
    }
}
