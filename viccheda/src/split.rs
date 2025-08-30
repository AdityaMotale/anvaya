use crate::{
    common::{AsChar, AsStr, IndepVowel, SoundClass, Vowel},
    rules::{dirgha::SvarDirgha, Rule, RuleData},
};
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) struct Sandhi {
    rules: Vec<Box<dyn Rule>>,
}

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub splits: Vec<String>,
    pub rule: Option<RuleData>,
}

impl Candidate {
    pub fn new(splits: Vec<String>, rule: Option<RuleData>) -> Self {
        Self { splits, rule }
    }
}

impl Sandhi {
    pub fn new() -> Self {
        Self {
            rules: Self::get_rules(),
        }
    }

    pub fn split(&self, morpheme: &str) -> Option<Vec<Candidate>> {
        let mut results: Vec<Candidate> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let graphemes: Vec<&str> = UnicodeSegmentation::graphemes(morpheme, true).collect();

        // sanity check
        if graphemes.len() < 2 {
            return None;
        }

        let mut seen: HashSet<String> = HashSet::new();

        for i in 1..graphemes.len() {
            let left = graphemes[..i].join("");
            let right = graphemes[i..].join("");

            for rule in &self.rules {
                if let Some(candidates) = rule.apply(self, &left, &right) {
                    for cand in candidates {
                        let key = cand.splits.join("|");

                        if seen.insert(key) {
                            results.push(cand);
                        }
                    }
                }
            }
        }

        Some(results)
    }

    fn get_rules() -> Vec<Box<dyn Rule>> {
        let mut rules = Vec::new();
        rules.extend(SvarDirgha::rules());

        rules
    }
}

#[cfg(test)]
mod sandhi_tests {
    use super::*;

    #[test]
    fn test_split_cases() {
        let sandhi = Sandhi::new();

        // each case: input word, expected candidate segmentations (pieces)
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("प्रार्थी", vec![vec!["प्र", "अर्थी"]]),
            ("श्रद्धास्ति", vec![vec!["श्रद्धा", "अस्ति"]]),
            (
                "रामानुजः",
                vec![
                    vec!["र", "अमानुजः"],
                    vec!["र", "म", "अनुजः"],
                    vec!["राम", "अनुजः"],
                ],
            ),
        ];

        for (morpheme, expected_list) in cases {
            // split now returns Option<Vec<Candidate>>
            let candidates = match sandhi.split(morpheme) {
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

                        if let Some(rule) = cand.rule {
                            let rule_name = rule.name;
                            let rule_tag = rule.tag;
                            let left_sc = rule.left.as_str().unwrap_or("<none>");
                            let right_sc = rule.right.as_str().unwrap_or("<none>");
                            let merged_sc = rule.merged.as_str().unwrap_or("<none>");

                            debug.push_str(&format!(
                            "candidate {}: {}\n  rule: {} (tag {})\n  left/right/merged: {} / {} / {}\n\n",
                            i, joined, rule_name, rule_tag, left_sc, right_sc, merged_sc
                        ));
                        }
                    }

                    panic!(
                        "morpheme '{}' missing expected split [{}]\nExpected key: '{}'\nActual candidates (normalized):\n{}\n",
                        morpheme,
                        expected.join(", "),
                        expected_key,
                        debug
                    );
                }
            }
        }
    }
}
