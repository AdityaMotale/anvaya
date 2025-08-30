use crate::{
    common::{AsChar, AsStr, IndepVowel, SoundClass, Vowel},
    rules::{dirgha::SvarDirgha, Rule, RuleData},
};
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) struct Sandhi {
    rules: Vec<Box<dyn Rule>>,
}

impl Sandhi {
    pub fn new() -> Self {
        Self {
            rules: Self::get_rules(),
        }
    }

    pub fn split(&self, morpheme: &str) -> Vec<Vec<String>> {
        let mut results: Vec<Vec<String>> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let graphemes: Vec<&str> = UnicodeSegmentation::graphemes(morpheme, true).collect();

        // sanity check
        if graphemes.len() < 2 {
            return results;
        }

        let mut seen: HashSet<String> = HashSet::new();

        for i in 1..graphemes.len() {
            let left = graphemes[..i].join("");
            let right = graphemes[i..].join("");

            for rule in &self.rules {
                if let Some(candidates) = rule.apply(self, &left, &right) {
                    for cand in candidates {
                        let key = cand.join("|");
                        if seen.insert(key) {
                            results.push(cand);
                        }
                    }
                }
            }
        }

        results
    }

    fn get_rules() -> Vec<Box<dyn Rule>> {
        vec![
            // NOTE: अ  should not be added at the end of left candidate,
            // that's why we did't choose [IndependentVowl] for the `left`
            // window in this rule
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a1",
                    desc: "आ  => अ + अ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::A),
                    right: SoundClass::IndepVowel(IndepVowel::A),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a2",
                    desc: "आ  => आ  + अ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::AA),
                    right: SoundClass::Vowel(Vowel::A),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
        ]
    }
}

#[cfg(test)]
mod sandhi_tests {
    use super::*;

    #[test]
    fn test_split_cases_a1() {
        let sandhi = Sandhi::new();

        // each case: input word, expected candidate segmentations
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("प्रार्थी", vec![vec!["प्र", "अर्थी"]]),
            (
                "रामानुजः",
                vec![
                    vec!["र", "अमानुजः"],
                    vec!["र", "म", "अनुजः"],
                    vec!["राम", "अनुजः"],
                ],
            ),
        ];

        for (morpheme, expected) in cases {
            let candidates = sandhi.split(morpheme);

            // normalization for easier debug/comparison
            let cand_sets: Vec<String> = candidates.into_iter().map(|seg| seg.join("|")).collect();

            for exp in expected {
                let key = exp.join("|");

                assert!(
                    cand_sets.contains(&key),
                    "morpheme '{}' missing expected split {:?}",
                    morpheme,
                    exp
                );
            }
        }
    }

    #[test]
    fn test_split_cases_a2() {
        let sandhi = Sandhi::new();

        // each case: input word, expected candidate segmentations
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("प्रार्थी", vec![vec!["प्र", "अर्थी"]]),
            (
                "रामानुजः",
                vec![
                    vec!["र", "अमानुजः"],
                    vec!["र", "म", "अनुजः"],
                    vec!["राम", "अनुजः"],
                ],
            ),
        ];

        for (morpheme, expected) in cases {
            let candidates = sandhi.split(morpheme);

            // normalization for easier debug/comparison
            let cand_sets: Vec<String> = candidates.into_iter().map(|seg| seg.join("|")).collect();

            for exp in expected {
                let key = exp.join("|");

                assert!(
                    cand_sets.contains(&key),
                    "morpheme '{}' missing expected split {:?}",
                    morpheme,
                    exp
                );
            }
        }
    }
}
