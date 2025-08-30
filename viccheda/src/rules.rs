use crate::common::{AsChar, AsStr, IndepVowel, SoundClass, Vowel};
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug)]
pub struct RuleData {
    name: &'static str,
    desc: &'static str,
    tag: &'static str,
    left: SoundClass,
    right: SoundClass,
    merged: SoundClass,
}

pub trait Rule: Send + Sync {
    fn data(&self) -> &RuleData;
    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Vec<String>>>;
}

pub struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Vec<String>>> {
        let mut out = Vec::new();

        let merged_str = self.data.merged.as_str();
        let merged_char = self.data.merged.as_char();

        if !ends_with(left, &self.data.merged) {
            return None;
        }

        let base = {
            let mut b = left.trim_end_matches(merged_char);

            if let Some(str) = merged_str {
                b = b.trim_end_matches(str);
            }

            b.to_string()
        };

        let direct_right = {
            let out;

            if let Some(str) = self.data.right.as_str() {
                out = format!("{}{}", str, right);
            } else {
                out = format!("{}", right);
            }

            out
        };

        // first candidate
        out.push(vec![base.clone(), direct_right]);

        for splits in sandhi.split(right) {
            if splits.len() > 1 {
                let first_combined = {
                    let lft_data = &self.data.left;
                    let out;

                    if let Some(str) = lft_data.as_str() {
                        out = format!("{}{}", str, splits[0]);
                    } else {
                        out = format!("{}", splits[0]);
                    }

                    out
                };

                let mut cand = Vec::with_capacity(1 + splits.len());
                cand.push(base.clone());
                cand.push(first_combined);
                cand.extend(splits.into_iter().skip(1));

                out.push(cand);
            }
        }

        Some(out)
    }
}

fn ends_with(s: &str, candidate: &SoundClass) -> bool {
    if let Some(str) = candidate.as_str() {
        if s.ends_with(str) {
            return true;
        }
    }

    s.chars().last() == Some(candidate.as_char())
}

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
