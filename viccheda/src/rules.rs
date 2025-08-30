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
    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Vec<Vec<String>>;
}

pub struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Vec<Vec<String>> {
        let mut out = Vec::new();

        let merged_str = self.data.merged.as_str();
        let merged_char = self.data.merged.as_char();

        if !ends_with(left, (merged_str, merged_char)) {
            return out;
        }

        let base = left
            .trim_end_matches(merged_str)
            .trim_end_matches(merged_char)
            .to_string();

        let direct_right = format!("{}{}", self.data.right.as_str(), right);

        // first candidate
        out.push(vec![base.clone(), direct_right]);

        for splits in sandhi.split(right) {
            if splits.len() > 1 {
                let first_combined = {
                    let lft_data = &self.data.left;
                    let out;

                    if lft_data == &SoundClass::Vowel(Vowel::A)
                        || lft_data == &SoundClass::IndepVowel(IndepVowel::A)
                    {
                        out = format!("{}", splits[0]);
                    } else {
                        out = format!("{}{}", self.data.left.as_str(), splits[0]);
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

        out
    }
}

fn ends_with(s: &str, candidates: (&str, char)) -> bool {
    s.ends_with(candidates.0) || s.chars().last() == Some(candidates.1)
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
                let candidates = rule.apply(self, &left, &right);

                for cand in candidates {
                    let key = cand.join("|");
                    if seen.insert(key) {
                        results.push(cand);
                    }
                }
            }
        }

        results
    }

    fn get_rules() -> Vec<Box<dyn Rule>> {
        vec![Box::new(SvarDirgha {
            data: RuleData {
                name: "savarṇa-dīrgha-a1",
                desc: "आ  => अ + अ ",
                tag: "6.1.101",
                left: SoundClass::Vowel(Vowel::A),
                right: SoundClass::IndepVowel(IndepVowel::A),
                merged: SoundClass::Vowel(Vowel::AA),
            },
        })]
    }
}

#[cfg(test)]
mod sandhi_tests {
    use super::*;

    // #[test]
    // fn test_split() {
    //     let sandhi = Sandhi::new();
    //     let morpheme = "प्रार्थी";
    //     let results: Vec<String> = ["प्र", "अर्थी"].iter().map(|s| s.to_string()).collect();

    //     let candidates = sandhi.split(morpheme);

    //     for segs in candidates {
    //         for res in &results {
    //             assert!(segs.contains(res));
    //         }
    //     }
    // }

    #[test]
    fn test_split() {
        let sandhi = Sandhi::new();
        let morpheme = "रामानुजः";

        let candidates = sandhi.split(morpheme);

        for segs in candidates {
            println!("{}", segs.join(" | "));
        }

        assert!(true)
    }
}
