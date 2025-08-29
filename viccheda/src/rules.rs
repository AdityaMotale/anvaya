use crate::common::{Adjuncts, AsChar, AsStr, Consonant, IndepVowel, Orthography, Vowel};
use std::{
    char,
    collections::{HashMap, HashSet},
    fmt::Write,
    sync::LazyLock,
};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug)]
enum SoundClass {
    Vowel(Vowel),
    IndepVowel(IndepVowel),
    Consonent(Consonant),
}

#[derive(Debug)]
struct Rule {
    name: &'static str,
    desc: &'static str,
    tag: &'static str,
    left: SoundClass,
    right: SoundClass,
    merged: SoundClass,
    outputs: Vec<(SoundClass, SoundClass)>,
}

struct Sandhi {
    rules: Vec<Rule>,
}

impl Sandhi {
    pub fn new() -> Self {
        Self {
            rules: Self::get_rules(),
        }
    }

    fn split(&self, morpheme: &str) -> Vec<Vec<String>> {
        fn ends_with_aa(s: &str) -> bool {
            s.ends_with(IndepVowel::AA.as_str()) || s.chars().last() == Some(Vowel::AA.as_char())
        }

        let graphemes: Vec<&str> = UnicodeSegmentation::graphemes(morpheme, true).collect();

        if graphemes.len() < 2 {
            return vec![vec![morpheme.to_string()]];
        }

        let mut results: Vec<Vec<String>> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for i in 1..graphemes.len() {
            let left = graphemes[..i].join("");
            let right = graphemes[i..].join("");

            if ends_with_aa(&left) {
                let base = left
                    .trim_end_matches(IndepVowel::AA.as_str())
                    .trim_end_matches(Vowel::AA.as_str())
                    .to_string();

                let direct_right = format!("{}{}", IndepVowel::A.as_str(), right);
                let direct = vec![base.clone(), direct_right.clone()];
                let key_direct = direct.join("|");

                if seen.insert(key_direct) {
                    results.push(direct);
                }

                let recs = self.split(&right);

                for rec in recs {
                    if rec.len() >= 2 {
                        let first_combined = format!("{}{}", IndepVowel::A.as_str(), rec[0]);
                        let mut cand = Vec::with_capacity(2 + rec.len() - 1);

                        cand.push(base.clone());
                        cand.push(first_combined);

                        for piece in rec.iter().skip(1) {
                            cand.push(piece.clone());
                        }

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

    fn get_rules() -> Vec<Rule> {
        vec![Rule {
            name: "savarṇa-dīrgha-a",
            desc: "आ  => अ + अ ",
            tag: "6.1.101",
            left: SoundClass::Vowel(Vowel::A),
            right: SoundClass::Vowel(Vowel::A),
            merged: SoundClass::Vowel(Vowel::AA),
            outputs: vec![(SoundClass::Vowel(Vowel::A), SoundClass::Vowel(Vowel::A))],
        }]
    }
}

#[cfg(test)]
mod sandhi_tests {
    use super::*;

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
