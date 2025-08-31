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
mod tests {
    use super::*;

    mod swar {
        use super::*;

        mod dirgha {
            use super::*;

            #[test]
            fn aa_to_a() {
                let sandhi = Sandhi::new();

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
                    ("शिवालयः", vec![vec!["शिव", "आलयः"]]),
                    ("विद्यालयः", vec![vec!["विद्या", "आलयः"]]),
                    ("पुस्तकालयः", vec![vec!["पुस्तक", "आलयः"]]),
                    ("हिमालयः", vec![vec!["हिम", "आलयः"]]),
                    ("कमलाकरः", vec![vec!["कमल", "आकरः"]]),
                    ("दैत्यारिः", vec![vec!["दैत्य", "अरिः"]]),
                    ("शशाङ्कः", vec![vec!["शश", "अङ्कः"]]),
                    ("गौराङ्गः", vec![vec!["गौर", "अङ्गः"]]),
                    ("रत्नाकरः", vec![vec!["रत्न", "आकरः"]]),
                    ("यथार्थः", vec![vec!["यथा", "अर्थः"]]),
                    ("विद्याभ्यासः", vec![vec!["विद्या", "अभ्यासः"]]),
                    ("विद्यार्थी", vec![vec!["विद्या", "अर्थी"]]),
                    ("परीक्षार्थी", vec![vec!["परीक्षा", "अर्थी"]]),
                    ("रामावतारः", vec![vec!["राम", "अवतारः"]]),
                    ("सूर्यास्तः", vec![vec!["सूर्य", "अस्तः"]]),
                    ("धर्मात्मा", vec![vec!["धर्म", "आत्मा"]]),
                    ("परमात्मा", vec![vec!["परम", "आत्मा"]]),
                    ("कदापि", vec![vec!["कदा", "अपि"]]),
                    ("आत्मानंदः", vec![vec!["आत्मा", "आनंदः"]]),
                    ("जन्मान्धः", vec![vec!["जन्म", "अन्धः"]]),
                    ("श्रद्धालु", vec![vec!["श्रद्धा", "आलु"]]),
                    ("सभाध्यक्षः", vec![vec!["सभा", "अध्यक्षः"]]),
                    ("पुरुषार्थः", vec![vec!["पुरुष", "अर्थः"]]),
                    ("परमार्थः", vec![vec!["परम", "अर्थः"]]),
                    ("पराधीनः", vec![vec!["पर", "अधीनः"]]),
                    ("वेदान्तः", vec![vec!["वेद", "अन्तः"]]),
                    ("सुषुप्तावस्था", vec![vec!["सुषुप्त", "अवस्था"]]),
                    ("अभयारण्यः", vec![vec!["अभय", "अरण्यः"]]),
                    ("श्रद्धानन्दः", vec![vec!["श्रद्धा", "आनन्दः"]]),
                    ("महाशयः", vec![vec!["महा", "आशयः"]]),
                    ("वार्तालापः", vec![vec!["वार्ता", "आलापः"]]),
                    ("महामात्यः", vec![vec!["महा", "अमात्यः"]]),
                    ("मुक्तावली", vec![vec!["मुक्त", "अवली"]]),
                    ("दीपावली", vec![vec!["दीप", "अवली"]]),
                    ("प्रश्नावली", vec![vec!["प्रश्न", "अवली"]]),
                    ("कृपाकांक्षी", vec![vec!["कृपा", "आकांक्षी"]]),
                    ("विस्मयादि", vec![vec!["विस्मय", "आदि"]]),
                    ("सत्याग्रहः", vec![vec!["सत्य", "आग्रहः"]]),
                    ("प्राणायामः", vec![vec!["प्राण", "आयामः"]]),
                    ("शुभारंभः", vec![vec!["शुभ", "आरंभः"]]),
                    ("मरणासन्नः", vec![vec!["मरण", "आसन्नः"]]),
                    ("शरणागतः", vec![vec!["शरण", "आगतः"]]),
                    ("नीलाकाशः", vec![vec!["नील", "आकाशः"]]),
                    ("परास्तः", vec![vec!["परा", "अस्तः"]]),
                    ("प्रधानाध्यापकः", vec![vec!["प्रधान", "अध्यापकः"]]),
                    ("विभागाध्यक्षः", vec![vec!["विभाग", "अध्यक्षः"]]),
                    // ("सर्वांगीणः", vec![vec!["सर्व", "अंगीणः"]]),
                    // ("मूल्यांकनः", vec![vec!["मूल्य", "अंकनः"]]),
                    // ("देहांतः", vec![vec!["देह", "अंतः"]]),
                    // ("सुखांतः", vec![vec!["सुख", "अन्तः"]]),
                    // ("दीक्षांतः", vec![vec!["दीक्षा", "अंतः"]]),
                    // ("रेखांकितः", vec![vec!["रेखा", "अंकितः"]]),
                    // ("गीतांजलिः", vec![vec!["गीत", "अंजलिः"]]),
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
    }
}
