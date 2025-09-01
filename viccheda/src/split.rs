use crate::rules::{dirgha::SvarDirgha, Rule, RuleData};
use logger::{tracef, Logger, PrettyVec};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub splits: Vec<String>,
    pub rule: Option<RuleData>,
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateList<'a>(pub &'a [Candidate]);

impl Candidate {
    pub fn new(splits: Vec<String>, rule: Option<RuleData>) -> Self {
        Self { splits, rule }
    }
}

impl std::fmt::Display for Candidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs: Vec<&str> = self.splits.iter().map(String::as_str).collect();
        let splits = PrettyVec(strs);

        match &self.rule {
            Some(rule) => write!(f, "{:?} -> {}", splits, rule),
            None => write!(f, "{:?}", splits),
        }
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
            rules: Self::get_rules(),
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

        tracef!(
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
                if let Some(candidates) = rule.apply(self, &left, &right) {
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

        tracef!(
            self.logger,
            "Generate candidates, {morpheme} => {}",
            CandidateList(&results)
        );
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
    use env_logger;
    use once_cell::sync::OnceCell;
    use orthography::AsStr;

    static INIT: OnceCell<()> = OnceCell::new();

    fn init_logger() {
        INIT.get_or_init(|| {
            let _ = env_logger::builder().is_test(true).try_init();
        });
    }

    fn run_sandhi_cases(cases: Vec<(&str, Vec<Vec<&str>>)>, debug: bool) {
        init_logger();
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

                        if let Some(rule) = &cand.rule {
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

    mod swar {
        use super::*;

        mod dirgha {
            use super::*;

            #[test]
            fn aa_to_a_a_debug() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> =
                    vec![("परास्तः", vec![vec!["परा", "अस्तः"]])];

                run_sandhi_cases(cases, true);
            }

            #[test]
            fn aa_to_a_a() {
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
                ];

                run_sandhi_cases(cases, false);
            }

            #[test]
            fn aa_to_a_a_anusvara() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
                    ("सर्वांगीणः", vec![vec!["सर्व", "अंगीणः"]]),
                    ("मूल्यांकनः", vec![vec!["मूल्य", "अंकनः"]]),
                    ("देहांतः", vec![vec!["देह", "अंतः"]]),
                    ("सुखांतः", vec![vec!["सुख", "अन्तः"]]),
                    ("दीक्षांतः", vec![vec!["दीक्षा", "अंतः"]]),
                    ("रेखांकितः", vec![vec!["रेखा", "अंकितः"]]),
                    ("गीतांजलिः", vec![vec!["गीत", "अंजलिः"]]),
                ];

                run_sandhi_cases(cases, true);
            }

            #[test]
            fn ii_to_i_i_debug() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("श्रीशः", vec![vec!["श्री", "ईशः"]])];

                run_sandhi_cases(cases, true);
            }

            #[test]
            fn ii_to_i_i() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
                    ("श्रीशः", vec![vec!["श्री", "ईशः"]]),
                    ("गौरीशः", vec![vec!["गौरी", "ईशः"]]),
                    ("नदीशः", vec![vec!["नदी", "ईशः"]]),
                    ("रजनीशः", vec![vec!["रजनी", "ईशः"]]),
                    ("महीशः", vec![vec!["मही", "ईशः"]]),
                    ("पृथ्वीश्वरः", vec![vec!["पृथ्वी", "ईश्वरः"]]),
                    ("नारीच्छा", vec![vec!["नारी", "इच्छा"]]),
                    ("महतीच्छा", vec![vec!["महती", "इच्छा"]]),
                    ("नारीश्वरः", vec![vec!["नारी", "ईश्वरः"]]),
                    ("गिरीशः", vec![vec!["गिरि", "ईशः"]]),
                    ("हरीशः", vec![vec!["हरि", "ईशः"]]),
                    ("कवीशः", vec![vec!["कवि", "ईशः"]]),
                    ("कपीशः", vec![vec!["कपि", "ईशः"]]),
                    ("इतीवः", vec![vec!["इति", "इवः"]]),
                    ("अतीवः", vec![vec!["अति", "इवः"]]),
                    ("रवीन्द्रः", vec![vec!["रवि", "इन्द्रः"]]),
                    ("मुनीन्द्रः", vec![vec!["मुनि", "इन्द्रः"]]),
                    ("कवीन्द्रः", vec![vec!["कवि", "इन्द्रः"]]),
                    ("फणीन्द्रः", vec![vec!["फणी", "इन्द्रः"]]),
                    ("गिरीन्द्रः", vec![vec!["गिरि", "इन्द्रः"]]),
                    ("शचीन्द्रः", vec![vec!["शचि", "इन्द्रः"]]),
                    ("यतीन्द्रः", vec![vec!["यति", "इन्द्रः"]]),
                    ("अभीष्टः", vec![vec!["अभि", "इष्टः"]]),
                    ("मुनीश्वरः", vec![vec!["मुनि", "ईश्वरः"]]),
                    ("प्रतीक्षा", vec![vec!["प्रति", "ईक्षा"]]),
                    ("परीक्षा", vec![vec!["परि", "ईक्षा"]]),
                    ("अधीक्षकः", vec![vec!["अधि", "ईक्षकः"]]),
                    ("वीक्षणः", vec![vec!["वि", "ईक्षणः"]]),
                    ("प्रतीतः", vec![vec!["प्रति", "इतः"]]),
                    ("परीक्षितः", vec![vec!["परि", "ईक्षितः"]]),
                    ("परीक्षकः", vec![vec!["परि", "ईक्षकः"]]),
                ];

                run_sandhi_cases(cases, false);
            }

            #[test]
            fn uu_to_u_u_debug() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> =
                    vec![("विष्णूदयः", vec![vec!["विष्णु", "उदयः"]])];

                run_sandhi_cases(cases, true);
            }

            #[test]
            fn uu_to_u_u() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
                    ("विष्णूदयः", vec![vec!["विष्णु", "उदयः"]]),
                    ("भानूदयः", vec![vec!["भानु", "उदयः"]]),
                    ("भानूष्मा", vec![vec!["भानु", "ऊष्मा"]]),
                    ("साधूपदेशः", vec![vec!["साधु", "उपदेशः"]]),
                    ("गुरूपदेशः", vec![vec!["गुरु", "उपदेशः"]]),
                    ("वधूत्सवः", vec![vec!["वधु", "उत्सवः"]]),
                    ("मधूत्तमम्", vec![vec!["मधु", "उत्तमम्"]]),
                    ("लघूत्तमम्", vec![vec!["लघु", "उत्तमम्"]]),
                    ("विधूर्ध्वम्", vec![vec!["विधु", "उर्ध्वम्"]]),
                    ("तरूर्ध्वम्", vec![vec!["तरु", "उर्ध्वम्"]]),
                    ("वधूर्मिः", vec![vec!["वधू", "उर्मिः"]]),
                    ("लघूर्मिः", vec![vec!["लघु", "उर्मिः"]]),
                    ("सिँधूर्मिः", vec![vec!["सिँधु", "उर्मिः"]]),
                    ("सूक्तिः", vec![vec!["सु", "उक्तिः"]]),
                    ("वधूक्तिः", vec![vec!["वधू", "उक्तिः"]]),
                    ("मंजूषा", vec![vec!["मंजु", "उषा"]]),
                    ("अनूदितः", vec![vec!["अनु", "उदितः"]]),
                ];

                run_sandhi_cases(cases, false);
            }

            #[test]
            fn rr_to_r_r_debug() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("पितृृणम्", vec![vec!["पितृ", "ऋणम्"]])];

                run_sandhi_cases(cases, true);
            }

            #[test]
            fn rr_to_r_r() {
                let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
                    ("होतृृकारः", vec![vec!["होतृ", "ऋकारः"]]),
                    ("पितृृणम्", vec![vec!["पितृ", "ऋणम्"]]),
                    ("मातृृणम्", vec![vec!["मातृ", "ऋणम्"]]),
                    ("कर्तृृणम्", vec![vec!["कर्तृ", "ऋणम्"]]),
                    ("कर्तृृणि", vec![vec!["कर्तृ", "ऋणि"]]),
                    ("कर्तृृद्धिः", vec![vec!["कर्तृ", "ऋद्धि"]]),
                    ("धातृृकारः", vec![vec!["धातृ", "ऋकारः"]]),
                    ("भर्तृृद्धिः", vec![vec!["भर्तृ", "ऋद्धि"]]),
                    ("होतृृषिः", vec![vec!["होतृ", "ऋषिः"]]),
                ];

                run_sandhi_cases(cases, false);
            }
        }
    }
}
