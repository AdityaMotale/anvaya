use crate::{
    freq_table::FreqTable,
    rules::{
        get_all_rules,
        rule::{Candidate, Rule},
    },
};
use logger::{debugf, infof, tracef, warnf, Logger, PrettyVec};
use orthography::{sanitize, Akshara};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

pub(crate) struct Splitter {
    pub logger: Logger,
    rules: Vec<Box<dyn Rule>>,
}

impl Splitter {
    // a const to be used if word is not present in freq table
    const RULE_COST: f64 = 0.1;

    const EPS: f64 = 1e-9;

    pub fn new(debug: bool) -> Self {
        let rules = get_all_rules();
        let logger = Logger::new(debug, "Viccheda::Splitter");

        debugf!(logger, "Init splitter w/ {} rules", rules.len());

        Self { logger, rules }
    }

    pub fn best_candidate(&self, input: &str) -> Option<(Candidate, f64)> {
        let candis = Self::generate_candidates(&self, input);

        // sanity check
        if candis.is_empty() {
            return None;
        }

        // choose the first as default
        let mut best_candi: Option<(Candidate, f64)> = None;

        for c in candis {
            if let Some(candi) = &best_candi {
                debugf!(
                    self.logger,
                    "Current candidate is {} & w/ score of {}",
                    candi.0,
                    candi.1
                );
            }

            // sanity check
            if c.splits.len() != 2 {
                warnf!(self.logger, "Generated splits are invalid (!= 2) for {c}");
                continue;
            }

            let (left, right) = (&c.splits[0], &c.splits[1]);

            let raw_f1 = FreqTable::get(left).unwrap_or(0) as f64 + Self::RULE_COST;
            let raw_f2 = FreqTable::get(right).unwrap_or(0) as f64 + Self::RULE_COST;
            let score = raw_f1.ln() + raw_f2.ln();

            // if no candidate is yet selected
            if let None = best_candi {
                best_candi = Some((c, score));
                continue;
            }

            let (ref old_c, old_score) = best_candi.clone().unwrap();

            // if current score is better then prev
            if old_score > score + Self::EPS {
                best_candi = Some((c, score));
                continue;
            }

            // if scores are tied, we must choose a better option
            if (old_score - score).abs() <= Self::EPS {
                let old_left = &old_c.splits[0];
                let old_right = &old_c.splits[1];

                let old_raw_f1 = FreqTable::get(old_left).unwrap_or(0) as f64;
                let old_raw_f2 = FreqTable::get(old_right).unwrap_or(0) as f64;

                let both_in = raw_f1 > 0.0 && raw_f2 > 0.0;
                let old_both_in = old_raw_f1 > 0.0 && old_raw_f2 > 0.0;

                // ▶ TIE-BREAKER 1: Both windows (RHS & LHS) are present in [FREQ_TABLE]
                if both_in && !old_both_in {
                    best_candi = Some((c, score));
                    continue;
                }

                // If both windows of prev's are present and new's are
                // not the prev is the better candidate
                if old_both_in && !both_in {
                    continue;
                }

                // ▶ TIE-BREAKER 2: Higher frequency product (i.e. the more the frequency
                // of windows the better)
                let new_freq = raw_f1 * raw_f2;
                let old_freq = old_raw_f1 * old_raw_f2;

                if new_freq > old_freq + Self::EPS {
                    best_candi = Some((c, score));
                    continue;
                }

                // old candi is better
                if new_freq < old_freq - Self::EPS {
                    continue;
                }

                // ▶ TIE-BREAKER 3 [EXP]: prefer earlier split (i.e. prefer smaller left window)
                let left_len = left.chars().count();
                let old_left_len = old_left.chars().count();

                if left_len < old_left_len {
                    best_candi = Some((c, score));
                    continue;
                }

                // otherwise keep the current candi and continue
            }
        }

        infof!(
            self.logger,
            "Final candi is {} w/ score of {}",
            best_candi.clone().unwrap().0,
            best_candi.clone().unwrap().1
        );

        best_candi
    }

    fn generate_candidates(&self, morpheme: &str) -> Vec<Candidate> {
        let mut results: Vec<Candidate> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let charset: Vec<char> = morpheme.chars().collect();

        debugf!(
            self.logger,
            "\n{morpheme} => {:?}",
            PrettyVec(
                charset
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
            ),
        );

        // sanity check (returns empty vec)
        if charset.len() < 2 {
            return results;
        }

        for i in 1..=charset.len() {
            let left: String = charset[..i].iter().collect();
            let right: String = charset[i..].iter().collect();

            for rule in &self.rules {
                // iterator for special cases
                let custom_iter: Box<dyn Iterator<Item = Option<&(Akshara, bool)>>> =
                    match &rule.data().special_sequence {
                        Some(seq) => Box::new(seq.iter().map(Some)),
                        None => Box::new(std::iter::once(None)),
                    };

                for special_seq in custom_iter {
                    if let Some(candidates) = rule.apply(&left, &right, &self.logger, special_seq) {
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

        debugf!(
            self.logger,
            "Generated candidates for [{morpheme}] => {}",
            results
                .iter()
                .map(|r| format!("\n{r}"))
                .collect::<Vec<_>>()
                .join("")
        );

        results
    }
}

#[cfg(test)]
pub(crate) fn test_sandhi_cases(cases: Vec<(&str, Vec<Vec<&str>>)>, debug: bool) {
    use orthography::{sanitize, AsStr};

    let splitter = Splitter::new(debug);

    for (morpheme, expected_list) in cases {
        let candidates = splitter.generate_candidates(morpheme);

        // normalized keys for contains-check
        let cand_keys: Vec<String> = candidates
            .iter()
            .map(|cand| {
                cand.splits
                    .iter()
                    .map(|c| sanitize(c))
                    .collect::<Vec<String>>()
                    .join("|")
            })
            .collect();

        for expected in expected_list {
            use orthography::sanitize;

            let expected_key = expected
                .iter()
                .map(|c| sanitize(*c))
                .collect::<Vec<String>>()
                .join("|");

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
