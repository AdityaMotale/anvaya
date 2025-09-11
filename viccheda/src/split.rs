use crate::{
    freq_table::FreqTable,
    rules::{
        get_all_rules,
        rule::{InternalCandidate, Rule, RuleData},
    },
    Candidate,
};
use logger::{debugf, infof, tracef, warnf, Logger, PrettyVec};
use orthography::Akshara;
use std::collections::HashSet;

pub(crate) struct Splitter {
    pub logger: Logger,
    rules: Vec<Box<dyn Rule>>,
}

impl Splitter {
    const ALPHA: f64 = 0.1;
    const EPS: f64 = 1e-9;

    pub fn new(debug: bool) -> Self {
        let rules = get_all_rules();
        let logger = Logger::new(debug, "Viccheda::Splitter");

        debugf!(logger, "Init splitter w/ {} rules", rules.len());

        Self { logger, rules }
    }

    pub fn best_candidate(&self, input: &str) -> Option<Candidate> {
        let candis = Self::generate_candidates(&self, input);
        let total = FreqTable::get_total_freq();
        let vocab = FreqTable::get_vocab_size();

        // sanity check
        if candis.is_empty() {
            return None;
        }

        let mut best_candi: Option<Candidate> = None;

        for c in candis {
            // sanity check
            if c.splits.len() != 2 {
                warnf!(self.logger, "Generated splits are invalid (!= 2) for {c}");
                continue;
            }

            let (left, right) = (&c.splits[0], &c.splits[1]);

            let count1 = FreqTable::get(left).unwrap_or(0);
            let count2 = FreqTable::get(right).unwrap_or(0);

            let logp1 = Self::log_prob(count1, total, Self::ALPHA, vocab);
            let logp2 = Self::log_prob(count2, total, Self::ALPHA, vocab);

            let score = logp1 + logp2;

            // to debug each candidate score count
            tracef!(
                self.logger,
                "cand {} -> counts: ({}, {}), logp: ({:.6}, {:.6}), score: {:.6}",
                c,
                count1,
                count2,
                logp1,
                logp2,
                score
            );

            // we start by setting the first candi as the best one
            if best_candi.is_none() {
                best_candi = Some(Candidate {
                    rule: Some(c.rule),
                    score: Some(score),
                    splits: c.splits,
                });

                continue;
            }

            let old_c = best_candi.as_ref().unwrap();
            let old_score = {
                if let None = old_c.score {
                    continue;
                }

                old_c.score.unwrap()
            };

            // if current score is better then prev
            if score > old_score + Self::EPS {
                best_candi = Some(Candidate {
                    rule: Some(c.rule),
                    score: Some(score),
                    splits: c.splits,
                });

                continue;
            }

            // if scores are tied, we must choose a better option
            if (old_score - score).abs() <= Self::EPS {
                let old_left = &old_c.splits[0];
                let old_right = &old_c.splits[1];

                let raw_f1 = count1 as f64;
                let raw_f2 = count2 as f64;

                let old_raw_f1 = FreqTable::get(old_left).unwrap_or(0) as f64;
                let old_raw_f2 = FreqTable::get(old_right).unwrap_or(0) as f64;

                let both_in = raw_f1 > 0.0 && raw_f2 > 0.0;
                let old_both_in = old_raw_f1 > 0.0 && old_raw_f2 > 0.0;

                // ▶ TIE-BREAKER 1: Both windows (RHS & LHS) are present in [FREQ_TABLE]
                if both_in && !old_both_in {
                    best_candi = Some(Candidate {
                        rule: Some(c.rule),
                        score: Some(score),
                        splits: c.splits,
                    });

                    continue;
                }

                // If both windows of prev's are present and new's aren't,
                // the prev is the better candidate
                if old_both_in && !both_in {
                    continue;
                }

                // ▶ TIE-BREAKER 2: Higher frequency product
                // (i.e. the more the frequency of windows the better)
                let new_freq_prod = raw_f1 * raw_f2;
                let old_freq_prod = old_raw_f1 * old_raw_f2;

                if new_freq_prod > old_freq_prod {
                    best_candi = Some(Candidate {
                        rule: Some(c.rule),
                        score: Some(score),
                        splits: c.splits,
                    });

                    continue;
                }

                // current best is a better candi
                if old_freq_prod > new_freq_prod {
                    continue;
                }

                // ▶ TIE-BREAKER 3 [EXP]: prefer earlier split
                // (i.e. prefer smaller left window)
                let left_len = left.chars().count();
                let old_left_len = old_left.chars().count();

                if left_len < old_left_len {
                    best_candi = Some(Candidate {
                        rule: Some(c.rule),
                        score: Some(score),
                        splits: c.splits,
                    });

                    continue;
                }

                // otherwise we keep the current best
            }
        }

        if let Some(candi) = &best_candi {
            infof!(
                self.logger,
                "Final candi is {} w/ score of {}",
                InternalCandidate {
                    splits: candi.splits.clone(),
                    rule: candi.rule.clone().unwrap_or(RuleData::default())
                },
                candi.score.unwrap_or(0.0),
            );
        }

        best_candi
    }

    #[inline]
    fn log_prob(count: usize, total: usize, alpha: f64, vocab: usize) -> f64 {
        let num = (count as f64) + alpha;
        let denom = (total as f64) + alpha * (vocab as f64);
        (num / denom).ln()
    }

    fn generate_candidates(&self, morpheme: &str) -> Vec<InternalCandidate> {
        let mut results: Vec<InternalCandidate> = Vec::new();
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
