mod cache_table;
mod freq_table;
mod rules;
mod split;

use crate::{cache_table::CacheTable, rules::rule::RuleData, split::Splitter};
use orthography::to_nfc;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub rule: Option<RuleData>,
    pub score: Option<f64>,
    pub splits: Vec<String>,
}

pub struct Viccheda {
    splitter: Splitter,
}

impl Viccheda {
    pub fn new(debug: bool) -> Self {
        Self {
            splitter: Splitter::new(debug),
        }
    }

    pub fn split(&self, word: &str) -> Option<Candidate> {
        let nfc_word = to_nfc(word);

        // sandhi cache
        if let Some(res) = CacheTable::get(&nfc_word) {
            return Some(res);
        }

        self.splitter.best_candidate(&nfc_word)
    }
}

#[cfg(test)]
pub(crate) fn init_logger(subject: &'static str) -> once_cell::sync::OnceCell<logger::Logger> {
    use env_logger;
    use logger::Logger;
    use once_cell::sync::OnceCell;

    static INIT: OnceCell<Logger> = OnceCell::new();

    INIT.get_or_init(|| {
        let _ = env_logger::builder().is_test(true).try_init();
        Logger::new(true, subject)
    });

    INIT.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use orthography::sanitize;
    use rand::Rng;

    const CACHE_FILE: &'static str = "../raw_data/cache.txt";

    fn join_parts(parts: Vec<String>) -> String {
        parts.join("|")
    }

    fn create_logger() {
        let _ = crate::init_logger("Viccheda (Test)");
    }

    fn read_candidates(txt_path: &str, n: usize) -> Vec<(String, Vec<String>)> {
        let data = std::fs::read_to_string(txt_path)
            .unwrap_or_else(|e| panic!("ERROR {}: {}", txt_path, e));

        let lines: Vec<&str> = data.lines().collect();
        let total = lines.len();

        if total == 0 {
            panic!("{}: file contains no lines", txt_path);
        }

        if n > total {
            panic!(
                "{}: requested {} lines but file only has {} lines",
                txt_path, n, total
            );
        }

        // choose start index so that start + n <= total
        let max_start = total - n;

        let start = if max_start == 0 {
            0usize
        } else {
            let mut rng = rand::rng();
            rng.random_range(0..=max_start)
        };

        let mut candis = Vec::with_capacity(n);

        for (offset, &line) in lines[start..start + n].iter().enumerate() {
            // keep original line-numbering style for panics (1-based)
            let line_no = start + offset;
            let mut parts = line.splitn(2, ',');

            let key = parts
                .next()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| panic!("line {}: missing key", line_no + 1));

            let value = parts
                .next()
                .map(str::trim)
                .unwrap_or_else(|| panic!("line {}: missing value part", line_no + 1));

            let val_list: Vec<String> = value
                .split('+')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();

            if val_list.is_empty() {
                panic!(
                    "line {}: no components found for key `{}`",
                    line_no + 1,
                    key
                );
            }

            candis.push((sanitize(key), val_list));
        }

        candis
    }

    #[test]
    // test is to validate the raw data path's are redable
    // if this fails, other tests won't work
    fn sanity_check() {
        for path in [CACHE_FILE] {
            match std::fs::metadata(path) {
                Err(_) => panic!("Unable to read {}", path),
                _ => {}
            }
        }
    }

    #[test]
    fn test_cached_splits() {
        create_logger();
        let cases = read_candidates(CACHE_FILE, 20);

        for (word, expected_parts) in cases {
            let expected_str = join_parts(expected_parts);
            let viccheda = Viccheda::new(true);
            let candi = viccheda.split(&word);

            assert!(
                candi.is_some(),
                "Expected at least one candidate for `{}`",
                word
            );
            assert!(
                candi.clone().is_some_and(|f| !f.splits.is_empty()),
                "Expected at least one candidate for `{}`",
                word
            );

            let joined: String = join_parts(candi.unwrap().splits);
            assert!(
                joined.contains(&expected_str.to_string()),
                "Word `{}` missing expected split `{}`.\nCandidates: {:?}",
                word,
                expected_str,
                joined
            );
        }
    }

    #[test]
    fn test_valid_splits() {
        create_logger();
        let cases = vec![
            ("शैब्यश्च", vec!["शैब्य", "च"]),
            ("युधामन्युश्च", vec!["युधामन्यु", "च"]),
            ("कर्णश्च", vec!["कर्ण", "च"]),
            ("कृपश्च", vec!["कृप", "च"]),
            ("विकर्णश्च", vec!["विकर्ण", "च"]),
        ];

        for (word, expected_parts) in cases {
            let expected_str = join_parts(
                expected_parts
                    .iter()
                    .map(|s| sanitize(s))
                    .collect::<Vec<String>>(),
            );

            let viccheda = Viccheda::new(true);
            let cands = viccheda.split(&word);

            assert!(
                cands.clone().is_some_and(|f| !f.splits.is_empty()),
                "Expected at least one candidate for `{}`",
                word
            );

            let joined: String = join_parts(cands.unwrap().splits);
            assert!(
                joined.contains(&expected_str.to_string()),
                "Word `{}` missing expected split `{}`.\nCandidates: {:?}",
                word,
                expected_str,
                joined
            );
        }
    }
}
