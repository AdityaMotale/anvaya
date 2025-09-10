#![allow(unused)]

use crate::{cache_table::CacheTable, rules::rule::Candidate, split::Splitter};

mod cache_table;
mod freq_table;
mod rules;
mod split;

pub struct Viccheda {
    splitter: Splitter,
}

impl Viccheda {
    pub fn new(debug: bool) -> Self {
        Self {
            splitter: Splitter::new(debug),
        }
    }

    pub fn split(&self, word: &str) -> Option<(Candidate, f64)> {
        // sandhi cache
        if let Some(res) = CacheTable::get(word) {
            return Some((res, 1.0));
        }

        self.splitter.best_candidate(word)
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

    const CACHE_FILE: &'static str = "../raw_data/cache.txt";

    fn join_parts(parts: Vec<String>) -> String {
        parts.join("|")
    }

    fn create_logger() {
        let _ = crate::init_logger("Viccheda (Test)");
    }

    fn read_candidates(txt_path: &str) -> Vec<(String, Vec<String>)> {
        let mut candis = Vec::new();

        let data = std::fs::read_to_string(txt_path)
            .unwrap_or_else(|e| panic!("ERROR {}: {}", txt_path, e));

        for (line_no, line) in data.lines().enumerate() {
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
                .map(|s| sanitize(s))
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
    #[ignore]
    fn test_cached_splits() {
        create_logger();
        let cases = read_candidates(CACHE_FILE);

        for (word, expected_parts) in cases {
            let expected_str = join_parts(expected_parts);
            let viccheda = Viccheda::new(true);
            let cands = viccheda.split(&word);

            assert!(
                cands.clone().is_some_and(|(f, _)| !f.splits.is_empty()),
                "Expected at least one candidate for `{}`",
                word
            );

            let joined: String = join_parts(cands.unwrap().0.splits);
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
    #[ignore]
    fn test_valid_splits() {
        create_logger();
        let cases = vec![
            ("शैब्यश्च", vec!["शैब्य", "च"]),
            ("युधामन्युश्च", vec!["युधामन्यु", "च"]),
            ("तान्निबोध", vec!["तान", "निबोध"]),
            ("द्वीजोत्तम", vec!["द्विज", "उत्तम"]),
            ("कर्णश्च", vec!["कर्ण", "च"]),
            ("कृपश्च", vec!["कृप", "च"]),
            ("समितिञ्जय", vec!["समितिम", "जय"]),
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
                cands.clone().is_some_and(|(f, _)| !f.splits.is_empty()),
                "Expected at least one candidate for `{}`",
                word
            );

            let joined: String = join_parts(cands.unwrap().0.splits);
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
