#![allow(unused)]

use crate::{candi_table::CacheTable, rules::rule::Candidate};

mod candi_table;
mod freq_table;
mod rules;
mod split;

pub struct Viccheda;

impl Viccheda {
    pub fn split(word: &str) -> Vec<Candidate> {
        let mut candis = Vec::new();

        // sandhi cache
        if let Some(res) = CacheTable::get(word) {
            return res;
        }

        candis
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

    fn join_parts(parts: Vec<String>) -> String {
        parts.join("|")
    }

    #[test]
    fn test_cached_splits() {
        let cases = vec![
            ("भीष्ममेवाभिरक्षन्तु", vec!["भीष्मम", "एव", "अभिरक्षन्तु"]),
            ("सहसैवाभ्यहन्यन्त", vec!["सहसा", "एव", "अभ्यहन्यन्त"]),
            ("श्वेतैर्हयैर्युक्ते", vec!["श्वेतै", "हयै", "युक्ते"]),
            ("पाण्डवश्चैव", vec!["पाण्डव", "च", "एव"]),
            ("सात्यकिश्चापराजित", vec!["सात्यकि", "च", "अपराजित"]),
            ("वाक्यमिदमाह", vec!["वाक्यम", "इदम", "आह"]),
            ("सेनयोरुभयोर्मध्ये", vec!["सेनयो", "उभयो", "मध्ये"]),
            ("योद्धव्यमस्मिन्रणसमुद्यमे", vec!["योद्धव्यम", "अस्मिन", "रणसमुद्यमे"]),
            (
                "पश्यैतान्समवेतान्कुरूनिति",
                vec!["पश्य", "एतान", "समवेतान", "कुरून", "इति"],
            ),
            ("तत्रापश्यत्स्थितान्पार्थ", vec!["तत्र", "अपश्यत", "स्थितान", "पार्थ"]),
        ];

        for (word, expected_parts) in cases {
            let expected_str = join_parts(
                expected_parts
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            );

            let cands = Viccheda::split(word);

            assert!(
                !cands.is_empty(),
                "Expected at least one candidate for `{}`",
                word
            );

            let joined: Vec<String> = cands.iter().map(|c| join_parts(c.splits.clone())).collect();

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
