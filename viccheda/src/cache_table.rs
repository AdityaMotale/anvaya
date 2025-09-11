include!(concat!(env!("OUT_DIR"), "/cache_map.rs"));

use crate::rules::rule::{Candidate, RuleData};
use orthography::{sanitize, Akshara};

pub struct CacheTable;

impl CacheTable {
    pub fn get(key: &str) -> Option<Candidate> {
        if let Some(res) = Self::_get(key) {
            return Some(Candidate {
                // NOTE: We sanitize the output here
                splits: res.iter().map(|s| sanitize(s)).collect::<Vec<String>>(),
                rule: RuleData {
                    name: "Cache",
                    desc: "NA",
                    tag: "NA",
                    left: Akshara(vec![]),
                    right: Akshara(vec![]),
                    merged: Akshara(vec![]),
                    special_sequence: None,
                },
            });
        }

        None
    }

    #[inline]
    fn _get(key: &str) -> Option<&'static [&'static str]> {
        CACHE_TABLE.get(key).copied()
    }

    #[inline]
    fn _iter() -> impl Iterator<Item = (&'static str, &'static [&'static str])> {
        CACHE_TABLE.entries().map(|(k, v)| (*k, *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_no_empty_values_and_len_bounds() {
        for (k, vals) in CacheTable::_iter() {
            assert!(!vals.is_empty(), "key {:?} has empty value slice", k);

            for comp in vals {
                assert!(!comp.is_empty(), "key {:?} has empty component", k);
            }
        }
    }

    #[test]
    fn test_iter_get_roundtrip() {
        for (k, vals) in CacheTable::_iter() {
            assert_eq!(CacheTable::_get(k).unwrap(), vals);
        }
    }
}
