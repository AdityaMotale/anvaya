use orthography::Akshara;

use crate::rules::rule::{Candidate, RuleData};

include!(concat!(env!("OUT_DIR"), "/cache_table.rs"));

pub struct CacheTable;

impl CacheTable {
    pub fn get(key: &str) -> Option<Vec<Candidate>> {
        if let Some(res) = Self::_get(key) {
            let mut out = Vec::new();

            out.push(Candidate {
                splits: res.iter().map(|s| s.to_string()).collect::<Vec<String>>(),
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

            return Some(out);
        }

        None
    }

    #[inline]
    fn _get(key: &str) -> Option<&'static [&'static str]> {
        CACHE_TABLE.get(key).copied()
    }

    #[inline]
    pub fn iter() -> impl Iterator<Item = (&'static str, &'static [&'static str])> {
        CACHE_TABLE.entries().map(|(k, v)| (*k, *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_no_empty_values_and_len_bounds() {
        for (k, vals) in CacheTable::iter() {
            assert!(!vals.is_empty(), "key {:?} has empty value slice", k);

            for comp in vals {
                assert!(!comp.is_empty(), "key {:?} has empty component", k);
            }
        }
    }

    #[test]
    fn test_iter_get_roundtrip() {
        for (k, vals) in CacheTable::iter() {
            assert_eq!(CacheTable::_get(k).unwrap(), vals);
        }
    }
}
