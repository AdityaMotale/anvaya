include!(concat!(env!("OUT_DIR"), "/candi_map.rs"));

pub struct SplitTable;

impl SplitTable {
    #[inline]
    pub fn get(key: &str) -> Option<&'static [&'static str]> {
        CANDI_MAP.get(key).copied()
    }

    #[inline]
    pub fn iter() -> impl Iterator<Item = (&'static str, &'static [&'static str])> {
        CANDI_MAP.entries().map(|(k, v)| (*k, *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_no_empty_values_and_len_bounds() {
        for (k, vals) in SplitTable::iter() {
            assert!(!vals.is_empty(), "key {:?} has empty value slice", k);

            for comp in vals {
                assert!(!comp.is_empty(), "key {:?} has empty component", k);
            }
        }
    }

    #[test]
    fn test_iter_get_roundtrip() {
        for (k, vals) in SplitTable::iter() {
            assert_eq!(SplitTable::get(k).unwrap(), vals);
        }
    }
}
