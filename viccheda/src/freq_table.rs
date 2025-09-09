include!(concat!(env!("OUT_DIR"), "/freq_table.rs"));

pub(crate) struct FreqTable;

impl FreqTable {
    #[inline]
    pub fn get(key: &str) -> Option<usize> {
        FREQ_TABLE.get(key).copied()
    }

    #[inline]
    pub fn iter() -> impl Iterator<Item = (&'static str, usize)> {
        FREQ_TABLE.entries().map(|(k, v)| (*k, *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_values_are_some() {
        for (key, value) in FreqTable::iter() {
            assert!(
                value > 0,
                "Key {key:?} has non-positive or missing value: {value}"
            );
        }
    }

    #[test]
    fn test_roundtrip_get_vs_iter() {
        for (key, value) in FreqTable::iter() {
            assert_eq!(FreqTable::get(key), Some(value));
        }
    }
}
