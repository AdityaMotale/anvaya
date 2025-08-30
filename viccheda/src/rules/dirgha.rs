use crate::{
    common::{AsChar, AsStr},
    rules::{ends_with, Rule, RuleData},
    split::Sandhi,
};

pub(crate) struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Vec<String>>> {
        let mut out = Vec::new();

        let merged_str = self.data.merged.as_str();
        let merged_char = self.data.merged.as_char();

        if !ends_with(left, &self.data.merged) {
            return None;
        }

        let base = {
            let mut b = left.trim_end_matches(merged_char);

            if let Some(str) = merged_str {
                b = b.trim_end_matches(str);
            }

            b.to_string()
        };

        let direct_right = {
            let out;

            if let Some(str) = self.data.right.as_str() {
                out = format!("{}{}", str, right);
            } else {
                out = format!("{}", right);
            }

            out
        };

        // first candidate
        out.push(vec![base.clone(), direct_right]);

        for splits in sandhi.split(right) {
            if splits.len() > 1 {
                let first_combined = {
                    let lft_data = &self.data.left;
                    let out;

                    if let Some(str) = lft_data.as_str() {
                        out = format!("{}{}", str, splits[0]);
                    } else {
                        out = format!("{}", splits[0]);
                    }

                    out
                };

                let mut cand = Vec::with_capacity(1 + splits.len());
                cand.push(base.clone());
                cand.push(first_combined);
                cand.extend(splits.into_iter().skip(1));

                out.push(cand);
            }
        }

        Some(out)
    }
}
