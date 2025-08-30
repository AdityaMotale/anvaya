use crate::{
    common::{AsChar, AsStr, Consonant, IndepVowel, SoundClass, Vowel},
    rules::{ends_with_soundclass, nfc, trim_end_soundclass, Rule, RuleData},
    split::{Candidate, Sandhi},
};

pub(crate) struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(&self, sandhi: &Sandhi, left: &str, right: &str) -> Option<Vec<Candidate>> {
        let mut out = Vec::new();

        let merged_str = self.data.merged.as_str();
        let merged_char = self.data.merged.as_char();

        if !ends_with_soundclass(left, &self.data.merged) {
            return None;
        }

        let base = trim_end_soundclass(&left, &self.data.merged);

        let direct_right = if let Some(str) = self.data.right.as_str() {
            nfc(format!("{}{}", str, right))
        } else {
            nfc(format!("{}", right))
        };

        // this is the current candidate with split based on the [merged]
        // value of the rule
        out.push(Candidate::new(
            vec![base.clone(), direct_right],
            Some(self.data),
        ));

        if let Some(candidates) = sandhi.split(right) {
            for candi in candidates {
                if candi.splits.len() > 1 {
                    let first_combined = {
                        let lft_data = &self.data.left;
                        let out;

                        if let Some(str) = lft_data.as_str() {
                            out = format!("{}{}", str, candi.splits[0]);
                        } else {
                            out = format!("{}", candi.splits[0]);
                        }

                        out
                    };

                    let mut cand: Candidate =
                        Candidate::new(Vec::with_capacity(1 + candi.splits.len()), candi.rule);

                    cand.splits.push(base.clone());
                    cand.splits.push(first_combined);
                    cand.splits.extend(candi.splits.into_iter().skip(1));

                    out.push(cand);
                }
            }
        }

        Some(out)
    }
}

impl SvarDirgha {
    pub fn rules() -> Vec<Box<dyn Rule>> {
        // HACK: (आ ) directly at the word as a sandi is rare, so we just ignore it,
        // split on the [Vowel] ("ा") form and not the [IndepVowel] form

        vec![
            // NOTE: अ  should not be added at the end of left candidate, that's why
            // we did't choose [IndependentVowl] for the `left` window in this rule
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a1",
                    desc: "आ  => अ + अ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::A),
                    right: SoundClass::IndepVowel(IndepVowel::A),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a2",
                    desc: "आ  => आ  + अ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::AA),
                    right: SoundClass::IndepVowel(IndepVowel::A),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a3",
                    desc: "आ  => अ + आ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::A),
                    right: SoundClass::IndepVowel(IndepVowel::AA),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-a4",
                    desc: "आ  => आ  + आ ",
                    tag: "6.1.101",
                    left: SoundClass::Vowel(Vowel::AA),
                    right: SoundClass::IndepVowel(IndepVowel::AA),
                    merged: SoundClass::Vowel(Vowel::AA),
                },
            }),
        ]
    }
}
