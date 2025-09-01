use crate::{
    rules::{Rule, RuleData, RuleUtils},
    split::{Candidate, Splitter},
};
use orthography::{AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }

    fn apply(&self, splitter: &Splitter, left: &str, right: &str) -> Option<Vec<Candidate>> {
        let mut out = Vec::new();

        // let merged_str = self.data.merged.as_str();
        // let merged_char = self.data.merged.as_char();

        // if !RuleUtils::ends_with_soundclass(left, &self.data.merged, &splitter.logger) {
        //     return None;
        // }

        // let base = match RuleUtils::trim_end_soundclass(&left, &self.data.merged, &splitter.logger)
        // {
        //     Some(b) => b,
        //     None => return None,
        // };

        // if base.is_empty() {
        //     return None;
        // }

        // let direct_right = if let Some(str) = self.data.right.as_str() {
        //     RuleUtils::nfc(format!("{}{}", str, right))
        // } else {
        //     RuleUtils::nfc(format!("{}", right))
        // };

        // let mut pushed = std::collections::HashSet::new();

        // // candidate with original left (keep merged)
        // let key_left = format!("{}|{}", left, direct_right);

        // if pushed.insert(key_left.clone()) {
        //     out.push(Candidate::new(
        //         vec![left.to_string(), direct_right.clone()],
        //         Some(self.data),
        //     ));
        // }

        // // candidate with trimmed base (drop merged) — only if base != left
        // if base != left {
        //     let left_with_sound = if let Some(str) = self.data.left.as_str() {
        //         RuleUtils::nfc(format!("{}{}", base, str))
        //     } else {
        //         base.clone()
        //     };

        //     let key_base = format!("{}|{}", left_with_sound, direct_right);

        //     if pushed.insert(key_base) {
        //         out.push(Candidate::new(
        //             vec![left_with_sound, direct_right.clone()],
        //             Some(self.data),
        //         ));
        //     }
        // }

        // if let Some(candidates) = splitter.candidates(right) {
        //     for candi in candidates {
        //         if candi.splits.len() > 1 {
        //             let first_combined = {
        //                 let lft_data = &self.data.left;
        //                 let out;

        //                 if let Some(str) = lft_data.as_str() {
        //                     out = format!("{}{}", str, candi.splits[0]);
        //                 } else {
        //                     out = format!("{}", candi.splits[0]);
        //                 }

        //                 out
        //             };

        //             let mut cand: Candidate =
        //                 Candidate::new(Vec::with_capacity(1 + candi.splits.len()), candi.rule);

        //             cand.splits.push(base.clone());
        //             cand.splits.push(first_combined);
        //             cand.splits.extend(candi.splits.into_iter().skip(1));

        //             out.push(cand);
        //         }
        //     }
        // }

        Some(out)
    }
}

impl SvarDirgha {
    pub fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::aa_to_a_a_rules());
        // rls.extend(Self::ii_to_i_i_rules());
        // rls.extend(Self::uu_to_u_u_rules());
        // rls.extend(Self::rr_to_r_r_rules());

        rls
    }

    fn aa_to_a_a_rules() -> Vec<Box<dyn Rule>> {
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
                    left: vec![SoundClass::Vowel(Vowel::A)],
                    right: vec![SoundClass::IndependentVowel(IndependentVowel::A)],
                    merged: vec![SoundClass::Vowel(Vowel::AA)],
                },
            }),
            // Box::new(SvarDirgha {
            //     data: RuleData {
            //         name: "savarṇa-dīrgha-a2",
            //         desc: "आ  => आ  + अ ",
            //         tag: "6.1.101",
            //         left: SoundClass::Vowel(Vowel::AA),
            //         right: SoundClass::IndepVowel(IndepVowel::A),
            //         merged: SoundClass::Vowel(Vowel::AA),
            //     },
            // }),
            // Box::new(SvarDirgha {
            //     data: RuleData {
            //         name: "savarṇa-dīrgha-a3",
            //         desc: "आ  => अ + आ ",
            //         tag: "6.1.101",
            //         left: SoundClass::Vowel(Vowel::A),
            //         right: SoundClass::IndepVowel(IndepVowel::AA),
            //         merged: SoundClass::Vowel(Vowel::AA),
            //     },
            // }),
            // Box::new(SvarDirgha {
            //     data: RuleData {
            //         name: "savarṇa-dīrgha-a4",
            //         desc: "आ  => आ  + आ ",
            //         tag: "6.1.101",
            //         left: SoundClass::Vowel(Vowel::AA),
            //         right: SoundClass::IndepVowel(IndepVowel::AA),
            //         merged: SoundClass::Vowel(Vowel::AA),
            //     },
            // }),
        ]
    }

    // fn ii_to_i_i_rules() -> Vec<Box<dyn Rule>> {
    //     vec![
    //         // NOTE: इ should not be added at the end of left candidate, that's why
    //         // we did't choose [IndependentVowl] for the `left` window in this rule
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-i1",
    //                 desc: "ई => इ + इ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::I),
    //                 right: SoundClass::IndepVowel(IndepVowel::I),
    //                 merged: SoundClass::Vowel(Vowel::II),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-i2",
    //                 desc: "ई => ई + इ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::II),
    //                 right: SoundClass::IndepVowel(IndepVowel::I),
    //                 merged: SoundClass::Vowel(Vowel::II),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-i3",
    //                 desc: "ई => इ + ई",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::I),
    //                 right: SoundClass::IndepVowel(IndepVowel::II),
    //                 merged: SoundClass::Vowel(Vowel::II),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-i4",
    //                 desc: "ई => ई + ई",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::II),
    //                 right: SoundClass::IndepVowel(IndepVowel::II),
    //                 merged: SoundClass::Vowel(Vowel::II),
    //             },
    //         }),
    //     ]
    // }

    // fn uu_to_u_u_rules() -> Vec<Box<dyn Rule>> {
    //     vec![
    //         // NOTE: उ should not be added at the end of left candidate, that's why
    //         // we did't choose [IndependentVowl] for the `left` window in this rule
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-u1",
    //                 desc: "ऊ => उ + उ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::U),
    //                 right: SoundClass::IndepVowel(IndepVowel::U),
    //                 merged: SoundClass::Vowel(Vowel::UU),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-u2",
    //                 desc: "ऊ  => ऊ  + उ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::UU),
    //                 right: SoundClass::IndepVowel(IndepVowel::U),
    //                 merged: SoundClass::Vowel(Vowel::UU),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-u3",
    //                 desc: "ऊ => उ + ऊ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::U),
    //                 right: SoundClass::IndepVowel(IndepVowel::UU),
    //                 merged: SoundClass::Vowel(Vowel::UU),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-u4",
    //                 desc: "ऊ => ऊ  + ऊ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::UU),
    //                 right: SoundClass::IndepVowel(IndepVowel::UU),
    //                 merged: SoundClass::Vowel(Vowel::UU),
    //             },
    //         }),
    //     ]
    // }

    // fn rr_to_r_r_rules() -> Vec<Box<dyn Rule>> {
    //     vec![
    //         // NOTE: ॠ  (IndepVowel::RR) should not be added at the end of left candidate, that's why
    //         // we did't choose [IndependentVowl] for the `left` window in this rule
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-r1",
    //                 desc: "ॠ => ॠ + ॠ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::R),
    //                 right: SoundClass::IndepVowel(IndepVowel::R),
    //                 merged: SoundClass::Vowel(Vowel::RR),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-r2",
    //                 desc: "ॠ => ॠ + ॠ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::RR),
    //                 right: SoundClass::IndepVowel(IndepVowel::R),
    //                 merged: SoundClass::Vowel(Vowel::RR),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-r3",
    //                 desc: "ॠ => ॠ + ॠ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::R),
    //                 right: SoundClass::IndepVowel(IndepVowel::RR),
    //                 merged: SoundClass::Vowel(Vowel::RR),
    //             },
    //         }),
    //         Box::new(SvarDirgha {
    //             data: RuleData {
    //                 name: "savarṇa-dīrgha-r4",
    //                 desc: "ॠ => ॠ + ॠ ",
    //                 tag: "6.1.101",
    //                 left: SoundClass::Vowel(Vowel::RR),
    //                 right: SoundClass::IndepVowel(IndepVowel::RR),
    //                 merged: SoundClass::Vowel(Vowel::RR),
    //             },
    //         }),
    //     ]
    // }
}
