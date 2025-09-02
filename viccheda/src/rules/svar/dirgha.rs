use crate::{
    rules::{Rule, RuleData, RuleUtils},
    split::{Candidate, Splitter},
};
use orthography::{Akshara, AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarDirgha {
    pub data: RuleData,
}

impl Rule for SvarDirgha {
    fn data(&self) -> &RuleData {
        &self.data
    }
}

impl SvarDirgha {
    pub fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::aa_to_a_a_rules());
        rls.extend(Self::rr_to_r_r_rules());

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
                    left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                    right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                    merged: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                },
            }),
        ]
    }

    fn rr_to_r_r_rules() -> Vec<Box<dyn Rule>> {
        vec![
            // NOTE: ॠ  (IndepVowel::RR) should not be added at the end of left candidate, that's why
            // we did't choose [IndependentVowl] for the `left` window in this rule
            Box::new(SvarDirgha {
                data: RuleData {
                    name: "savarṇa-dīrgha-r1",
                    desc: "ॠ => ॠ + ॠ ",
                    tag: "6.1.101",
                    left: Akshara(vec![SoundClass::Vowel(Vowel::R)]),
                    right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                    merged: Akshara(vec![
                        SoundClass::Vowel(Vowel::R),
                        SoundClass::Vowel(Vowel::R),
                    ]),
                },
            }),
        ]
    }
}
