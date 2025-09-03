use crate::{
    rules::{BaseRule, Rule, RuleData, RuleGroup, RuleUtils},
    split::{Candidate, Splitter},
};
use orthography::{
    Adjuncts, Akshara, AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel,
};

pub(crate) struct SvarGuna;

impl RuleGroup for SvarGuna {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::e_to_a_i());

        rls
    }
}

impl SvarGuna {
    fn e_to_a_i() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e1",
                desc: "ए = अ + इ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::I)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e2",
                desc: "ए = अ + ई",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e3",
                desc: "ए = आ  + इ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::I)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e4",
                desc: "ए = आ  + इ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarGuna Rules (Test)");
    }

    #[test]
    fn e_to_a_i_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामेतिहास:", vec![vec!["राम", "इतिहास:"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn e_to_a_i() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("देवेन्द्रः", vec![vec!["देव", "इन्द्रः"]]),
            ("उपेन्द्रः", vec![vec!["उप", "इन्द्रः"]]),
            ("नरेन्द्रः", vec![vec!["नर", "इन्द्रः"]]),
            ("सुरेन्द्रः", vec![vec!["सुर", "इन्द्रः"]]),
            ("गजेन्द्रः", vec![vec!["गज", "इन्द्रः"]]),
            ("महेन्द्रः", vec![vec!["महा", "इन्द्रः"]]),
            ("रमेन्द्रः", vec![vec!["रमा", "इन्द्रः"]]),
            ("राजेन्द्रः", vec![vec!["राजा", "इन्द्रः"]]),
            ("नेति", vec![vec!["न", "इति"]]),
            ("तथेति", vec![vec!["तथा", "इति"]]),
            ("स्वेच्छा", vec![vec!["स्व", "इच्छा"]]),
            ("विकलेन्द्रियः", vec![vec!["विकल", "इन्द्रियः"]]),
            ("यथेच्छम्", vec![vec!["यथा", "इच्छम्"]]),
            ("यथेष्टम्", vec![vec!["यथा", "इष्टम्"]]),
            ("गणेशः", vec![vec!["गण", "ईशः"]]),
            ("सर्वेशः", vec![vec!["सर्व", "ईशः"]]),
            ("सुरेशः", vec![vec!["सुर", "ईशः"]]),
            ("दिनेशः", vec![vec!["दिन", "ईशः"]]),
            ("रमेशः", vec![vec!["रमा", "ईशः"]]),
            ("गङ्गेश्वरः", vec![vec!["गङ्गा", "ईश्वरः"]]),
            ("परमेश्वरः", vec![vec!["परम", "ईश्वरः"]]),
            ("महेश्वरः", vec![vec!["महा", "ईश्वरः"]]),
            ("उमेशः", vec![vec!["उमा", "ईशः"]]),
            ("महेशः", vec![vec!["महा", "ईशः"]]),
            ("गणेशः", vec![vec!["गण", "ईशः"]]),
            ("लंकेशः", vec![vec!["लंका", "ईशः"]]),
            ("नरेशः", vec![vec!["नर", "ईशः"]]),
            ("सोमेशः", vec![vec!["सोम", "ईशः"]]),
            ("अंत्येष्टि", vec![vec!["अंत्य", "इष्टि"]]),
            ("उपेक्षा", vec![vec!["उप", "ईक्षा"]]),
            ("प्रेक्षकः", vec![vec!["प्र", "ईक्षकः"]]),
            ("प्रेक्षते", vec![vec!["प्र", "ईक्षते"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
