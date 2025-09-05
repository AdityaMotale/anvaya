use crate::{
    rules::{
        rule::{BaseRule, RuleData, RuleGroup},
        Rule,
    },
    split::Splitter,
};
use orthography::{
    Adjuncts, Akshara, AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel,
};

pub(crate) struct SvarVriddhi;

impl RuleGroup for SvarVriddhi {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ai_to_a_e());
        rls.extend(Self::au_to_a_o());
        rls.extend(Self::ar_to_a_r());

        rls
    }
}

impl SvarVriddhi {
    fn ai_to_a_e() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai1",
                desc: "ऐ = अ + ए",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::E)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai2",
                desc: "ऐ = अ + ऐ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AI)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai3",
                desc: "ऐ = आ  + ए",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::E)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai4",
                desc: "ऐ = आ  + ऐ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AI)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
        ]
    }

    fn au_to_a_o() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-au1",
                desc: "औ  = अ + औ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AU)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-au2",
                desc: "औ  = अ + ओ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AU)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-au3",
                desc: "औ  = आ  + ओ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AU)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-au4",
                desc: "औ  = आ  + औ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AU)]),
                special_sequence: None,
            })),
        ]
    }

    fn ar_to_a_r() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ar1",
                desc: "आर् = अ + ऋ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::AA),
                    SoundClass::Consonant(Consonant::Ra),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ar2",
                desc: "आर् = आ  + ऋ ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::AA),
                    SoundClass::Consonant(Consonant::Ra),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarVriddhi Rules (Test)");
    }

    #[test]
    fn ai_to_a_e_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("जनैकता", vec![vec!["जन", "एकता"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ai_to_a_e_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("जनैकता", vec![vec!["जन", "एकता"]]),
            ("एकैकः", vec![vec!["एक", "एकः"]]),
            ("अत्रैकमत्यम्", vec![vec!["अत्र", "एकमत्यम्"]]),
            ("राजैषः", vec![vec!["राज", "एषः"]]),
            ("बालैषा", vec![vec!["बाला", "एषा"]]),
            ("तथैव", vec![vec!["तथा", "एव"]]),
            ("वसुधैव", vec![vec!["वसुधा", "एव"]]),
            ("गंगैषा", vec![vec!["गंगा", "एषा"]]),
            ("पुत्रैषणा", vec![vec!["पुत्र", "ऐषणा"]]),
            ("सदैव", vec![vec!["सदा", "एव"]]),
            ("देवैश्वर्यम्", vec![vec!["देव", "ऐश्वर्यम्"]]),
            ("नृपैश्वर्यम्", vec![vec!["नृप", "ऐश्वर्यम्"]]),
            ("महैश्वर्यम्", vec![vec!["महा", "ऐश्वर्यम्"]]),
            ("ममैश्वर्यम्", vec![vec!["मम", "ऐश्वर्यम्"]]),
            ("गङ्गैश्वर्यम्", vec![vec!["गङ्गा", "ऐश्वर्यम्"]]),
            ("मतैक्यम्", vec![vec!["मत", "ऐक्यम्"]]),
            ("सर्वैक्यम्", vec![vec!["सर्व", "ऐक्यम्"]]),
            ("कृष्णैकत्वम्", vec![vec!["कृष्ण", "एकत्वम्"]]),
            ("देवैकत्वम्", vec![vec!["देव", "एकत्वम्"]]),
            ("हितैषी", vec![vec!["हित", "एषी"]]),
            ("वित्तैषणा", vec![vec!["वित्त", "एषणा"]]),
            ("लोकैषणा", vec![vec!["लोक", "एषणा"]]),
            ("विद्यैषणा", vec![vec!["विद्या", "एषणा"]]),
            ("सभैका", vec![vec!["सभा", "एका"]]),
            ("वीरैकः", vec![vec!["वीर", "एकः"]]),
            ("राजैष:", vec![vec!["राजा", "एष:"]]),
            ("मैवम्", vec![vec!["मा", "एवम्"]]),
            ("तवैवम्", vec![vec!["तव", "एवम्"]]),
            ("अत्रैकम्", vec![vec!["अत्र", "एकम्"]]),
            ("जनैकता", vec![vec!["जन", "एकता"]]),
            ("स्वैच्छिकम्", vec![vec!["स्व", "ऐच्छिकम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn au_to_a_o_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("कृष्णौत्कण्ठ्यम्", vec![vec!["कृष्ण", "औत्कण्ठ्यम्"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn au_to_a_o_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("कृष्णौत्कण्ठ्यम्", vec![vec!["कृष्ण", "औत्कण्ठ्यम्"]]),
            ("ममौत्कण्ठ्यम्", vec![vec!["मम", "औत्कण्ठ्यम्"]]),
            ("तवौदार्यम्", vec![vec!["तव", "औदार्यम्"]]),
            ("जनौचित्यम्", vec![vec!["जन", "औचित्यम्"]]),
            ("विद्यौचित्यम्", vec![vec!["विद्या", "औचित्यम्"]]),
            ("महौदार्यम्", vec![vec!["महा", "औदार्यम्"]]),
            ("परमौदार्यम्", vec![vec!["परम", "औदार्यम्"]]),
            ("देवौदार्यम्", vec![vec!["देव", "औदार्यम्"]]),
            ("रामौत्सुक्यम्", vec![vec!["राम", "औत्सुक्यम्"]]),
            ("क्रीडौत्सुक्यम्", vec![vec!["क्रीडा", "औत्सुक्यम्"]]),
            ("दर्शनौत्सुक्यम्", vec![vec!["दर्शन", "औत्सुक्यम्"]]),
            ("सदौत्सुक्यम्", vec![vec!["सदा", "औत्सुक्यम्"]]),
            ("महौषधिः", vec![vec!["महा", "औषधिः"]]),
            ("वनौषधिः", vec![vec!["वन", "औषधिः"]]),
            ("तीक्ष्णौषधिः", vec![vec!["तीक्ष्ण", "औषधिः"]]),
            ("परमौषधिः", vec![vec!["परम", "औषधिः"]]),
            ("प्रौद्योगिकी", vec![vec!["प्र", "औद्योगिकी"]]),
            ("बिम्बौष्ठी", vec![vec!["बिम्ब", "औष्ठी"]]),
            ("ममौदासीन्यम्", vec![vec!["मम", "औदासीन्यम्"]]),
            ("तण्डुलौदनम्", vec![vec!["तण्डुल", "ओदनम्"]]),
            ("परमौजः", vec![vec!["परम", "ओजः"]]),
            ("महौघः", vec![vec!["महा", "औघः"]]),
            ("जलौघः", vec![vec!["जल", "ओघः"]]),
            ("गंगौघः", vec![vec!["गंगा", "ओघः"]]),
            ("मधुरौदनः", vec![vec!["मधुर", "ओदनः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn ar_to_a_r_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("दुखार्तः", vec![vec!["दुख", "ऋतः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ar_to_a_r_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("दुखार्तः", vec![vec!["दुख", "ऋतः"]]),
            ("सुखार्तः", vec![vec!["सुख", "ऋतः"]]),
            ("बुभुक्षार्तः", vec![vec!["बुभुक्षा", "ऋतः"]]),
            ("पिपासार्तः", vec![vec!["पिपासा", "ऋतः"]]),
            ("दीनार्तः", vec![vec!["दीन", "ऋतः"]]),
            ("प्रार्च्छति", vec![vec!["प्र", "ऋच्छति"]]),
            ("कम्बलार्णम्", vec![vec!["कम्बल", "ऋणम्"]]),
            ("वसनार्णम्", vec![vec!["वसन", "ऋणम्"]]),
            ("दशार्णः", vec![vec!["दश", "ऋणः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
