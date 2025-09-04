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
        rls.extend(Self::o_indep_to_a_u());
        rls.extend(Self::o_to_a_u());
        rls.extend(Self::ar_to_a_ar());
        rls.extend(Self::al_to_a_lr());

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
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e2",
                desc: "ए = अ + ई",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e3",
                desc: "ए = आ  + इ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::I)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-e4",
                desc: "ए = आ  + इ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
                special_sequence: None,
            })),
        ]
    }

    fn o_to_a_u() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-o1",
                desc: "ओ  = अ + उ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-o2",
                desc: "ओ  = अ + ऊ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-o3",
                desc: "ओ  = आ  + उ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-o4",
                desc: "ओ  = आ  + ई ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-o5",
                desc: "ओ  = आ  + ऊ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            })),
        ]
    }

    fn o_indep_to_a_u() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-O1",
                desc: "ओ (INDEP) = अ + उ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-O2",
                desc: "ओ (INDEP) = अ + ऊ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-O3",
                desc: "ओ (INDEP) = आ  + उ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-O4",
                desc: "ओ (INDEP) = आ  + उ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-O5",
                desc: "ओ (INDEP) = आ  + ऊ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::O)]),
                special_sequence: None,
            })),
        ]
    }

    // FIXME: This currently will not work, cuase the appearence of
    // (अर्) is at the end of the morpheme, which gets skipped cuase
    // it's a single grapheme.
    //
    // NOTE: In order to fix this, we'll have to split the last token
    // on chars, and not grapheme
    fn ar_to_a_ar() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-ar1",
                desc: "अर् = अ + ऋ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-ar2",
                desc: "अर् = आ  + ऋ ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]),
                special_sequence: None,
            })),
        ]
    }

    fn al_to_a_lr() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-al1",
                desc: "अल् = अ + लृ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![
                    SoundClass::Consonant(Consonant::La),
                    SoundClass::Vowel(Vowel::R),
                ]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::La),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-al2",
                desc: "अल् = आ  + लृ",
                tag: "6.1.87",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![
                    SoundClass::Consonant(Consonant::La),
                    SoundClass::Vowel(Vowel::R),
                ]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::La),
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
        let _ = crate::init_logger("SvarGuna Rules (Test)");
    }

    #[test]
    fn e_to_a_i_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामेतिहास:", vec![vec!["राम", "इतिहास:"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn e_to_a_i_test() {
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

    #[test]
    fn o_to_a_u_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("जलोंर्मिः", vec![vec!["जल", "ऊर्मिः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn o_to_a_u_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सूर्योदयः", vec![vec!["सूर्य", "उदयः"]]),
            ("पूर्वोदयः", vec![vec!["पूर्व", "उदयः"]]),
            ("महोदयः", vec![vec!["महा", "उदयः"]]),
            ("परोपकारः", vec![vec!["पर", "उपकारः"]]),
            ("लोकोक्तिः", vec![vec!["लोक", "उक्तिः"]]),
            ("वृक्षोपरि", vec![vec!["वृक्ष", "उपरि"]]),
            ("हितोपदेशः", vec![vec!["हित", "उपदेशः"]]),
            ("पुरुषोत्तमः", vec![vec!["पुरुष", "उत्तमः"]]),
            ("परमोत्तमः", vec![vec!["परम", "उत्तमः"]]),
            ("महोत्सवः", vec![vec!["महा", "उत्सवः"]]),
            ("परीक्षोत्सवः", vec![vec!["परीक्षा", "उत्सवः"]]),
            ("गङ्गोदकम्", vec![vec!["गङ्गा", "उदकम्"]]),
            ("अत्यन्तोर्ध्वम्", vec![vec!["अत्यन्त", "ऊर्ध्वम्"]]),
            ("एकोनः", vec![vec!["एक", "ऊनः"]]),
            ("गगनोर्ध्वम्", vec![vec!["गगन", "ऊर्ध्वम्"]]),
            ("माययोर्जस्वि", vec![vec!["मायया", "ऊर्जस्वि"]]),
            ("महोर्णम्", vec![vec!["महा", "ऊर्णम्"]]),
            ("समुद्रोर्मिः", vec![vec!["समुद्र", "ऊर्मिः"]]),
            ("गंगोर्मिः", vec![vec!["गंगा", "ऊर्मिः"]]),
            ("महोर्मिः", vec![vec!["महा", "ऊर्मिः"]]),
            ("वीरोचितः", vec![vec!["वीर", "उचितः"]]),
            ("आद्योपान्तः", vec![vec!["आद्य", "उपान्तः"]]),
            ("नवोढ़ा", vec![vec!["नव", "ऊढ़ा"]]),
            ("महोदधिः", vec![vec!["महा", "उदधिः"]]),
            ("यथोचितम्", vec![vec!["यथा", "उचितम्"]]),
            ("कथोपकथनम्", vec![vec!["कथा", "उपकथनम्"]]),
            ("विद्योपार्जनम्", vec![vec!["विद्या", "उपार्जनम्"]]),
            ("कण्ठोच्चारणम्", vec![vec!["कण्ठ", "उच्चारणम्"]]),
            ("तवोतिः", vec![vec!["तव", "ऊतिः"]]),
            ("नोपलब्धि:", vec![vec!["न", "उपलब्धि:"]]),
            ("पादोनः", vec![vec!["पाद", "ऊनः"]]),
            ("आत्मोत्सर्गः", vec![vec!["आत्म", "उत्सर्गः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    #[ignore]
    fn ar_to_a_ar_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("कृष्णर्द्धिः", vec![vec!["कृष्ण", "ऋद्धिः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn ar_to_a_ar_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("कृष्णर्द्धिः", vec![vec!["कृष्ण", "ऋद्धिः"]]),
            ("महर्द्धिः", vec![vec!["महा", "ऋद्धिः"]]),
            ("ममर्द्धिः", vec![vec!["मम", "ऋद्धिः"]]),
            ("पापर्द्धिः", vec![vec!["पाप", "ऋद्धिः"]]),
            ("ग्रीष्मर्तुः", vec![vec!["ग्रीष्म", "ऋतु:"]]),
            ("वर्षतु:", vec![vec!["वर्षा", "ऋतुः"]]),
            ("वसन्तर्तुः", vec![vec!["वसन्त", "ऋतु:"]]),
            ("सदर्तुः", vec![vec!["सदा", "ऋतुः"]]),
            ("शिशिरर्तुः", vec![vec!["शिशिर", "ऋतु:"]]),
            ("राजर्षिः", vec![vec!["राज", "ऋषिः"]]),
            ("महर्षिः", vec![vec!["महा", "ऋषिः"]]),
            ("देवर्षिः", vec![vec!["देव", "ऋषिः"]]),
            ("सप्तर्षिः", vec![vec!["सप्त", "ऋषिः"]]),
            ("ब्रह्मर्षिः", vec![vec!["ब्रह्म", "ऋषिः"]]),
            ("महर्णः", vec![vec!["महा", "ऋणः"]]),
            ("सदर्णः", vec![vec!["सदा", "ऋणः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn al_to_a_lr_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("तवल्कारः", vec![vec!["तव", "लृकारः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn al_to_a_lr_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("तवल्कारः", vec![vec!["तव", "लृकारः"]]),
            ("ममल्कारः", vec![vec!["मम", "लृकारः"]]),
            ("यथल्कार:", vec![vec!["यथा", "लृकारः"]]),
            ("कदल्कारः", vec![vec!["कदा", "लृकारः"]]),
            ("मालाल्कारः", vec![vec!["माला", "लृकारः"]]),
            ("तवल्दन्तः", vec![vec!["तव", "लृदन्तः"]]),
            ("सदल्वर्णः", vec![vec!["सदा", "लृवर्णः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
