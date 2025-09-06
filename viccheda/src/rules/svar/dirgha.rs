use crate::rules::{
    Rule,
    rule::{BaseRule, RuleData, RuleGroup},
};
use orthography::{Adjuncts, Akshara, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarDirgha;

impl RuleGroup for SvarDirgha {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::aa_to_a_a());
        rls.extend(Self::ii_to_i_i());
        rls.extend(Self::uu_to_u_u());
        rls.extend(Self::rr_to_r_r());

        rls
    }
}

// HACK: Independent Vowels directly at the left candidate or window in the
// split process of sandi are rare, so we just ignore them in this rules,
// we mainly split on the Vowel form (e.g. "ा") and not the IndepVowel form (e.g. आ )
impl SvarDirgha {
    fn aa_to_a_a() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-a1",
                desc: "आ  => अ + अ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-a2",
                desc: "आ  => आ  + अ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-a3",
                desc: "आ  => अ + आ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-a4",
                desc: "आ  => आ  + आ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
        ]
    }

    fn ii_to_i_i() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-i1",
                desc: "ई => इ + इ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::I)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::I)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-i2",
                desc: "ई => ई + इ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::I)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-i3",
                desc: "ई => इ + ई",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::I)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-i4",
                desc: "ई => ई + ई",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::II)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
        ]
    }

    fn uu_to_u_u() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-u1",
                desc: "ऊ => उ + उ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::U)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-u2",
                desc: "ऊ => ऊ + उ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::U)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-u3",
                desc: "ऊ => उ + ऊ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::U)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-u4",
                desc: "ऊ => ऊ + ऊ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::UU)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::UU)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
        ]
    }

    fn rr_to_r_r() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-r1",
                desc: "ॠ => ॠ + ॠ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::R)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-r2",
                desc: "ॠ => ॠ + ॠ ",
                tag: "6.1.101",
                left: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::R)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-r3",
                desc: "ॠ => ॠ + ॠ ",
                tag: "6.1.101",
                left: Akshara(vec![SoundClass::Vowel(Vowel::R)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::RR)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-dīrgha-r4",
                desc: "ॠ => ॠ + ॠ ",
                tag: "6.1.101",
                left: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::RR)]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarDirgha Rules (Test)");
    }

    #[test]
    fn aa_to_a_a_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("परास्तः", vec![vec!["परा", "अस्तः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn aa_to_a_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("प्रार्थी", vec![vec!["प्र", "अर्थी"]]),
            ("श्रद्धास्ति", vec![vec!["श्रद्धा", "अस्ति"]]),
            ("रामानुजः", vec![vec!["राम", "अनुजः"]]),
            ("शिवालयः", vec![vec!["शिव", "आलयः"]]),
            ("विद्यालयः", vec![vec!["विद्या", "आलयः"]]),
            ("पुस्तकालयः", vec![vec!["पुस्तक", "आलयः"]]),
            ("हिमालयः", vec![vec!["हिम", "आलयः"]]),
            ("कमलाकरः", vec![vec!["कमल", "आकरः"]]),
            ("दैत्यारिः", vec![vec!["दैत्य", "अरिः"]]),
            ("शशाङ्कः", vec![vec!["शश", "अङ्कः"]]),
            ("गौराङ्गः", vec![vec!["गौर", "अङ्गः"]]),
            ("रत्नाकरः", vec![vec!["रत्न", "आकरः"]]),
            ("यथार्थः", vec![vec!["यथा", "अर्थः"]]),
            ("विद्याभ्यासः", vec![vec!["विद्या", "अभ्यासः"]]),
            ("विद्यार्थी", vec![vec!["विद्या", "अर्थी"]]),
            ("परीक्षार्थी", vec![vec!["परीक्षा", "अर्थी"]]),
            ("रामावतारः", vec![vec!["राम", "अवतारः"]]),
            ("सूर्यास्तः", vec![vec!["सूर्य", "अस्तः"]]),
            ("धर्मात्मा", vec![vec!["धर्म", "आत्मा"]]),
            ("परमात्मा", vec![vec!["परम", "आत्मा"]]),
            ("कदापि", vec![vec!["कदा", "अपि"]]),
            ("आत्मानंदः", vec![vec!["आत्मा", "आनंदः"]]),
            ("जन्मान्धः", vec![vec!["जन्म", "अन्धः"]]),
            ("श्रद्धालु", vec![vec!["श्रद्धा", "आलु"]]),
            ("सभाध्यक्षः", vec![vec!["सभा", "अध्यक्षः"]]),
            ("पुरुषार्थः", vec![vec!["पुरुष", "अर्थः"]]),
            ("परमार्थः", vec![vec!["परम", "अर्थः"]]),
            ("पराधीनः", vec![vec!["पर", "अधीनः"]]),
            ("वेदान्तः", vec![vec!["वेद", "अन्तः"]]),
            ("सुषुप्तावस्था", vec![vec!["सुषुप्त", "अवस्था"]]),
            ("अभयारण्यः", vec![vec!["अभय", "अरण्यः"]]),
            ("श्रद्धानन्दः", vec![vec!["श्रद्धा", "आनन्दः"]]),
            ("महाशयः", vec![vec!["महा", "आशयः"]]),
            ("वार्तालापः", vec![vec!["वार्ता", "आलापः"]]),
            ("महामात्यः", vec![vec!["महा", "अमात्यः"]]),
            ("मुक्तावली", vec![vec!["मुक्त", "अवली"]]),
            ("दीपावली", vec![vec!["दीप", "अवली"]]),
            ("प्रश्नावली", vec![vec!["प्रश्न", "अवली"]]),
            ("कृपाकांक्षी", vec![vec!["कृपा", "आकांक्षी"]]),
            ("विस्मयादि", vec![vec!["विस्मय", "आदि"]]),
            ("सत्याग्रहः", vec![vec!["सत्य", "आग्रहः"]]),
            ("प्राणायामः", vec![vec!["प्राण", "आयामः"]]),
            ("शुभारंभः", vec![vec!["शुभ", "आरंभः"]]),
            ("मरणासन्नः", vec![vec!["मरण", "आसन्नः"]]),
            ("शरणागतः", vec![vec!["शरण", "आगतः"]]),
            ("नीलाकाशः", vec![vec!["नील", "आकाशः"]]),
            ("परास्तः", vec![vec!["परा", "अस्तः"]]),
            ("प्रधानाध्यापकः", vec![vec!["प्रधान", "अध्यापकः"]]),
            ("विभागाध्यक्षः", vec![vec!["विभाग", "अध्यक्षः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn aa_to_a_a_anusvara_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("सर्वांगीणः", vec![vec!["सर्व", "अंगीणः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    // special cases w/ sequence of AA + ANUSVARA
    fn aa_to_a_a_anusvara_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सर्वांगीणः", vec![vec!["सर्व", "अंगीणः"]]),
            ("मूल्यांकनः", vec![vec!["मूल्य", "अंकनः"]]),
            ("देहांतः", vec![vec!["देह", "अंतः"]]),
            ("दीक्षांतः", vec![vec!["दीक्षा", "अंतः"]]),
            ("रेखांकितः", vec![vec!["रेखा", "अंकितः"]]),
            ("गीतांजलिः", vec![vec!["गीत", "अंजलिः"]]),
        ];

        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ii_to_i_i_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("श्रीशः", vec![vec!["श्री", "ईशः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ii_to_i_i_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("श्रीशः", vec![vec!["श्री", "ईशः"]]),
            ("गौरीशः", vec![vec!["गौरी", "ईशः"]]),
            ("नदीशः", vec![vec!["नदी", "ईशः"]]),
            ("रजनीशः", vec![vec!["रजनी", "ईशः"]]),
            ("महीशः", vec![vec!["मही", "ईशः"]]),
            ("पृथ्वीश्वरः", vec![vec!["पृथ्वी", "ईश्वरः"]]),
            ("नारीच्छा", vec![vec!["नारी", "इच्छा"]]),
            ("महतीच्छा", vec![vec!["महती", "इच्छा"]]),
            ("नारीश्वरः", vec![vec!["नारी", "ईश्वरः"]]),
            ("गिरीशः", vec![vec!["गिरि", "ईशः"]]),
            ("हरीशः", vec![vec!["हरि", "ईशः"]]),
            ("कवीशः", vec![vec!["कवि", "ईशः"]]),
            ("कपीशः", vec![vec!["कपि", "ईशः"]]),
            ("इतीवः", vec![vec!["इति", "इवः"]]),
            ("अतीवः", vec![vec!["अति", "इवः"]]),
            ("रवीन्द्रः", vec![vec!["रवि", "इन्द्रः"]]),
            ("मुनीन्द्रः", vec![vec!["मुनि", "इन्द्रः"]]),
            ("कवीन्द्रः", vec![vec!["कवि", "इन्द्रः"]]),
            ("फणीन्द्रः", vec![vec!["फणी", "इन्द्रः"]]),
            ("गिरीन्द्रः", vec![vec!["गिरि", "इन्द्रः"]]),
            ("शचीन्द्रः", vec![vec!["शचि", "इन्द्रः"]]),
            ("यतीन्द्रः", vec![vec!["यति", "इन्द्रः"]]),
            ("अभीष्टः", vec![vec!["अभि", "इष्टः"]]),
            ("मुनीश्वरः", vec![vec!["मुनि", "ईश्वरः"]]),
            ("प्रतीक्षा", vec![vec!["प्रति", "ईक्षा"]]),
            ("परीक्षा", vec![vec!["परि", "ईक्षा"]]),
            ("अधीक्षकः", vec![vec!["अधि", "ईक्षकः"]]),
            ("वीक्षणः", vec![vec!["वि", "ईक्षणः"]]),
            ("प्रतीतः", vec![vec!["प्रति", "इतः"]]),
            ("परीक्षितः", vec![vec!["परि", "ईक्षितः"]]),
            ("परीक्षकः", vec![vec!["परि", "ईक्षकः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn uu_to_u_u_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("विष्णूदयः", vec![vec!["विष्णु", "उदयः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn uu_to_u_u_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("विष्णूदयः", vec![vec!["विष्णु", "उदयः"]]),
            ("भानूदयः", vec![vec!["भानु", "उदयः"]]),
            ("भानूष्मा", vec![vec!["भानु", "ऊष्मा"]]),
            ("साधूपदेशः", vec![vec!["साधु", "उपदेशः"]]),
            ("गुरूपदेशः", vec![vec!["गुरु", "उपदेशः"]]),
            ("वधूत्सवः", vec![vec!["वधु", "उत्सवः"]]),
            ("मधूत्तमम्", vec![vec!["मधु", "उत्तमम्"]]),
            ("लघूत्तमम्", vec![vec!["लघु", "उत्तमम्"]]),
            ("विधूर्ध्वम्", vec![vec!["विधु", "उर्ध्वम्"]]),
            ("तरूर्ध्वम्", vec![vec!["तरु", "उर्ध्वम्"]]),
            ("वधूर्मिः", vec![vec!["वधू", "उर्मिः"]]),
            ("लघूर्मिः", vec![vec!["लघु", "उर्मिः"]]),
            ("सिँधूर्मिः", vec![vec!["सिँधु", "उर्मिः"]]),
            ("सूक्तिः", vec![vec!["सु", "उक्तिः"]]),
            ("वधूक्तिः", vec![vec!["वधू", "उक्तिः"]]),
            ("मंजूषा", vec![vec!["मंजु", "उषा"]]),
            ("अनूदितः", vec![vec!["अनु", "उदितः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn rr_to_r_r_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("पितृृणम्", vec![vec!["पितृ", "ऋणम्"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn rr_to_r_r_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("होतृृकारः", vec![vec!["होतृ", "ऋकारः"]]),
            ("पितृृणम्", vec![vec!["पितृ", "ऋणम्"]]),
            ("मातृृणम्", vec![vec!["मातृ", "ऋणम्"]]),
            ("कर्तृृणम्", vec![vec!["कर्तृ", "ऋणम्"]]),
            ("कर्तृृणि", vec![vec!["कर्तृ", "ऋणि"]]),
            ("कर्तृृद्धिः", vec![vec!["कर्तृ", "ऋद्धिः"]]),
            ("धातृृकारः", vec![vec!["धातृ", "ऋकारः"]]),
            ("भर्तृृद्धिः", vec![vec!["भर्तृ", "ऋद्धिः"]]),
            ("होतृृषिः", vec![vec!["होतृ", "ऋषिः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
