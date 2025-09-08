use crate::rules::{
    Rule,
    rule::{BaseRule, RuleData, RuleGroup},
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VisargSatva;

impl RuleGroup for VisargSatva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::sae_to_vis_a());
        rls.extend(Self::shae_to_vis_c_ch_sh());
        rls.extend(Self::ssae_to_vis_a());

        rls
    }
}

impl VisargSatva {
    fn sae_to_vis_a() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "visarga-visarjanīyasya-saḥ-sae1",
            desc: "स् = : + अ ",
            tag: "8.3.37",
            left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Sa),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            special_sequence: None,
        }))]
    }

    fn shae_to_vis_c_ch_sh() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "visarga-visarjanīyasya-saḥ-shae1",
                desc: "श् (श्च) = : + च् ",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::Consonant(Consonant::Cha)]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Sha),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                    SoundClass::Consonant(Consonant::Cha),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "visarga-visarjanīyasya-saḥ-shae2",
                desc: "श् (श्च) = : + छ् ",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::Consonant(Consonant::Chha)]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Sha),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                    SoundClass::Consonant(Consonant::Cha),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
            Box::new(BaseRule(RuleData {
                name: "visarga-visarjanīyasya-saḥ-shae3",
                desc: "श् (श्च) = : + श् ",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::Consonant(Consonant::Sha)]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Sha),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                    SoundClass::Consonant(Consonant::Cha),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            })),
        ]
    }

    fn ssae_to_vis_a() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "visarga-visarjanīyasya-saḥ-ssae1",
            desc: "ष् = : + ट् ",
            tag: "8.3.37",
            left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Ssa),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            special_sequence: Some(vec![(
                Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                true,
            )]),
        }))]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VisargSatva Rules (Test)");
    }

    #[test]
    fn sae_to_vis_a_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("चन्द्रस्तमः", vec![vec!["चन्द्रः", "तमः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn sae_to_vis_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("चन्द्रस्तमः", vec![vec!["चन्द्रः", "तमः"]]),
            ("नमस्ते", vec![vec!["नमः", "ते"]]),
            ("शिरस्त्राणः", vec![vec!["शिरः", "त्राणः"]]),
            ("निस्तारः", vec![vec!["निः", "तारः"]]),
            ("निस्तेजः", vec![vec!["निः", "तेजः"]]),
            ("नमस्तरतिः", vec![vec!["नमः", "तरतिः"]]),
            ("पुरस्कारः", vec![vec!["पुरः", "कारः"]]),
            ("भास्करः", vec![vec!["भाः", "करः"]]),
            ("दुस्थकारः", vec![vec!["दुः", "थकारः"]]),
            ("दुस्साहसः", vec![vec!["दुः", "साहसः"]]),
            ("नमस्कारः", vec![vec!["नमः", "कारः"]]),
            ("बृहस्पतिः", vec![vec!["बृहः", "पतिः"]]),
            ("मनस्तापः", vec![vec!["मनः", "तापः"]]),
            ("निस्संदेहः", vec![vec!["निः", "संदेहः"]]),
            ("श्रेयस्करः", vec![vec!["श्रेयः", "करः"]]),
            ("दुस्तरः", vec![vec!["दुः", "तरः"]]),
            ("निस्संतानः", vec![vec!["निः", "संतानः"]]),
            ("निस्संकोचः", vec![vec!["निः", "संकोचः"]]),
            ("ततस्तेषु", vec![vec!["ततः", "तेषु"]]),
            ("धन्यास्तु", vec![vec!["धन्याः", "तु"]]),
            ("मनस्तोषः", vec![vec!["मनः", "तोषः"]]),
            ("भवतस्सर्वदा", vec![vec!["भवतः", "सर्वदा"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn shae_to_vis_c_ch_sh_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामश्च", vec![vec!["रामः", "च"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn shae_to_vis_c_ch_sh_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("निश्चलः", vec![vec!["निः", "छलः"]]),
            ("रामश्च", vec![vec!["रामः", "च"]]),
            ("आश्चर्यः", vec![vec!["आः", "चर्यः"]]),
            ("दुश्चरित्रः", vec![vec!["दुः", "चरित्रः"]]),
            ("निश्चिंतः", vec![vec!["निः", "चिंतः"]]),
            ("निश्चयः", vec![vec!["निः", "चयः"]]),
            ("अन्तश्चक्षुः", vec![vec!["अन्तः", "चक्षुः"]]),
            ("दुश्चक्रः", vec![vec!["दुः", "चक्रः"]]),
            ("हरिश्चंद्रः", vec![vec!["हरिः", "चंद्रः"]]),
            ("पुनश्च", vec![vec!["पुनः", "च"]]),
            ("व्याकुलश्चलितः", vec![vec!["व्याकुलः", "चलितः"]]),
            ("जन्मभूमिश्च", vec![vec!["जन्मभूमिः", "च"]]),
            ("मातुश्च", vec![vec!["मातुः", "च"]]),
            ("मधुरश्च", vec![vec!["मधुरः", "च"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn ssae_to_vis_a_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("चतुष्टीका", vec![vec!["चतुः", "टीका"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ssae_to_vis_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("चतुष्टीका", vec![vec!["चतुः", "टीका"]]),
            ("रामष्टीकते", vec![vec!["रामः", "टीकते"]]),
            ("धनुष्टंकारः", vec![vec!["धनुः", "टंकारः"]]),
            ("निष्ठुरः", vec![vec!["निः", "ठुरः"]]),
            ("ततष्ठकारः", vec![vec!["ततः", "ठकारः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
