use crate::rules::{
    Rule,
    rule::{BaseRule, RuleData, RuleGroup},
};
use orthography::{Adjuncts, Akshara, Consonant, IndependentVowel, SoundClass, Vowel};

pub(crate) struct VisargRutva;

impl RuleGroup for VisargRutva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::rae_to_a_a());

        rls
    }
}

impl VisargRutva {
    fn rae_to_a_a() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "visarga-shatva-rae1",
                desc: "र् = : + अ ",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![SoundClass::Consonant(Consonant::Ra)]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::U)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::UU)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::E)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AI)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::O)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AU)]), false),
                ]),
            })),
            Box::new(BaseRule(RuleData {
                name: "visarga-shatva-rae2",
                desc: "र् = : + अ (INDEP)",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
                merged: Akshara(vec![SoundClass::Consonant(Consonant::Ra)]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::U)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::UU)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::E)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AI)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::O)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AU)]), false),
                ]),
            })),
            Box::new(BaseRule(RuleData {
                name: "visarga-shatva-rae3",
                desc: "र् = : + आ ",
                tag: "8.3.37",
                left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AA)]),
                merged: Akshara(vec![SoundClass::Consonant(Consonant::Ra)]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AA)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::U)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::UU)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::E)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AI)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::O)]), false),
                    (Akshara(vec![SoundClass::Vowel(Vowel::AU)]), false),
                ]),
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VisargRutva Rules (Test)");
    }

    #[test]
    fn rae_to_a_a_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("आशीर्वादः", vec![vec!["आशीः", "वादः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn rae_to_a_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("आशीर्वादः", vec![vec!["आशीः", "वादः"]]),
            ("आयुर्वेदः", vec![vec!["आयुः", "वेदः"]]),
            ("यजुर्वेदः", vec![vec!["यजुः", "वेदः"]]),
            ("दुर्गः", vec![vec!["दुः", "गः"]]),
            ("दुर्गतिः", vec![vec!["दुः", "गतिः"]]),
            ("बहिर्द्वंद्वः", vec![vec!["बहिः", "द्वंद्वः"]]),
            ("दुर्व्यवहारः", vec![vec!["दुः", "व्यवहारः"]]),
            ("धनुर्धरः", vec![vec!["धनुः", "धरः"]]),
            ("भानुरसौ", vec![vec!["भानुः", "असौ"]]),
            ("दुराशा", vec![vec!["दुः", "आशा"]]),
            ("दुरात्मा", vec![vec!["दुः", "आत्मा"]]),
            ("दुर्गंधः", vec![vec!["दुः", "गंधः"]]),
            ("निर्झरः", vec![vec!["निः", "झरः"]]),
            ("दुर्नीतिः", vec![vec!["दुः", "नीतिः"]]),
            ("निर्गुणम्", vec![vec!["निः", "गुणम्"]]),
            ("तैरागतम्", vec![vec!["तैः", "आगतम्"]]),
            ("दुर्गुणम्", vec![vec!["दुः", "गुणम्"]]),
            ("निर्जलम्", vec![vec!["निः", "जलम्"]]),
            ("दुर्लभम्", vec![vec!["दुः", "लभम्"]]),
            ("निर्मलम्", vec![vec!["निः", "मलम्"]]),
            ("दुर्बलः", vec![vec!["दुः", "बलः"]]),
            ("निर्धनः", vec![vec!["निः", "धनः"]]),
            ("धेनुर्गच्छति", vec![vec!["धेनुः", "गच्छति"]]),
            ("निराहारः", vec![vec!["निः", "आहारः"]]),
            ("दुर्वासना", vec![vec!["दुः", "वासना"]]),
            ("दुर्जनम्", vec![vec!["दुः", "जनम्"]]),
            ("निर्भयः", vec![vec!["निः", "भयः"]]),
            ("दुराचारः", vec![vec!["दुः", "आचारः"]]),
            ("पुनरत्र", vec![vec!["पुनः", "अत्र"]]),
            ("निर्यातः", vec![vec!["निः", "यातः"]]),
            ("बहिर्मुखः", vec![vec!["बहिः", "मुखः"]]),
            ("एतैर्भक्षितम्", vec![vec!["एतैः", "भक्षितम्"]]),
            ("निर्विघ्नम्", vec![vec!["निः", "विघ्नम्"]]),
            ("गौरयम्", vec![vec!["गौः", "अयम्"]]),
            ("हरिरयम्", vec![vec!["हरिः", "अयम्"]]),
            ("मुनिरयम्", vec![vec!["मुनिः", "अयम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
