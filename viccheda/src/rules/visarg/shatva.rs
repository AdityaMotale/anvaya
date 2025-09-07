use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VisargShatva;

impl RuleGroup for VisargShatva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ssae_to_e_u_vis_a());

        rls
    }
}

impl VisargShatva {
    fn ssae_to_e_u_vis_a() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "visarga-shatva-ssae1",
                desc: "ष् = इ + : + अ ",
                tag: "8.3.37",
                left: Akshara(vec![
                    SoundClass::Vowel(Vowel::E),
                    SoundClass::Adjuncts(Adjuncts::VISARGA),
                ]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Ssa),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "visarga-shatva-ssae2",
                desc: "ष् = उ + : + अ ",
                tag: "8.3.37",
                left: Akshara(vec![
                    SoundClass::Vowel(Vowel::U),
                    SoundClass::Adjuncts(Adjuncts::VISARGA),
                ]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Ssa),
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
        let _ = crate::init_logger("VisargShatva Rules (Test)");
    }

    #[test]
    fn ssae_to_e_u_vis_a_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("दुष्करः", vec![vec!["दुः", "करः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ssae_to_e_u_vis_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("दुष्करः", vec![vec!["दुः", "करः"]]),
            ("निष्कलंकः", vec![vec!["निः", "कलंकः"]]),
            ("निष्पक्षः", vec![vec!["निः", "पक्षः"]]),
            ("निष्फलः", vec![vec!["निः", "फलः"]]),
            ("निष्पापः", vec![vec!["निः", "पापः"]]),
            ("निष्कपटः", vec![vec!["निः", "कपटः"]]),
            ("बहिष्कृतः", vec![vec!["बहिः", "कृतः"]]),
            ("आविष्कारः", vec![vec!["आविः", "कारः"]]),
            ("दुष्कर्मः", vec![vec!["दुः", "कर्मः"]]),
            ("चतुष्पादः", vec![vec!["चतुः", "पादः"]]),
            ("दुष्प्रभावः", vec![vec!["दुः", "प्रभावः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
