use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VisargSatva;

impl RuleGroup for VisargSatva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::sae_to_vis_a());

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
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarPoorvaroop Rules (Test)");
    }

    #[test]
    fn sae_to_vis_a_test_debug() {
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
}
