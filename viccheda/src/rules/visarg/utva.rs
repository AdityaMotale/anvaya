use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, IndependentVowel, SoundClass, Vowel};

pub(crate) struct VisargUtva;

impl RuleGroup for VisargUtva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::avagraha());

        rls
    }
}

impl VisargUtva {
    fn avagraha() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "visarga-utva-avagraha",
            desc: "ऽ = अ  + अ (INDEP)",
            tag: "8.3.37",
            left: Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]),
            right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
            merged: Akshara(vec![
                SoundClass::Vowel(Vowel::O),
                SoundClass::Adjuncts(Adjuncts::AVAGRAHA),
            ]),
            special_sequence: None,
        }))]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VisargUtva Rules (Test)");
    }

    #[test]
    fn avagraha_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("कोऽपि", vec![vec!["कः", "अपि"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn avagraha_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("कोऽपि", vec![vec!["कः", "अपि"]]),
            ("सोऽपि", vec![vec!["सः", "अपि"]]),
            ("सोऽवदत्", vec![vec!["सः", "अवदत्"]]),
            ("रामोऽवदत्", vec![vec!["रामः", "अवदत्"]]),
            ("नृपोऽवदत्", vec![vec!["नृपः", "अवदत्"]]),
            ("रामोऽयम्", vec![vec!["रामः", "अयम्"]]),
            ("देवोऽयम्", vec![vec!["देवः", "अयम्"]]),
            ("छात्रोऽयम्", vec![vec!["छात्रः", "अयम्"]]),
            ("कोऽत्र", vec![vec!["कः", "अत्र"]]),
            ("नृपोऽस्ति", vec![vec!["नृपः", "अस्ति"]]),
            ("शिवोऽर्चः", vec![vec!["शिवः", "अर्चः"]]),
            ("प्रथमोऽध्यायः", vec![vec!["प्रथमः", "अध्यायः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
