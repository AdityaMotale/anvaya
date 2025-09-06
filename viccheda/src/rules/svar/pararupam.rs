use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Akshara, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarPararupam;

impl RuleGroup for SvarPararupam {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::e_to_a_e());

        rls
    }
}

impl SvarPararupam {
    fn e_to_a_e() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "savarṇa-pararūpam-e1",
            desc: "ए = अ + ए",
            tag: "6.1.94",
            left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::E)]),
            merged: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
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
    fn e_to_a_e_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("प्रेजते", vec![vec!["प्र", "एजते"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn e_to_a_e_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("प्रेजते", vec![vec!["प्र", "एजते"]]),
            ("प्रेषयति", vec![vec!["प्र", "एषयति"]]),
            ("उपेहि", vec![vec!["उप", "एहि"]]),
            ("उपेजते", vec![vec!["उप", "एजते"]]),
            ("उपेषते", vec![vec!["उप", "एषते"]]),
            ("प्रेषणीयम्", vec![vec!["प्र", "एषणीयम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
