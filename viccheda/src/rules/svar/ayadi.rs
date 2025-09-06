use crate::rules::{
    rule::{AllKindRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarAyadi;

impl RuleGroup for SvarAyadi {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ayae_to_e_vowels());
        rls.extend(Self::aayae_to_ai_vowels());
        rls.extend(Self::aavae_to_o_vowels());

        rls
    }
}

impl SvarAyadi {
    fn ayae_to_e_vowels() -> Vec<Box<dyn Rule>> {
        vec![Box::new(AllKindRule {
            kind: SoundClass::AllVowel,
            data: RuleData {
                name: "savarṇa-ayādi-ayae1",
                desc: "अय् = ए + Vowel",
                tag: "6.1.78",
                left: Akshara(vec![SoundClass::Vowel(Vowel::E)]),
                right: Akshara(vec![]),
                merged: Akshara(vec![SoundClass::Consonant(Consonant::Ya)]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VISARGA)]), true),
                    (
                        Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                        true,
                    ),
                ]),
            },
        })]
    }

    fn aayae_to_ai_vowels() -> Vec<Box<dyn Rule>> {
        vec![Box::new(AllKindRule {
            kind: SoundClass::AllVowel,
            data: RuleData {
                name: "savarṇa-ayādi-aayae1",
                desc: "आय् = ऐ + Vowel",
                tag: "6.1.78",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                right: Akshara(vec![]),
                merged: Akshara(vec![
                    SoundClass::Vowel(Vowel::AA),
                    SoundClass::Consonant(Consonant::Ya),
                ]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            },
        })]
    }

    fn aavae_to_o_vowels() -> Vec<Box<dyn Rule>> {
        vec![Box::new(AllKindRule {
            kind: SoundClass::AllVowel,
            data: RuleData {
                name: "savarṇa-ayādi-aavae1",
                desc: "अव् = ओ  + Vowel",
                tag: "6.1.78",
                left: Akshara(vec![SoundClass::Vowel(Vowel::O)]),
                right: Akshara(vec![]),
                merged: Akshara(vec![SoundClass::Consonant(Consonant::Va)]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    true,
                )]),
            },
        })]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarGuna Rules (Test)");
    }

    #[test]
    fn ayae_to_e_vowel_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("संचयः", vec![vec!["संचे", "अः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ayae_to_e_vowel_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("नयनम्", vec![vec!["ने", "अनम्"]]),
            ("चयनम्", vec![vec!["चे", "अनम्"]]),
            ("शयनम्", vec![vec!["शे", "अनम्"]]),
            ("संचयः", vec![vec!["संचे", "अः"]]),
            ("आश्रयः", vec![vec!["आश्रे", "अः"]]),
            ("उदयः", vec![vec!["उदे", "अः"]]),
            ("प्रलयः", vec![vec!["प्रले", "अः"]]),
            ("कवये", vec![vec!["कवे", "ए"]]),
            ("हरयेहि", vec![vec!["हरे", "एहि"]]),
            ("मुनये", vec![vec!["मुने", "ए"]]),
            ("हरये", vec![vec!["हरे", "ए"]]),
            ("शयितः", vec![vec!["शे", "इतः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn aayae_to_e_vowel_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("गायकः", vec![vec!["गै", "अकः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn aayae_to_e_vowel_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("गायकः", vec![vec!["गै", "अकः"]]),
            ("नायकः", vec![vec!["नै", "अकः"]]),
            ("सायकः", vec![vec!["सै", "अकः"]]),
            ("विनायकः", vec![vec!["विनै", "अकः"]]),
            ("विधायकः", vec![vec!["विधै", "अकः"]]),
            ("दायकः", vec![vec!["दै", "अकः"]]),
            ("गायन्ति", vec![vec!["गै", "अन्ति"]]),
            ("गायनम्", vec![vec!["गै", "अनम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn aavae_to_o_vowel_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("विष्णवे", vec![vec!["विष्णो", "ए"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn aavae_to_o_vowel_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("विष्णवे", vec![vec!["विष्णो", "ए"]]),
            ("गवे", vec![vec!["गो", "ए"]]),
            ("गवीशः", vec![vec!["गो", "ईशः"]]),
            ("पवनः", vec![vec!["पो", "अनः"]]),
            ("भवनम्", vec![vec!["भो", "अनम्"]]),
            ("पवित्रम्", vec![vec!["पो", "इत्रम्"]]),
            ("हवनम्", vec![vec!["हो", "अनम्"]]),
            ("लवनः", vec![vec!["लो", "अनः"]]),
            ("भानवे", vec![vec!["भानो", "ए"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
