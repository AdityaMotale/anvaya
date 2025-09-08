use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VynjanChatrva;

impl RuleGroup for VynjanChatrva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::char_to_zal_khal());

        rls
    }
}

impl VynjanChatrva {
    fn char_to_zal_khal() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-chatrva-char1",
            desc: "त् = द् + अ ",
            tag: "8.4.44",
            left: Akshara(vec![
                SoundClass::Consonant(Consonant::Da),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Ta),
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
        let _ = crate::init_logger("VyanjanChatrva Rules (Test)");
    }

    #[test]
    #[ignore]
    fn char_to_zal_khal_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("सत्कारः", vec![vec!["सद्", "कारः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn char_to_zal_khal_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सत्कारः", vec![vec!["सद्", "कारः"]]),
            ("विपत्कालः", vec![vec!["विपद्", "कालः"]]),
            ("शरत्कालः", vec![vec!["शरद्", "कालः"]]),
            ("सम्पत्समयः", vec![vec!["सम्पद्", "समयः"]]),
            ("उत्पन्नः", vec![vec!["उद्", "पन्नः"]]),
            ("उत्तप्तः", vec![vec!["उद्", "तप्तः"]]),
            ("उत्तमः", vec![vec!["उद्", "तमः"]]),
            ("उत्कर्षः", vec![vec!["उद्", "कर्षः"]]),
            ("उत्कीर्णः", vec![vec!["उद्", "कीर्णः"]]),
            ("उत्पत्तिः", vec![vec!["उद्", "पत्तिः"]]),
            // ("उत्थानम्", vec![vec!["उद्", "स्थानम्"]]),
            ("तत्क्षणः", vec![vec!["तद्", "क्षणः"]]),
            ("तत्परः", vec![vec!["तद्", "परः"]]),
            ("तत्पुरुषः", vec![vec!["तद्", "पुरुषः"]]),
            ("तच्छविः", vec![vec!["तद्", "छविः"]]),
            ("तच्छिवः", vec![vec!["तद्", "शिवः"]]),
            ("संसत्सदस्यः", vec![vec!["संसद्", "सदस्यः"]]),
            ("आपत्तिः", vec![vec!["आपद्", "तिः"]]),
            ("लप्स्यते", vec![vec!["लभ्", "स्यते"]]),
            ("दिक्पालः", vec![vec!["दिग्", "पालः"]]),
            ("अस्मत्पुत्रः", vec![vec!["अस्मद्", "पुत्रः"]]),
            ("विराट्पुरुषः", vec![vec!["विराड्", "षुरुषः"]]),
            ("षट्खाद्यानि", vec![vec!["षड्", "खाद्यानि"]]),
            ("भेत्तुम्", vec![vec!["भेद्", "तुम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
