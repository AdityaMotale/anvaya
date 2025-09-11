use crate::rules::{
    rule::{BaseRule, MultiOptRule, RuleData, RuleGroup},
    Adjuncts, Rule,
};
use orthography::{Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VynjanParaswarn;

impl RuleGroup for VynjanParaswarn {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::paraswarn());
        rls.extend(Self::nn_to_n_n());
        rls.extend(Self::tj_to_m_j());

        rls
    }
}

impl VynjanParaswarn {
    fn paraswarn() -> Vec<Box<dyn Rule>> {
        let merge_items: [Consonant; 5] = [
            Consonant::Nga,
            Consonant::Nna,
            Consonant::Na,
            Consonant::Ma,
            Consonant::Nya,
        ];

        let merged_list: Vec<Akshara> = merge_items
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        let swap_list: Vec<Akshara> = (0..5)
            .map(|_| Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]))
            .collect();

        assert!(merged_list.len() == swap_list.len());

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-paraswarn",
                desc: "ङ्, ण्, न्, म् = क्, ठ्, त्, प् + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::ANUSVARA)]),
                    false,
                )]),
            },
        })]
    }

    fn nn_to_n_n() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-paraswarn-nn",
            desc: "न्न= न + अ ",
            tag: "8.3.37",
            left: Akshara(vec![SoundClass::Consonant(Consonant::Na)]),
            right: Akshara(vec![SoundClass::Consonant(Consonant::Na)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Na),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
                SoundClass::Consonant(Consonant::Na),
            ]),
            special_sequence: None,
        }))]
    }

    fn tj_to_m_j() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-paraswarn-nn",
            desc: "ञ्ज = म + ज ",
            tag: "8.3.37",
            left: Akshara(vec![SoundClass::Consonant(Consonant::Ma)]),
            right: Akshara(vec![SoundClass::Consonant(Consonant::Ja)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Nya),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
                SoundClass::Consonant(Consonant::Ja),
            ]),
            special_sequence: None,
        }))]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanAnunasik Rules (Test)");
    }

    #[test]
    fn nn_to_n_n_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("तान्निबोध", vec![vec!["तान", "निबोध"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn tj_to_m_j_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("समितिञ्जय", vec![vec!["समितिम", "जय"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn paraswarn_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("अङ्कितः", vec![vec!["अं", "कितः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn paraswarn_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("अङ्कितः", vec![vec!["अं", "कितः"]]),
            ("सङ्कल्पः", vec![vec!["सं", "कल्पः"]]),
            ("कुण्ठितः", vec![vec!["कुं", "ठितः"]]),
            ("मुञ्चति", vec![vec!["मुं", "चति"]]),
            ("मुण्डनम्", vec![vec!["मुं", "डनम्"]]),
            ("अञ्चितः", vec![vec!["अं", "चितः"]]),
            ("कम्पते", vec![vec!["कं", "पते"]]),
            ("त्वङ्करोषि", vec![vec!["त्वं", "करोषि"]]),
            ("सम्पृक्तौ", vec![vec!["सं", "पृक्तौ"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
