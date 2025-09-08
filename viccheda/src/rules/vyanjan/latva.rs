use crate::rules::{
    Rule,
    rule::{BaseRule, MultiOptRule, RuleData, RuleGroup},
};
use orthography::{Adjuncts, Akshara, Consonant, DENTALS, SoundClass, Vowel};

pub(crate) struct VynjanLatva;

impl RuleGroup for VynjanLatva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::lae_to_dentals_lae());
        rls.extend(Self::lan_to_nae_lae());

        rls
    }
}

impl VynjanLatva {
    fn lae_to_dentals_lae() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..5)
            .map(|_| {
                Akshara(vec![
                    SoundClass::Consonant(Consonant::La),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        let swap_list: Vec<Akshara> = DENTALS
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        assert!(merged_list.len() == swap_list.len());

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-latva-lae",
                desc: "ल् = त्, थ्, द्, ध् + ल",
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

    fn lan_to_nae_lae() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-latva-lan",
            desc: "लँ = न्  + ल् ",
            tag: "8.4.44",
            left: Akshara(vec![
                SoundClass::Consonant(Consonant::Na),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Adjuncts(Adjuncts::CHANDRABINDU),
                SoundClass::Consonant(Consonant::La),
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
        let _ = crate::init_logger("VyanjanLatva Rules (Test)");
    }

    #[test]
    fn lae_to_dentals_lae_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("तल्लीनः", vec![vec!["तत्", "लीनः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn lae_to_dentals_lae_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("तल्लीनः", vec![vec!["तत्", "लीनः"]]),
            ("तल्लयः", vec![vec!["तद्", "लयः"]]),
            ("जगल्लयः", vec![vec!["जगत्", "लयः"]]),
            ("पल्लवः", vec![vec!["पद्", "लवः"]]),
            ("उल्लासः", vec![vec!["उत्", "लासः"]]),
            ("जहल्लक्षणा", vec![vec!["जहत्", "लक्षणा"]]),
            ("विलसल्लङ्का", vec![vec!["विलसत्", "लङ्का"]]),
            ("उल्लेखः", vec![vec!["उत्", "लेखः"]]),
            ("उल्लङ्घनम्", vec![vec!["उत्", "लङ्घनम्"]]),
            ("उल्लिखितम्", vec![vec!["उत्", "लिखितम्"]]),
            ("भगवल्लीनः", vec![vec!["भगवत्", "लीनः"]]),
            ("विद्युल्लता", vec![vec!["विद्युत्", "लता"]]),
            ("जगल्लजते", vec![vec!["जगत्", "लजते"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn lan_to_nae_lae_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> =
            vec![("विद्वाँल्लिखति", vec![vec!["विद्वान्", "लिखति"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn lan_to_nae_lae_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("विद्वाँल्लिखति", vec![vec!["विद्वान्", "लिखति"]]),
            ("गुणवाँल्लुण्ठितः", vec![vec!["गुणवान्", "लुण्ठितः"]]),
            ("धीमाँल्लिखति", vec![vec!["धीमान्", "लिखति"]]),
            ("महाँल्लाभः", vec![vec!["महान्", "लाभः"]]),
            ("हसँल्लिखति", vec![vec!["हसन्", "लिखति"]]),
            ("खादँल्लसति", vec![vec!["खादन्", "लसति"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
