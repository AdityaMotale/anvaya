use crate::rules::{
    rule::{BaseRule, MultiOptRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel, DENTALS, PALATALS};

pub(crate) struct VynjanShchutva;

impl RuleGroup for VynjanShchutva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(VynjanShchutva::sh_to_s_sh());
        rls.extend(VynjanShchutva::palatals_to_dentals());

        rls
    }
}

impl VynjanShchutva {
    fn sh_to_s_sh() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-ṣṭutva-shae1",
            desc: "श् = स् + अ ",
            tag: "8.4.44",
            left: Akshara(vec![
                SoundClass::Consonant(Consonant::Sa),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Sha),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            special_sequence: None,
        }))]
    }

    fn palatals_to_dentals() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = PALATALS
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
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
                name: "vyanjan-ṣṭutva-palatals1",
                desc: "palatals = dentals + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: None,
            },
        })]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanShchutva Rules (Test)");
    }

    #[test]
    fn sh_to_s_sh_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामश्चिनोति", vec![vec!["रामस्", "चिनोति"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn sh_to_s_sh_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("रामश्चिनोति", vec![vec!["रामस्", "चिनोति"]]),
            ("हरिश्शेते", vec![vec!["हरिस्", "शेते"]]),
            ("बालकश्शेते", vec![vec!["बालकस्", "शेते"]]),
            ("शिशुश्शेते", vec![vec!["शिशुस्", "शेते"]]),
            ("रामश्च", vec![vec!["रामस्", "च"]]),
            ("कश्चित्", vec![vec!["कस्", "चित्"]]),
            ("मनश्चलति", vec![vec!["मनस्", "चलति"]]),
            ("मनश्चञ्चलम्", vec![vec!["मनस्", "चञ्चलम्"]]),
            ("निश्छलः", vec![vec!["निस्", "छलः"]]),
            ("निश्चयः", vec![vec!["निस्", "चयः"]]),
            ("दुश्चरित्रः", vec![vec!["दुस्", "चरित्रः"]]),
            ("श्यामश्छात्रः", vec![vec!["श्यामस्", "छात्रः"]]),
            ("सूर्यश्छन्नः", vec![vec!["सूर्यस्", "छन्नः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    fn palatals_to_dentals_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("सच्चित्", vec![vec!["सत्", "चित्"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn palatals_to_dentals_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सच्चित्", vec![vec!["सत्", "चित्"]]),
            ("सच्चेष्टा", vec![vec!["सत्", "चेष्टा"]]),
            ("सच्छात्रः", vec![vec!["सत्", "छात्रः"]]),
            ("सच्चरित्रम्", vec![vec!["सत्", "चरित्रम्"]]),
            ("उच्चारणम्", vec![vec!["उत्", "चारणम्"]]),
            ("महच्चित्रम्", vec![vec!["महत्", "चित्रम्"]]),
            ("जगच्चक्रम्", vec![vec!["जगत्", "चक्रम्"]]),
            ("तच्चलचित्रम्", vec![vec!["तत्", "चलचित्रम्"]]),
            ("तच्चः", vec![vec!["तत्", "चः"]]),
            ("उच्चः", vec![vec!["उत्", "चः"]]),
            ("भगवच्शक्तिः", vec![vec!["भगवत्", "शक्तिः"]]),
            ("तच्छविः", vec![vec!["तत्", "छविः"]]),
            ("उज्ज्वलः", vec![vec!["उद्", "ज्वलः"]]),
            ("उज्जयिनी", vec![vec!["उद्", "जयिनी"]]),
            ("सज्जनं", vec![vec!["सद्", "जनं"]]),
            ("तज्जयं", vec![vec!["तद्", "जयं"]]),
            ("तज्जयः", vec![vec!["तद्", "जयः"]]),
            ("बृहज्जनः", vec![vec!["बृहद्", "जनः"]]),
            ("विपज्जालः", vec![vec!["विपद्", "जालः"]]),
            ("विद्युच्चालकः", vec![vec!["विद्युत्", "चालकः"]]),
            ("सुहृज्जगाम", vec![vec!["सुहृद्", "जगाम"]]),
            ("विद्युच्छटा", vec![vec!["विद्युत्", "छटा"]]),
            ("शाङ्गिञ्जय", vec![vec!["शाङ्गिन्", "जय"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
