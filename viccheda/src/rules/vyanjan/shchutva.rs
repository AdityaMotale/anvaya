use crate::rules::{rule::RuleGroup, Rule};

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
        // नियम 1 – स् + श् = श्
        vec![]
    }

    fn palatals_to_dentals() -> Vec<Box<dyn Rule>> {
        // नियम 2 – तवर्ग (त्, थ्, द्, ध्, न्) + चवर्ग (च्, छ्, ज, झ्, ञ्) = चवर्ग (च्, छ्, ज, झ्, ञ्)
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanShchutva Rules (Test)");
    }

    #[test]
    #[ignore]
    fn sh_to_s_sh_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामश्चिनोति", vec![vec!["रामस", "चिनोति"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn sh_to_s_sh_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("रामश्चिनोति", vec![vec!["रामस", "चिनोति"]]),
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
    #[ignore]
    fn palatals_to_dentals_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("सच्चित्", vec![vec!["सत्", "चित"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn palatals_to_dentals_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सच्चित्", vec![vec!["सत्", "चित"]]),
            ("सच्चेष्टा", vec![vec!["सत्", "चेष्टा"]]),
            ("सच्छात्रः", vec![vec!["सत्", "छात्रः"]]),
            ("सच्चरित्रम्", vec![vec!["सत्", "चरित्रम्"]]),
            ("उच्चारणम्", vec![vec!["उत्", "चारणम्"]]),
            ("महच्चित्रम्", vec![vec!["महत्", "चित्रम्"]]),
            ("जगच्चक्रम्", vec![vec!["जगत्", "च्रकम्"]]),
            ("तच्चलचित्रम्", vec![vec!["तत्", "चलचित्रम्"]]),
            ("तच्चः", vec![vec!["तत्", "चः"]]),
            ("उच्चः", vec![vec!["उत्", "चः"]]),
            ("भगवच्शक्तिः", vec![vec!["भगवत्", "शक्तिः"]]),
            ("तच्छविः", vec![vec!["तत्", "छविः"]]),
            ("उज्ज्वलः", vec![vec!["उद्", "ज्वलः"]]),
            ("उज्जयिनी", vec![vec!["उद्", "जयिनी"]]),
            ("सज्जनं", vec![vec!["सद्", "जन:"]]),
            ("तज्जयं", vec![vec!["तद्", "जय:"]]),
            ("तज्जयः", vec![vec!["तद्", "जयः"]]),
            ("बृहज्जनः", vec![vec!["बृहत्", "जनः"]]),
            ("विपज्जालः", vec![vec!["विपत्", "जालः"]]),
            ("विद्युच्चालकः", vec![vec!["विद्युत्", "चालकः"]]),
            ("सुहृज्जगाम", vec![vec!["सुहृद्", "जगाम"]]),
            ("शरच्चिन्द्रिका", vec![vec!["शरत्", "चन्द्रिका"]]),
            ("विद्युच्छटा", vec![vec!["विद्युत्", "छटा"]]),
            ("शाङ्गिञ्जय", vec![vec!["शाङ्गिन", "जय"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
