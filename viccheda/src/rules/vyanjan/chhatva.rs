use crate::rules::{rule::RuleGroup, Rule};

pub(crate) struct VynjanChhatva;

impl RuleGroup for VynjanChhatva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::chha_to_varna_sha());

        rls
    }
}

impl VynjanChhatva {
    fn chha_to_varna_sha() -> Vec<Box<dyn Rule>> {
        // नियम 1 – वर्ग का  प्रथम  द्वितीय तृतीय  चतुर्थवर्ण या र् ल् व्  या  ह्श्  छ्
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanChhatva Rules (Test)");
    }

    #[test]
    #[ignore]
    fn chha_to_varna_sha_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("उच्छृंखलः", vec![vec!["उत्", "शृंखलः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn chha_to_varna_sha_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("उच्छृंखलः", vec![vec!["उत्", "शृंखलः"]]),
            ("तच्छ्रुत्वा", vec![vec!["तत्", "श्रुत्वा"]]),
            ("तच्छरीरम्", vec![vec!["तत्", "शरीरम्"]]),
            ("एतच्छोभनम्", vec![vec!["एतत्", "शोभनम्"]]),
            ("उच्छिष्टः", vec![vec!["उत्", "शिष्टः"]]),
            ("तच्छिव", vec![vec!["तत्", "शिव"]]),
            ("उच्छ्वासः", vec![vec!["उत्", "श्वासः"]]),
            ("सच्छासनम्", vec![vec!["सत्", "शासनम्"]]),
            ("सच्छास्त्रम्", vec![vec!["सत्", "शास्त्रम्"]]),
            ("उच्छ्वसनम्", vec![vec!["उत्", "श्वसनम्"]]),
            ("शरच्छशिः", vec![vec!["शरत्", "शशिः"]]),
            ("जगच्छान्तिः", vec![vec!["जगत्", "शान्तिः"]]),
            ("तच्छंकरः", vec![vec!["तत्", "शंकरः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
