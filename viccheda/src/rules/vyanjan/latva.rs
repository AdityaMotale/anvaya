use crate::rules::{rule::RuleGroup, Rule};

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
        // नियम 1 – तवर्ग त्, थ्, द्, ध् + ल्  = ल्
        vec![]
    }

    fn lan_to_nae_lae() -> Vec<Box<dyn Rule>> {
        // नियम 2 – न्  + ल्  = लँ
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanLatva Rules (Test)");
    }

    #[test]
    #[ignore]
    fn lae_to_dentals_lae_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("तल्लीनः", vec![vec!["तत्", "लीनः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
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
    #[ignore]
    fn lan_to_nae_lae_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> =
            vec![("विद्वाँल्लिखति", vec![vec!["विद्वान्", "लिखति"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn lan_to_nae_lae_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("विद्वाँल्लिखति", vec![vec!["विद्वान्", "लिखति"]]),
            ("गुणवाँल्लुण्ठितः", vec![vec!["गुणवान्", "लुण्ठित"]]),
            ("धीमाँल्लिखति", vec![vec!["धीमान्", "लिखति"]]),
            ("महाँल्लाभः", vec![vec!["महान्", "लाभः"]]),
            ("हसँल्लिखति", vec![vec!["हसन्", "लिखति"]]),
            ("खादँल्लसति", vec![vec!["खादन्", "लसति"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
