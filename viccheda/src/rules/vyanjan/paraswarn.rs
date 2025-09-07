use crate::rules::{rule::RuleGroup, Rule};

pub(crate) struct VynjanParaswarn;

impl RuleGroup for VynjanParaswarn {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::da_to_anuswara_da());

        rls
    }
}

impl VynjanParaswarn {
    fn da_to_anuswara_da() -> Vec<Box<dyn Rule>> {
        // नियम 1 – अनुस्वार ( -ं ) + कोई भी वर्गीय व्यंजन = उसी वर्ग का पंचम वर्ण ङ्, ञ्, ण्, न्, म्
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanParaswarn Rules (Test)");
    }

    #[test]
    #[ignore]
    fn da_to_anuswar_da_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("अङ्कितः", vec![vec!["अं", "कितः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn da_to_anuswar_da_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("अङ्कितः", vec![vec!["अं", "कितः"]]),
            ("सङ्कल्पः", vec![vec!["सं", "कल्पः"]]),
            ("कुण्ठितः", vec![vec!["कुं", "ठितः"]]),
            ("मुञ्चति", vec![vec!["मुं", "चति"]]),
            ("मुण्डनम्", vec![vec!["मुं", "डनम्"]]),
            ("अञ्चितः", vec![vec!["अं", "चितः"]]),
            ("नन्दति", vec![vec!["नं", "दतिः"]]),
            ("कम्पते", vec![vec!["कं", "पते"]]),
            ("त्वङ्करोषि", vec![vec!["त्वं", "करोषि"]]),
            ("सम्पृक्तौ", vec![vec!["सं", "पृक्तौ"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
