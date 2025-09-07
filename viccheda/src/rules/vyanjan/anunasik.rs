use crate::rules::{rule::RuleGroup, Rule};

pub(crate) struct VynjanAnunasik;

impl RuleGroup for VynjanAnunasik {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::da_to_consonant_da());

        rls
    }
}

impl VynjanAnunasik {
    fn da_to_consonant_da() -> Vec<Box<dyn Rule>> {
        // नियम 1 – ह् वर्ण को छोडकर कोई भी व्यंजन  + ङ्, ञ्, ण्, न्, म् = उसी वर्ग का पंचम वर्ण ङ्, ञ्, ण्, न्, म्
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanAnunasik Rules (Test)");
    }

    #[test]
    #[ignore]
    fn da_to_consonant_da_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("दिङ्नाथः", vec![vec!["दिक्", "नाथः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn da_to_consonant_da_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("दिङ्नाथः", vec![vec!["दिक्", "नाथः"]]),
            ("षण्मयूखाः", vec![vec!["षट्", "मयूखाः"]]),
            ("षण्मुखः", vec![vec!["षट्", "मुखः"]]),
            ("विभ्रन्नः", vec![vec!["विभ्रत्", "नः"]]),
            ("उन्नतिः", vec![vec!["उत्", "नतिः"]]),
            ("यन्माध्यमेन", vec![vec!["यत्", "माध्यमेन"]]),
            ("एतन्मुरारिः", vec![vec!["एतत्", "मुरारिः"]]),
            ("जगन्नाथः", vec![vec!["जगत्", "नाथः"]]),
            ("वाङ्मयः", vec![vec!["वाक्", "मयः"]]),
            ("वाङ्नियमः", vec![vec!["वाक्", "नियमः"]]),
            ("वाङ्निपुणः", vec![vec!["वाक्", "निपुणः"]]),
            ("दिङ्नागः", vec![vec!["दिक्", "नागः"]]),
            ("धिङ्मूर्खः", vec![vec!["धिक्", "मूर्खः"]]),
            ("तन्मात्रम्", vec![vec!["तद्", "मात्रम्"]]),
            ("तन्निरुप्यताम्", vec![vec!["तत्", "निरुप्यताम्"]]),
            ("सन्निधानम्", vec![vec!["सत्", "निधानम्"]]),
            ("सन्नियमम्", vec![vec!["सत्", "नियमम्"]]),
            ("वाङ्मूलम्", vec![vec!["वाग्", "मूलम्"]]),
            ("अम्मयम्", vec![vec!["अप्", "मयम्"]]),
            ("तन्नयति", vec![vec!["तत्", "नयति"]]),
            ("विद्युन्नगरी", vec![vec!["विद्युत्", "नगरी"]]),
            ("चिन्मयम्", vec![vec!["चित्", "मयम्"]]),
            ("तन्नयनम्", vec![vec!["तत्", "नयनम्"]]),
            ("सन्मार्गम्", vec![vec!["सत्", "मार्गम्"]]),
            ("उन्नयनम्", vec![vec!["उत्", "नयनम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
