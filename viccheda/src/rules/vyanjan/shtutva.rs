use crate::rules::{rule::RuleGroup, Rule};

pub(crate) struct VynjanShtuvta;

impl RuleGroup for VynjanShtuvta {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ss_to_s_ss());
        rls.extend(Self::retroflex_to_dentals());

        rls
    }
}

impl VynjanShtuvta {
    fn ss_to_s_ss() -> Vec<Box<dyn Rule>> {
        // नियम 1 – स् + ष् = ष्
        vec![]
    }

    fn retroflex_to_dentals() -> Vec<Box<dyn Rule>> {
        // नियम 2 – तवर्ग (त्, थ्, द्, ध्, न्) + टवर्ग (ट्, ठ्, ड्, ढ्, ण्) = चवर्ग टवर्ग (ट्, ठ्, ड्, ढ्, ण्)
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanShtutva Rules (Test)");
    }

    #[test]
    #[ignore]
    fn ss_to_s_s_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn ss_to_s_ss_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]]),
            ("बालष्षष्ठः", vec![vec!["बालस्", "षष्ठः"]]),
            ("रामष्टीकते", vec![vec!["रामस्", "टीकते"]]),
            ("बालाष्टीकते", vec![vec!["बालास्", "टीकते"]]),
            ("धनुष्टंकारः", vec![vec!["धनुस्", "टंकारः"]]),
        ];

        test_sandhi_cases(cases, false);
    }

    #[test]
    #[ignore]
    fn retroflex_to_dentals_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    #[ignore]
    fn retroflex_to_dentals_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]]),
            ("बालष्षष्ठः", vec![vec!["बालस्", "षष्ठः"]]),
            ("रामष्टीकते", vec![vec!["रामस्", "टीकते"]]),
            ("बालाष्टीकते", vec![vec!["बालास्", "टीकते"]]),
            ("धनुष्टंकारः", vec![vec!["धनुस्", "टंकारः"]]),
            ("पेष्टा", vec![vec!["पेष्", "ता"]]),
            ("राष्ट्रम्", vec![vec!["राष्", "त्रम्"]]),
            ("इष्टः", vec![vec!["इष्", "तः"]]),
            ("दुष्टः", vec![vec!["दुष्", "तः"]]),
            ("तुष्टः", vec![vec!["तुष्", "तः"]]),
            ("आकृष्टः", vec![vec!["आकृष्", "तः"]]),
            ("तट्टीका", vec![vec!["तत्", "टीका"]]),
            ("बृहट्टीका", vec![vec!["बृहत्", "टीका"]]),
            ("सट्टीका", vec![vec!["सत्", "टीका"]]),
            ("उड्डीनः", vec![vec!["उत्", "डीनः"]]),
            ("उड्डयनम्", vec![vec!["उत्", "डयनम्"]]),
            ("सट्टिप्पणी", vec![vec!["सत्", "टिप्पणी"]]),
            ("बृहट्टंकशाला", vec![vec!["बृहत्", "टंकशाला"]]),
            ("चक्रिण्ढौकसे", vec![vec!["चक्रिन्", "ढौकसे"]]),
            ("कृष्णः", vec![vec!["कृष्", "नः"]]),
            ("महाण्डामरः", vec![vec!["महान्", "डामरः"]]),
            ("षण्णनवतिः", vec![vec!["षट्", "नवतिः"]]),
            ("षण्णाम्", vec![vec!["षड्", "नाम्"]]),
            ("महड्ठालम्", vec![vec!["महत्", "ठालम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
