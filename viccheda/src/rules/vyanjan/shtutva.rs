use crate::rules::{
    rule::{BaseRule, MultiOptRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, Consonant, SoundClass, Vowel, DENTALS, PALATALS, RETROFLEX};

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
        vec![Box::new(BaseRule(RuleData {
            name: "vyanjan-shtutva-ss1",
            desc: "ष् = स् + अ ",
            tag: "8.4.44",
            left: Akshara(vec![
                SoundClass::Consonant(Consonant::Sa),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            merged: Akshara(vec![
                SoundClass::Consonant(Consonant::Ssa),
                SoundClass::Adjuncts(Adjuncts::VIRAMA),
            ]),
            special_sequence: None,
        }))]
    }

    fn retroflex_to_dentals() -> Vec<Box<dyn Rule>> {
        // नियम 2 – तवर्ग (त्, थ्, द्, ध्, न्) + टवर्ग (ट्, ठ्, ड्, ढ्, ण्) = चवर्ग टवर्ग (ट्, ठ्, ड्, ढ्, ण्)
        let merged_list: Vec<Akshara> = RETROFLEX
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
                name: "vyanjan-shtutva-retroflex1",
                desc: "retroflex = dentals + अ ",
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
        let _ = crate::init_logger("VyanjanShtutva Rules (Test)");
    }

    #[test]
    fn ss_to_s_ss_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
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
    fn retroflex_to_dentals_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("रामष्षष्ठः", vec![vec!["रामस्", "षष्ठः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn retroflex_to_dentals_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("तट्टीका", vec![vec!["तत्", "टीका"]]),
            ("बृहट्टीका", vec![vec!["बृहत्", "टीका"]]),
            ("सट्टीका", vec![vec!["सत्", "टीका"]]),
            ("उड्डीनः", vec![vec!["उद्", "डीनः"]]),
            ("उड्डयनम्", vec![vec!["उद्", "डयनम्"]]),
            ("सट्टिप्पणी", vec![vec!["सत्", "टिप्पणी"]]),
            ("बृहट्टंकशाला", vec![vec!["बृहत्", "टंकशाला"]]),
            ("चक्रिण्ढौकसे", vec![vec!["चक्रिन्", "ढौकसे"]]),
            ("कृष्णः", vec![vec!["कृस्", "णः"]]),
            ("महाण्डामरः", vec![vec!["महान्", "डामरः"]]),
            ("महड्ठालम्", vec![vec!["महद्", "ठालम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
