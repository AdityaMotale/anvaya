use crate::rules::{
    rule::{MultiOptRule, RuleData, RuleGroup},
    Adjuncts, Rule,
};
use orthography::{Akshara, Consonant, SoundClass, Vowel};

pub(crate) struct VynjanAnunasik;

impl RuleGroup for VynjanAnunasik {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::anunasik());

        rls
    }
}

impl VynjanAnunasik {
    fn anunasik() -> Vec<Box<dyn Rule>> {
        let merge_items: [Consonant; 6] = [
            Consonant::Nga,
            Consonant::Nga,
            Consonant::Dda,
            Consonant::Nna,
            Consonant::Na,
            Consonant::Ma,
        ];

        let swap_list: [Consonant; 6] = [
            Consonant::Ka,
            Consonant::Ga,
            Consonant::Ka,
            Consonant::Tta,
            Consonant::Ta,
            Consonant::Pa,
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

        let swap_list: Vec<Akshara> = swap_list
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
                name: "vyanjan-anunasik",
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
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanAnunasik Rules (Test)");
    }

    #[test]
    fn anunasik_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("दिङ्नाथः", vec![vec!["दिक्", "नाथः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn anunasik_test() {
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
