use crate::{
    rules::{BaseRule, Rule, RuleData, RuleGroup, RuleUtils},
    split::{Candidate, Splitter},
};
use orthography::{
    Adjuncts, Akshara, AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel,
};

pub(crate) struct SvarVriddhi;

impl RuleGroup for SvarVriddhi {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ai_to_a_e());

        rls
    }
}

//  अ
//  आ
//  ए
//  ऐ
impl SvarVriddhi {
    fn ai_to_a_e() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai1",
                desc: "ऐ = अ + ए",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::E)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai2",
                desc: "ऐ = अ + ऐ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AI)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai3",
                desc: "ऐ = आ  + ए",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::E)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-vṛddhi-ai4",
                desc: "ऐ = आ  + ऐ",
                tag: "6.1.88",
                left: Akshara(vec![SoundClass::Vowel(Vowel::AA)]),
                right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::AI)]),
                merged: Akshara(vec![SoundClass::Vowel(Vowel::AI)]),
                special_sequence: None,
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarVriddhi Rules (Test)");
    }

    #[test]
    fn ai_to_a_e_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("जनैकता", vec![vec!["जन", "एकता"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn ai_to_a_e_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("जनैकता", vec![vec!["जन", "एकता"]]),
            ("एकैकः", vec![vec!["एक", "एकः"]]),
            ("अत्रैकमत्यम्", vec![vec!["अत्र", "एकमत्यम्"]]),
            ("राजैषः", vec![vec!["राज", "एषः"]]),
            ("बालैषा", vec![vec!["बाला", "एषा"]]),
            ("तथैव", vec![vec!["तथा", "एव"]]),
            ("वसुधैव", vec![vec!["वसुधा", "एव"]]),
            ("गंगैषा", vec![vec!["गंगा", "एषा"]]),
            ("पुत्रैषणा", vec![vec!["पुत्र", "ऐषणा"]]),
            ("सदैव", vec![vec!["सदा", "एव"]]),
            ("देवैश्वर्यम्", vec![vec!["देव", "ऐश्वर्यम्"]]),
            ("नृपैश्वर्यम्", vec![vec!["नृप", "ऐश्वर्यम्"]]),
            ("महैश्वर्यम्", vec![vec!["महा", "ऐश्वर्यम्"]]),
            ("ममैश्वर्यम्", vec![vec!["मम", "ऐश्वर्यम्"]]),
            ("गङ्गैश्वर्यम्", vec![vec!["गङ्गा", "ऐश्वर्यम्"]]),
            ("मतैक्यम्", vec![vec!["मत", "ऐक्यम्"]]),
            ("सर्वैक्यम्", vec![vec!["सर्व", "ऐक्यम्"]]),
            ("कृष्णैकत्वम्", vec![vec!["कृष्ण", "एकत्वम्"]]),
            ("देवैकत्वम्", vec![vec!["देव", "एकत्वम्"]]),
            ("हितैषी", vec![vec!["हित", "एषी"]]),
            ("वित्तैषणा", vec![vec!["वित्त", "एषणा"]]),
            ("लोकैषणा", vec![vec!["लोक", "एषणा"]]),
            ("विद्यैषणा", vec![vec!["विद्या", "एषणा"]]),
            ("सभैका", vec![vec!["सभा", "एका"]]),
            ("वीरैकः", vec![vec!["वीर", "एकः"]]),
            ("राजैष:", vec![vec!["राजा", "एष:"]]),
            ("मैवम्", vec![vec!["मा", "एवम्"]]),
            ("तवैवम्", vec![vec!["तव", "एवम्"]]),
            ("अत्रैकम्", vec![vec!["अत्र", "एकम्"]]),
            ("जनैकता", vec![vec!["जन", "एकता"]]),
            ("स्वैच्छिकम्", vec![vec!["स्व", "ऐच्छिकम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
