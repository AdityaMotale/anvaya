use crate::{
    rules::{
        rule::{BaseRule, RuleData, RuleGroup},
        Rule,
    },
    split::Splitter,
};
use orthography::{
    Adjuncts, Akshara, AsChar, AsStr, Consonant, IndependentVowel, SoundClass, Vowel,
};

pub(crate) struct SvarYan;

impl RuleGroup for SvarYan {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::yae_to_e_vowels());

        rls
    }
}

impl SvarYan {
    fn yae_to_e_vowels() -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-yae1",
                desc: "य् = इ + Vowel",
                tag: "6.1.77",
                left: Akshara(vec![SoundClass::Vowel(Vowel::I)]),
                right: Akshara(vec![SoundClass::AllVowel]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Ya),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
            Box::new(BaseRule(RuleData {
                name: "savarṇa-guṇa-yae2",
                desc: "य् = ई + Vowel",
                tag: "6.1.77",
                left: Akshara(vec![SoundClass::Vowel(Vowel::II)]),
                right: Akshara(vec![SoundClass::AllVowel]),
                merged: Akshara(vec![
                    SoundClass::Consonant(Consonant::Ya),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ]),
                special_sequence: None,
            })),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarGuna Rules (Test)");
    }

    #[test]
    fn yae_to_e_vowel_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("सुध्युपास्यः", vec![vec!["सुधी", "उपास्यः"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn yae_to_e_vowel_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("सुध्युपास्यः", vec![vec!["सुधी", "उपास्यः"]]),
            ("पर्यवेक्षकः", vec![vec!["परि", "अवेक्षकः"]]),
            ("अध्यापकः", vec![vec!["अधि", "आपकः"]]),
            ("अध्यायः", vec![vec!["अधि", "आयः"]]),
            ("अध्यादेशः", vec![vec!["अधि", "आदेशः"]]),
            ("अध्यात्मः", vec![vec!["अधि", "आत्मः"]]),
            ("अत्युत्तमः", vec![vec!["अति", "उत्तमः"]]),
            ("अप्येषः", vec![vec!["अपि", "एषः"]]),
            ("स्त्र्युत्सवः", vec![vec!["स्त्री", "उत्सवः"]]),
            ("प्रत्यक्षः", vec![vec!["प्रति", "अक्षः"]]),
            ("प्रत्ययः", vec![vec!["प्रति", "अयः"]]),
            ("प्रत्युत्पन्नः", vec![vec!["प्रति", "उत्पन्नः"]]),
            ("प्रत्युपकारः", vec![vec!["प्रति", "उपकारः"]]),
            ("दध्यानयः", vec![vec!["दधि", "आनयः"]]),
            ("अभ्युदयः", vec![vec!["अभि", "उदयः"]]),
            ("अभ्यागतः", vec![vec!["अभि", "आगतः"]]),
            ("अभ्यासः", vec![vec!["अभि", "आसः"]]),
            ("नद्यादयः", vec![vec!["नदी", "आदयः"]]),
            ("व्यस्तः", vec![vec!["वि", "अस्तः"]]),
            ("व्यंग्यः", vec![vec!["वि", "अंग्यः"]]),
            ("व्ययः", vec![vec!["वि", "अयः"]]),
            ("व्याप्तः", vec![vec!["वि", "आप्तः"]]),
            ("व्यासः", vec![vec!["वि", "आसः"]]),
            ("व्योमः", vec![vec!["वि", "ओमः"]]),
            ("व्यूहः", vec![vec!["वि", "ऊहः"]]),
            ("व्यष्टिः", vec![vec!["वि", "अष्टिः"]]),
            ("व्याधिः", vec![vec!["वि", "आधिः"]]),
            ("व्यभिचारः", vec![vec!["वि", "अभिचारः"]]),
            ("व्यवसायः", vec![vec!["वि", "अवसायः"]]),
            ("व्यायामः", vec![vec!["वि", "आयामः"]]),
            ("व्याकुलः", vec![vec!["वि", "आकुलः"]]),
            ("व्यक्तिः", vec![vec!["वि", "अक्तिः"]]),
            ("व्युत्पत्तिः", vec![vec!["वि", "उत्पत्तिः"]]),
            ("गौर्यात्मजः", vec![vec!["गौरी", "आत्मजः"]]),
            ("अत्युक्तिः", vec![vec!["अति", "उक्तिः"]]),
            ("इत्यत्र", vec![vec!["इति", "अत्र"]]),
            ("इत्यादि", vec![vec!["इति", "आदि"]]),
            ("यद्यपि", vec![vec!["यदि", "अपि"]]),
            ("व्यवस्था", vec![vec!["वि", "अवस्था"]]),
            ("महत्येष्णा", vec![vec!["महती", "एषणा"]]),
            ("करोत्ययम्", vec![vec!["करोति", "अयम्"]]),
            ("इत्यलम्", vec![vec!["इति", "अलम्"]]),
            ("पर्यटनम्", vec![vec!["परि", "अटनम्"]]),
            ("नार्यौदार्यम्", vec![vec!["नारी", "औदार्यम्"]]),
            ("नार्युचितम्", vec![vec!["नारी", "उचितम्"]]),
            ("देव्यर्पणम्", vec![vec!["देवी", "अर्पणम्"]]),
            ("प्रत्यर्पणम्", vec![vec!["प्रति", "अर्पणम्"]]),
            ("नद्यर्पणम्", vec![vec!["नदी", "अर्पणम्"]]),
            ("नद्युदकम्", vec![vec!["नदी", "उदकम्"]]),
            ("पर्यावरणम्", vec![vec!["परि", "आवरणम्"]]),
            ("वाण्यौचित्यम्", vec![vec!["वाणी", "औचित्यम्"]]),
            ("व्यग्रम्", vec![vec!["वि", "अग्रम्"]]),
            ("व्यवहारम्", vec![vec!["वि", "अवहारम्"]]),
            ("व्यंजनम्", vec![vec!["वि", "अंजनम्"]]),
            ("व्याख्यानम्", vec![vec!["वि", "आख्यानम्"]]),
            ("प्रत्येकम्", vec![vec!["प्रति", "एकम्"]]),
            ("प्रत्युत्तरम्", vec![vec!["प्रति", "उत्तरम्"]]),
            ("प्रत्यावर्तनम्", vec![vec!["प्रति", "आवर्तनम्"]]),
            ("प्रत्यारोपणम्", vec![vec!["प्रति", "आरोपणम्"]]),
            ("सख्यागमनम्", vec![vec!["सखी", "आगमनम्"]]),
            ("पर्यन्तम्", vec![vec!["परि", "अन्तम्"]]),
            ("अभ्युत्थानम्", vec![vec!["अभि", "उत्थानम्"]]),
            ("उपर्युक्तम्", vec![vec!["उपरि", "उक्तम्"]]),
            ("न्यूनम्", vec![vec!["नि", "ऊनम्"]]),
            ("वार्यस्ति", vec![vec!["वारि", "अस्ति"]]),
            ("गौर्यायाती", vec![vec!["गौरी", "आयाती"]]),
            ("अभ्यर्थी", vec![vec!["अभि", "अर्थी"]]),
            ("प्रत्याशी", vec![vec!["प्रति", "आशी"]]),
            ("स्त्र्युपयोगी", vec![vec!["स्त्री", "उपयोगी"]]),
            ("भात्यम्बरे", vec![vec!["भाति", "अम्बरे"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
