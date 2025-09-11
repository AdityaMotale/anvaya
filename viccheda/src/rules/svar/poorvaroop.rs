use crate::rules::{
    rule::{BaseRule, RuleData, RuleGroup},
    Rule,
};
use orthography::{Adjuncts, Akshara, IndependentVowel, SoundClass, Vowel};

pub(crate) struct SvarPoorvaroop;

impl RuleGroup for SvarPoorvaroop {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::avg_to_a_a());

        rls
    }
}

impl SvarPoorvaroop {
    fn avg_to_a_a() -> Vec<Box<dyn Rule>> {
        vec![Box::new(BaseRule(RuleData {
            name: "savarṇa-pūrvarūpa-avg1",
            desc: "ऽ = अ + अ ",
            tag: "6.1.109",
            left: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
            right: Akshara(vec![SoundClass::IndependentVowel(IndependentVowel::A)]),
            merged: Akshara(vec![SoundClass::Adjuncts(Adjuncts::AVAGRAHA)]),
            special_sequence: None,
        }))]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("SvarPoorvaroop Rules (Test)");
    }

    #[test]
    fn avg_to_a_a_test_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("अन्तेऽपि", vec![vec!["अन्ते", "अपि"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn avg_to_a_a_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            // e
            ("अन्तेऽपि", vec![vec!["अन्ते", "अपि"]]),
            ("केऽपि", vec![vec!["के", "अपि"]]),
            ("कोऽपि", vec![vec!["को", "अपि"]]),
            ("तेऽपि", vec![vec!["ते", "अपि"]]),
            ("येऽपि", vec![vec!["ये", "अपि"]]),
            ("वनेऽपि", vec![vec!["वने", "अपि"]]),
            ("ग्रामेऽपि", vec![vec!["ग्रामे", "अपि"]]),
            ("सर्वेऽपि", vec![vec!["सर्वे", "अपि"]]),
            ("वृक्षेऽपि", vec![vec!["वृक्षे", "अपि"]]),
            ("भवनेऽस्मि", vec![vec!["भवने", "अस्मि"]]),
            ("तेऽत्र", vec![vec!["ते", "अत्र"]]),
            ("वनेऽत्र", vec![vec!["वने", "अत्र"]]),
            ("हरेऽव", vec![vec!["हरे", "अव"]]),
            ("अरण्येऽस्मिन्", vec![vec!["अरण्ये", "अस्मिन्"]]),
            ("जलेऽस्ति", vec![vec!["जले", "अस्ति"]]),
            ("कोऽस्ति", vec![vec!["को", "अस्ति"]]),
            ("कृतेऽयम्", vec![vec!["कृते", "अयम्"]]),
            ("अरण्येऽगच्छत्", vec![vec!["अरण्ये", "अगच्छत्"]]),
            ("मेऽन्तिके", vec![vec!["मे", "अन्तिके"]]),
            ("दीर्घऽहनि", vec![vec!["दीर्घ", "अहनि"]]),
            // au
            ("विष्णोऽत्र", vec![vec!["विष्णो", "अत्र"]]),
            ("विष्णोऽव", vec![vec!["विष्णो", "अव"]]),
            ("एषोऽस्मि", vec![vec!["एषो", "अस्मि"]]),
            ("सोऽवदत्", vec![vec!["सो", "अवदत्"]]),
            ("रामोऽहसत्", vec![vec!["रामो", "अहसत्"]]),
            ("कोऽपि", vec![vec!["को", "अपि"]]),
            ("प्रभोऽत्र", vec![vec!["प्रभो", "अत्र"]]),
            ("गोपोलोऽहम्", vec![vec!["गोपोलो", "अहम्"]]),
            ("लोकोऽयम्", vec![vec!["लोको", "अयम्"]]),
            ("निर्धनोऽयम्", vec![vec!["निर्धनो", "अयम्"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
