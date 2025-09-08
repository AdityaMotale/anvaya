use orthography::{Adjuncts, Akshara, Consonant, IndependentVowel, SoundClass, Vowel};

use crate::rules::{
    rule::{MultiOptRule, RuleData, RuleGroup},
    Rule,
};

pub(crate) struct VynjanJashtva;

impl RuleGroup for VynjanJashtva {
    fn rules() -> Vec<Box<dyn Rule>> {
        let mut rls = Vec::new();

        rls.extend(Self::ga_to_zal());
        rls.extend(Self::ja_to_zal());
        rls.extend(Self::dda_to_zal());
        rls.extend(Self::da_to_zal());
        rls.extend(Self::ba_to_zal());

        rls
    }
}

impl VynjanJashtva {
    fn ga_to_zal() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..4)
            .map(|_| Akshara(vec![SoundClass::Consonant(Consonant::Ga)]))
            .collect();

        let swap_consonants: [Consonant; 4] =
            [Consonant::Ka, Consonant::Kha, Consonant::Ga, Consonant::Gha];

        let swap_list: Vec<Akshara> = swap_consonants
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-jashtva-ga",
                desc: "ग् = क्, ख्, ग्, घ् + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (
                        Akshara(vec![
                            SoundClass::IndependentVowel(IndependentVowel::A),
                            SoundClass::Adjuncts(Adjuncts::ANUSVARA),
                        ]),
                        true,
                    ),
                ]),
            },
        })]
    }

    fn ja_to_zal() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..4)
            .map(|_| Akshara(vec![SoundClass::Consonant(Consonant::Ja)]))
            .collect();

        let swap_consonants: [Consonant; 4] = [
            Consonant::Cha,
            Consonant::Chha,
            Consonant::Ja,
            Consonant::Jha,
        ];

        let swap_list: Vec<Akshara> = swap_consonants
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-jashtva-ja",
                desc: "ज् = च्, छ्, ज्, झ् + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]),
                    false,
                )]),
            },
        })]
    }

    fn dda_to_zal() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..5)
            .map(|_| Akshara(vec![SoundClass::Consonant(Consonant::Dda)]))
            .collect();

        let swap_consonants: [Consonant; 5] = [
            Consonant::Tta,
            Consonant::Ttha,
            Consonant::Dda,
            Consonant::Ddha,
            Consonant::Ssa,
        ];

        let swap_list: Vec<Akshara> = swap_consonants
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-jashtva-dda",
                desc: "ड् = ट्, ठ्, ड्, ढ्, (ष्) + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![(
                    Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]),
                    false,
                )]),
            },
        })]
    }

    fn da_to_zal() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..4)
            .map(|_| Akshara(vec![SoundClass::Consonant(Consonant::Da)]))
            .collect();

        let swap_consonants: [Consonant; 4] =
            [Consonant::Ta, Consonant::Tha, Consonant::Da, Consonant::Dha];

        let swap_list: Vec<Akshara> = swap_consonants
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-jashtva-da",
                desc: "द् = त्, थ्, द्, ध् + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (
                        Akshara(vec![
                            SoundClass::IndependentVowel(IndependentVowel::A),
                            SoundClass::Adjuncts(Adjuncts::ANUSVARA),
                        ]),
                        true,
                    ),
                ]),
            },
        })]
    }

    fn ba_to_zal() -> Vec<Box<dyn Rule>> {
        let merged_list: Vec<Akshara> = (0..4)
            .map(|_| Akshara(vec![SoundClass::Consonant(Consonant::Ba)]))
            .collect();

        let swap_consonants: [Consonant; 4] =
            [Consonant::Pa, Consonant::Pha, Consonant::Ba, Consonant::Bha];

        let swap_list: Vec<Akshara> = swap_consonants
            .iter()
            .map(|consonant| {
                Akshara(vec![
                    SoundClass::Consonant(*consonant),
                    SoundClass::Adjuncts(Adjuncts::VIRAMA),
                ])
            })
            .collect();

        vec![Box::new(MultiOptRule {
            merged_list,
            swap_list,
            data: RuleData {
                name: "vyanjan-jashtva-ba",
                desc: "ब् = प्, फ्, ब्, भ् + अ ",
                tag: "8.4.44",
                left: Akshara(vec![]),
                right: Akshara(vec![SoundClass::Vowel(Vowel::A)]),
                merged: Akshara(vec![]),
                special_sequence: Some(vec![
                    (Akshara(vec![SoundClass::Adjuncts(Adjuncts::VIRAMA)]), false),
                    (
                        Akshara(vec![
                            SoundClass::IndependentVowel(IndependentVowel::A),
                            SoundClass::Adjuncts(Adjuncts::ANUSVARA),
                        ]),
                        true,
                    ),
                ]),
            },
        })]
    }
}

#[cfg(test)]
mod tests {
    use crate::split::test_sandhi_cases;

    fn create_logger() {
        let _ = crate::init_logger("VyanjanJashtva Rules (Test)");
    }

    #[test]
    fn jash_to_zal_debug() {
        create_logger();
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![("वागीश्वरी", vec![vec!["वाक्", "ईश्वरी"]])];
        test_sandhi_cases(cases, true);
    }

    #[test]
    fn jash_to_zal_test() {
        let cases: Vec<(&str, Vec<Vec<&str>>)> = vec![
            ("वागीश्वरी", vec![vec!["वाक्", "ईश्वरी"]]),
            ("वागीशः", vec![vec!["वाक्", "ईशः"]]),
            ("वाग्जालः", vec![vec!["वाक्", "जालः"]]),
            ("वाग्वज्रः", vec![vec!["वाक्", "वज्रः"]]),
            ("वाग्यंत्रः", vec![vec!["वाक्", "यंत्रः"]]),
            ("वाग्विदग्धता", vec![vec!["वाक्", "विदग्धता"]]),
            ("वाग्दत्ता", vec![vec!["वाक्", "दत्ता"]]),
            ("वाग्दानम्", vec![vec!["वाक्", "दानम्"]]),
            ("दिग्गजः", vec![vec!["दिक्", "गजः"]]),
            ("दिगंतः", vec![vec!["दिक्", "अंतः"]]),
            ("दिग्गयंदः", vec![vec!["दिक्", "गयंदः"]]),
            ("दिग्भ्रमः", vec![vec!["दिक्", "भ्रमः"]]),
            ("दिग्विजयः", vec![vec!["दिक्", "विजयः"]]),
            ("दिग्वधूः", vec![vec!["दिक्", "वधूः"]]),
            ("दिग्हस्ती", vec![vec!["दिक्", "हस्ती"]]),
            ("दिग्दर्शनम्", vec![vec!["दिक्", "दर्शनम्"]]),
            ("प्रागैतिहासिकः", vec![vec!["प्राक्", "ऐतिहासिकः"]]),
            ("सम्यग्ज्ञानम्", vec![vec!["सम्यक्", "ज्ञानम्"]]),
            ("ऋग्वेदः", vec![vec!["ऋक्", "वेदः"]]),
            ("अजंतः", vec![vec!["अच्", "अंतः"]]),
            ("अजादिः", vec![vec!["अच्", "आदिः"]]),
            ("अज्झीनः", vec![vec!["अच्", "झीनः"]]),
            ("षडंगः", vec![vec!["षट्", "अंगः"]]),
            ("षड्गुणः", vec![vec!["षट्", "गुणः"]]),
            ("षड्रसः", vec![vec!["षट्", "रसः"]]),
            ("षड्रागः", vec![vec!["षट्", "रागः"]]),
            ("षड्विकारः", vec![vec!["षट्", "विकारः"]]),
            ("षड्यंत्रः", vec![vec!["षट्", "यंत्रः"]]),
            ("षड्रिपुः", vec![vec!["षट्", "रिपुः"]]),
            ("षड्दर्शनम्", vec![vec!["षट्", "दर्शनम्"]]),
            ("षडेवम्", vec![vec!["षट्", "एवम्"]]),
            ("षड्भुजा", vec![vec!["षट्", "भुजा"]]),
            ("सदाशयः", vec![vec!["सत्", "आशयः"]]),
            ("सद्धर्मः", vec![vec!["सत्", "धर्मः"]]),
            ("सद्गतिः", vec![vec!["सत्", "गतिः"]]),
            ("सदुपयोगः", vec![vec!["सत्", "उपयोगः"]]),
            ("सद्वाणी", vec![vec!["सत्", "वाणी"]]),
            ("सद्भावना", vec![vec!["सत्", "भावना"]]),
            ("चिदानन्दः", vec![vec!["चित्", "आनन्दः"]]),
            ("चिद्रूपम्", vec![vec!["चित्", "रूपम्"]]),
            ("तद्भवः", vec![vec!["तत्", "भवः"]]),
            ("तद्देवम्", vec![vec!["तत्", "देवम्"]]),
            ("तद्नुसारम्", vec![vec!["तत्", "नुसारम्"]]),
            ("उद्यानः", vec![vec!["उत्", "यानः"]]),
            ("उद्धारः", vec![vec!["उत्", "धारः"]]),
            ("उद्देश्यम्", vec![vec!["उत्", "देश्यम्"]]),
            ("उद्घाटनम्", vec![vec!["उत्", "घाटनम्"]]),
            ("मृदंगः", vec![vec!["मृत्", "अंगः"]]),
            ("जगदीशः", vec![vec!["जगत्", "ईशः"]]),
            ("जगद्बन्धुः", vec![vec!["जगत्", "बन्धुः"]]),
            ("जगद्गुरुः", vec![vec!["जगत्", "गुरुः"]]),
            ("जगदाधारः", vec![vec!["जगत्", "आधारः"]]),
            ("जगदानंदः", vec![vec!["जगत्", "आनंदः"]]),
            ("भगवद्गीता", vec![vec!["भगवत्", "गीता"]]),
            ("वृहदाकारः", vec![vec!["वृहत्", "आकारः"]]),
            ("समृद्धिः", vec![vec!["समृध्", "धिः"]]),
            ("कुद्धः", vec![vec!["कुध्", "धः"]]),
            ("अब्जः", vec![vec!["अप्", "जः"]]),
            ("अब्दः", vec![vec!["अप्", "दः"]]),
            ("लब्धा", vec![vec!["लभ्", "धा"]]),
            ("उपलब्धिः", vec![vec!["उपलभ्", "धिः"]]),
        ];

        test_sandhi_cases(cases, false);
    }
}
