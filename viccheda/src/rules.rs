use std::{collections::HashMap, sync::LazyLock};

use crate::common::{Consonant, Vowel};

#[derive(Debug)]
enum SoundClass {
    Vowel(Vowel),
    Consonent(Consonant),
}

#[derive(Debug)]
struct Rule {
    name: &'static str,
    desc: &'static str,
    tag: &'static str,
    left: SoundClass,
    right: SoundClass,
    outputs: Vec<(SoundClass, SoundClass)>,
}

struct Sandhi {
    rules: Vec<Rule>,
}

impl Sandhi {
    pub fn new() -> Self {
        Self {
            rules: Self::get_rules(),
        }
    }

    pub fn split(&self, morpheme: &str) -> Vec<(&'static str, &'static str)> {
        Vec::new()
    }

    fn get_rules() -> Vec<Rule> {
        vec![Rule {
            name: "savarṇa-dīrgha-a",
            desc: "आ  => अ + अ ",
            tag: "6.1.101",
            left: SoundClass::Vowel(Vowel::A),
            right: SoundClass::Vowel(Vowel::A),
            outputs: vec![(SoundClass::Vowel(Vowel::A), SoundClass::Vowel(Vowel::A))],
        }]
    }
}
