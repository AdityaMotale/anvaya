pub mod rule;
mod svar;

use crate::rules::rule::{Rule, RuleGroup};
use logger::{debugf, errorf, tracef, Logger};
use orthography::{Akshara, AsChar, AsStr};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) fn get_all_rules() -> Vec<Box<dyn Rule>> {
    let mut all_rules = Vec::new();

    // svar rules
    all_rules.extend(svar::dirgha::SvarDirgha::rules());
    all_rules.extend(svar::guna::SvarGuna::rules());
    all_rules.extend(svar::vriddhi::SvarVriddhi::rules());
    all_rules.extend(svar::yan::SvarYan::rules());

    all_rules
}

// trim input sound from right with the [Akshara], returns `None`
// if [Akshara] does not matches w/ input sound
pub(crate) fn trim_sound_with_akshara(
    sound: &str,
    akshara: &Akshara,
    logger: &Logger,
) -> Option<String> {
    // sanity check
    assert!(akshara.0.len() != 0);

    // fast fail for invalid input
    if sound.is_empty() {
        return None;
    }

    tracef!(
        logger,
        "----\n[Sound Trim] matching sound=`{}` with akshara {}",
        sound,
        akshara
    );

    // in the loop we consume suffixes/last-chars (bytes) from right -> left,
    // this way we get our trimmed sound
    let mut tail = sound.to_string();

    // we iter w/ reverse sequence cause we are matching from right -> left,
    // or end -> start
    for soundclass in akshara.0.iter().rev() {
        if tail.is_empty() {
            tracef!(
                logger,
                "[Sound Trim] (ERROR) tail empty but still have {} to match",
                soundclass.as_char()
            );

            return None;
        }

        if let Some(expected_str) = soundclass.as_str() {
            if tail.ends_with(expected_str) {
                let new_len = tail.len() - expected_str.len();
                tail.truncate(new_len);

                tracef!(
                    logger,
                    "[Sound Trim] (MATCH) `{}` for {} ;; new tail=`{}`",
                    expected_str,
                    soundclass.as_char(),
                    tail
                );

                continue;
            } else {
                tracef!(
                    logger,
                    "[Sound Trim] (SKIP) tail `{}` does not end with `{}` for {}",
                    tail,
                    expected_str,
                    soundclass.as_char(),
                );

                return None;
            }
        } else {
            // NOTE: Except [Vowel::A], all sound class have string representation (as_str)
            // And [Vowel::A] must never be used in merged sequence, cause we never split on
            // single [Vowel::A] or [IndependentVowel::A]. So, we're mostly safe here ;)

            errorf!(
                logger,
                "[Sound Trim] {akshara} contains empty string sequence {}",
                soundclass.as_char(),
            );

            return None;
        }
    }

    Some(tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_logger;
    use orthography::{Adjuncts, Vowel};
    use orthography::{AsStr, SoundClass};

    #[test]
    fn test_trim_sound_with_akshara() {
        let log_cell = init_logger("RuleUtils (Test)");
        let logger = log_cell.get().expect("Custom Logger for test");

        let candidates: Vec<(String, &'static str, Akshara, &'static str)> = vec![
            (
                String::from("पित"),
                "पितृृ",
                Akshara(vec![
                    SoundClass::Vowel(Vowel::R),
                    SoundClass::Vowel(Vowel::R),
                ]),
                "[Vowel::R, Vowel::R] matching with Vowel::RR",
            ),
            (
                String::from("सर्व"),
                "सर्वां",
                Akshara(vec![
                    SoundClass::Vowel(Vowel::AA),
                    SoundClass::Adjuncts(Adjuncts::ANUSVARA),
                ]),
                "[Vowel::AA, Adjuncts::ANUSVARA] matching with [Vowel::AA, Adjuncts::ANUSVARA]",
            ),
        ];

        for (out, inp, aksh, desc) in candidates {
            let sound = trim_sound_with_akshara(inp, &aksh, logger);

            assert_eq!(Some(out), sound, "Failed for {}", desc);
        }
    }
}
