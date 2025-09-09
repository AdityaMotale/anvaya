use std::env;
use std::fs::{read_to_string, File};
use std::io::Write;

const FREQ_TABLE: &'static str = "../raw_data/freq.txt";

fn main() {
    let data = read_to_string(FREQ_TABLE).expect(&format!("failed to read {FREQ_TABLE}"));
    let mut builder = phf_codegen::Map::new();

    for line in data.lines() {
        let mut parts = line.split(",");

        let key = parts.next().expect("Unable to read the key");
        let value = parts
            .next()
            .expect("Unable to read the value")
            .parse::<usize>()
            .expect("Unable to parse value from string");

        builder.entry(key, value.to_string());
    }

    let out = format!(
        "{}/freq_map.rs",
        env::var("OUT_DIR").expect("Unable to get env::OUT_DIR")
    );
    let mut file = File::create(&out).expect(&format!("Unable to create the {out} file"));

    write!(
        file,
        "static FREQ_MAP: phf::Map<&'static str, usize> = {};\n",
        builder.build()
    )
    .expect(&format!("Unable to write to {out}"));
}
