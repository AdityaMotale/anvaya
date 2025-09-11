use std::env;
use std::fs::{read_to_string, File};
use std::io::Write;

use orthography::sanitize;

const FREQ_TABLE: &'static str = "../raw_data/freq.txt";
const CANDI_TABLE: &'static str = "../raw_data/cache.txt";

fn main() {
    build_freq_table();
    build_cache_table();
}

fn build_freq_table() {
    let data = read_to_string(FREQ_TABLE).expect(&format!("failed to read {FREQ_TABLE}"));
    let mut builder = phf_codegen::Map::new();

    let mut total_count: usize = 0;
    let mut total_freq: usize = 0;

    for line in data.lines() {
        let mut parts = line.split(",");

        let key = parts.next().expect("Unable to read the key");
        let value = parts
            .next()
            .expect("Unable to read the value")
            .parse::<usize>()
            .expect("Unable to parse value from string");

        builder.entry(sanitize(key), value.to_string());

        total_count += 1;
        total_freq += value;
    }

    let out = format!(
        "{}/freq_map.rs",
        env::var("OUT_DIR").expect("Unable to get env::OUT_DIR")
    );
    let mut file = File::create(&out).unwrap_or_else(|e| panic!("Unable to create {}: {}", out, e));

    write!(
        file,
        "static FREQ_TABLE: phf::Map<&'static str, usize> = {};\n\
        static FREQ_TABLE_COUNT: usize = {};\n\
        static FREQ_TABLE_TOTAL_FREQ: usize = {};\n",
        builder.build(),
        total_count,
        total_freq
    )
    .unwrap_or_else(|e| panic!("Unable to write {}: {}", out, e));
}

fn build_cache_table() {
    let data = read_to_string(CANDI_TABLE).expect(&format!("failed to read {CANDI_TABLE}"));
    let mut builder = phf_codegen::Map::new();

    for (line_no, raw_line) in data.lines().enumerate() {
        let line = raw_line.trim();

        let mut parts = line.splitn(2, ',');

        let key = parts
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| panic!("line {}: missing key", line_no + 1));
        let value = parts
            .next()
            .map(str::trim)
            .unwrap_or_else(|| panic!("line {}: missing value part", line_no + 1));

        let val_list: Vec<String> = value
            .split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!("{:?}", s))
            .collect();

        if val_list.is_empty() {
            panic!(
                "line {}: no components found for key `{}`",
                line_no + 1,
                key
            );
        }

        builder.entry(key, format!("&[{}]", val_list.join(", ")));
    }

    let out_path = format!(
        "{}/cache_map.rs",
        env::var("OUT_DIR").expect("OUT_DIR not set")
    );
    let mut file =
        File::create(&out_path).unwrap_or_else(|e| panic!("Unable to create {}: {}", out_path, e));

    write!(
        file,
        "static CACHE_TABLE: phf::Map<&'static str, &'static [&'static str]> = {};\n",
        builder.build()
    )
    .unwrap_or_else(|e| panic!("Unable to write {}: {}", out_path, e));
}
