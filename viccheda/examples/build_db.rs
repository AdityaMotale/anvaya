use std::{fs::File, io::BufRead, path::Path};
use turbocache::{TurboCache, TurboResult};

fn main() -> TurboResult<()> {
    let word_path = Path::new("../raw_data/words_freq.csv");
    let cache_path = Path::new("./cache");

    clear_cache(&cache_path);

    let cache = TurboCache::new("./cache", 16_000)?;
    let file = File::open(&word_path).expect("Open words file");
    let reader = std::io::BufReader::new(file);

    let mut line_count = 0;
    let mut duplicate_count = 0;

    for line in reader.lines() {
        let line = line.expect("Read line from db");
        let mut parts = line.splitn(2, ',');

        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
            let k = k.trim();
            let v = v.trim();

            if let Some(_) = cache.get(k.as_bytes())? {
                duplicate_count += 1;
            } else {
                cache.set(k.as_bytes(), v.as_bytes())?;
            }
        } else {
            eprintln!("Malformed line: {}", line);
        }

        line_count += 1;
    }

    println!("DB Size: {}", cache.total_count()?);
    println!("Total Lines: {}", line_count - duplicate_count);
    println!("Duplicate Lines: {}", duplicate_count);

    Ok(())
}

fn clear_cache(path: &Path) {
    match std::fs::remove_dir_all(path) {
        Ok(_) => {
            println!("[INFO] Cleared Old cache");
        }
        Err(error) => {
            eprintln!("Unable to clear old cache, [ERROR]: {}", error);
        }
    }
}
