use orthography::sanitize;
use std::time::Instant;
use viccheda::Viccheda;

const CANDI_FILE: &'static str = "../raw_data/candi.txt";

fn read_candidates(txt_path: &str) -> Vec<(String, Vec<String>)> {
    let data =
        std::fs::read_to_string(txt_path).unwrap_or_else(|e| panic!("ERROR {}: {}", txt_path, e));

    let mut candis = Vec::new();

    for (line_no, line) in data.lines().enumerate() {
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
            .map(|s| sanitize(s))
            .collect();

        if val_list.is_empty() {
            panic!(
                "line {}: no components found for key `{}`",
                line_no + 1,
                key
            );
        }

        candis.push((sanitize(key), val_list));
    }

    candis
}

#[derive(Debug)]
struct CandidateResult {
    idx: usize,
    word: String,
    expected: String,
    actual: Option<String>, // None if skipped / no split
    correct: bool,
    time_ms: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn stddev(v: &[f64], mean: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }

    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() as f64);
    var.sqrt()
}

fn main() {
    println!("Bench init...");

    let viccheda = Viccheda::new(false);
    let candi_list = read_candidates(CANDI_FILE);

    let total = candi_list.len();
    let mut results: Vec<CandidateResult> = Vec::with_capacity(total);

    println!("Starting bench on {} candidates...", total);

    for (i, (word, expected_splits)) in candi_list.into_iter().enumerate() {
        if i % 100 == 0 {
            println!(" -> Processing candidate {}/{}", i, total);
        }

        let expected_join = expected_splits.join("|");
        let start = Instant::now();
        let sp = viccheda.split(&word);
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0;

        match sp {
            None => {
                results.push(CandidateResult {
                    idx: i,
                    word,
                    expected: expected_join,
                    actual: None,
                    correct: false,
                    time_ms,
                });
            }
            Some(res) => {
                let actual_vec: Vec<String> = res.splits.iter().map(|s| sanitize(s)).collect();
                let actual_join = actual_vec.join("|");
                let correct = actual_join == expected_join;
                results.push(CandidateResult {
                    idx: i,
                    word,
                    expected: expected_join,
                    actual: Some(actual_join),
                    correct,
                    time_ms,
                });
            }
        }
    }

    println!("Finished processing all candidates! Computing stats...");

    // stats for summery
    let attempted_results: Vec<&CandidateResult> =
        results.iter().filter(|r| r.actual.is_some()).collect();

    let skipped_count = results.iter().filter(|r| r.actual.is_none()).count();
    let attempted = attempted_results.len();
    let correct = results.iter().filter(|r| r.correct).count();

    let times: Vec<f64> = attempted_results.iter().map(|r| r.time_ms).collect();
    let total_time_ms: f64 = times.iter().sum();

    let avg_time = if times.is_empty() {
        0.0
    } else {
        total_time_ms / (times.len() as f64)
    };

    let med_time = median(times.clone());
    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let sd_time = stddev(&times, avg_time);

    println!();
    println!("Benching results for file: {}", CANDI_FILE);
    println!("----------------------------------------");
    println!("Total candidates read : {}", total);
    println!("Attempted (had output) : {}", attempted);
    println!("Skipped (no output)    : {}", skipped_count);
    println!();
    println!(
        "Correct                : {} / {} ({} tried)",
        correct, total, attempted
    );

    let pct_of_attempted = if attempted == 0 {
        0.0
    } else {
        (correct as f64) / (attempted as f64) * 100.0
    };

    println!("Accuracy (of attempted): {:.2} %", pct_of_attempted);
    println!();
    println!("Timing (only attempted):");
    println!("  Total time       = {:.3} ms", total_time_ms);
    println!("  Avg per candidate= {:.3} ms", avg_time);
    println!("  Median           = {:.3} ms", med_time);
    println!(
        "  Min              = {:.3} ms",
        if min_time.is_finite() { min_time } else { 0.0 }
    );
    println!(
        "  Max              = {:.3} ms",
        if max_time.is_finite() { max_time } else { 0.0 }
    );
    println!("  Stddev           = {:.3} ms", sd_time);
    println!("----------------------------------------");

    // Top 10 slowest
    let mut by_time: Vec<&CandidateResult> = results.iter().collect();
    by_time.sort_by(|a, b| b.time_ms.partial_cmp(&a.time_ms).unwrap());

    println!("\nTop 10 slowest (by time_ms):");

    for r in by_time.iter().take(10) {
        println!(
            "#{} | {:.3} ms | {} | expected: {} | actual: {} | {}",
            r.idx,
            r.time_ms,
            r.word,
            r.expected,
            r.actual.clone().unwrap_or_else(|| "<SKIPPED>".into()),
            if r.correct { "OK" } else { "WRONG" }
        );
    }

    // 10 mismatches
    let mismatches: Vec<&CandidateResult> = results.iter().filter(|r| !r.correct).collect();

    println!("\nMismatches: {} (showing up to 10)", mismatches.len());

    for r in mismatches.iter().take(10) {
        println!(
            "#{} | {:.3} ms | {} | expected: {} | actual: {}",
            r.idx,
            r.time_ms,
            r.word,
            r.expected,
            r.actual.clone().unwrap_or_else(|| "<SKIPPED>".into())
        );
    }

    println!();
}
