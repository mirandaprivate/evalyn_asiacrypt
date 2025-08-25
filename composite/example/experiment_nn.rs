use std::fs::{self, File};
use std::io::{self, Write};

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use composite::protocols::nn::ProtocolNN;

use mat::MyShortInt;

const DEPTH: usize = 16;
const SHAPE: (usize, usize) = (1024, 1024);

fn main() -> io::Result<()> {
    init_log(true)?; 

    run_with_monitor(
        |iter| experiment_nn(iter),
        Duration::from_secs(60),      // total duration
        Duration::from_secs(60),       // report interval (suppress intermediate output)
        Duration::from_millis(200),   // sampling interval
    )?;
    Ok(())
}

// Contains only the core experiment logic; called once per iteration
fn experiment_nn(_iter: u64) {
    type E = ark_bls12_381::Bls12_381;
    let depth = DEPTH;
    let shape = SHAPE;

    println!("====Running experiment for depth: {}, shape: {:?}, bitwidth: {:?}=====", depth, shape, 8 * std::mem::size_of::<MyShortInt>());

    let mut nn: ProtocolNN<E> = ProtocolNN::new(depth, shape);

   
    nn.commit_to_pars();
    nn.commit_to_witness();
    nn.reduce_prover_and_building_pop_circuit();
    nn.commit_to_pop_circuits();



    nn.gen_pop_proof();
    nn.open_leaf_commitment();
    nn.prove_fs();


    let timer_verify = Instant::now();
    // nn.verify_leaf_commitment();
    nn.verify();
    let duration_verify = timer_verify.elapsed().as_secs_f64();

    println!("*************************************************************************");
    println!("========NN Experiment Results======================================");
    println!("🕒 \x1b[1m Verifying took {:.6} seconds \x1b[0m", duration_verify);
    println!("⬜ \x1b[1m Proof size: {} B \x1b[0m", nn.get_compressed_proof_size());
    println!("⬜ \x1b[1m NN Commitment size: {:?} B \x1b[0m", nn.get_nn_commitment_size());
    println!("========End NN Experiment Results==================================");
    println!("*************************************************************************");

}

// Generic wrapper: handles memory monitoring, time loop and reporting
fn run_with_monitor<F>(mut body: F, max_duration: Duration, report_interval: Duration, sample_interval: Duration) -> io::Result<()>
where
    F: FnMut(u64),
{
    let max_memory = Arc::new(Mutex::new(0u64));
    let max_memory_clone = Arc::clone(&max_memory);
    thread::spawn(move || {
        loop {
            if let Some(val) = get_memory_usage() {
                if let Ok(mut max) = max_memory_clone.lock() {
                    if val > *max { *max = val; }
                }
            }
            thread::sleep(sample_interval);
        }
    });

    let start = Instant::now();
    // Print thread information once
    match std::env::var("RAYON_NUM_THREADS") {
        Ok(v) => { let _ = log_line(&format!("RAYON_NUM_THREADS: {}", v)); },
        Err(_) => {
            let inferred = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            let _ = log_line(&format!("RAYON_NUM_THREADS (unset, inferred): {}", inferred));
        }
    }
    let mut last_report = Instant::now(); // Not used, kept for structure
    let mut iter: u64 = 0;
    // Silent mode
    while start.elapsed() < max_duration {
        body(iter);
        if last_report.elapsed() >= report_interval { last_report = Instant::now(); }
        iter += 1;
    }
    let peak_kb = *max_memory.lock().unwrap();
    log_line(&format!("⬜ \x1b[1m Peak RAM Usage: {}KB \x1b[0m", peak_kb))?;
    Ok(())
}

// =============== Logging helpers ===============
use std::sync::OnceLock;
static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

fn init_log(append: bool) -> io::Result<()> {
    // Always write to the `example` directory under the composite crate: <crate_root>/example/experiment.log
    // Use compile-time env var to ensure independence from the working directory
    let path = format!("{}/example/experiment.log", env!("CARGO_MANIFEST_DIR"));
    let file = if append { File::options().create(true).append(true).open(&path)? } else { File::create(&path)? };
    LOG_FILE.get_or_init(|| Mutex::new(file));
    log_line(&format!("[log] writing to {}", path)).ok();
    Ok(())
}

fn log_line(line: &str) -> io::Result<()> {
    if let Some(m) = LOG_FILE.get() {
        if let Ok(mut f) = m.lock() {
            writeln!(f, "{}", line)?;
            f.flush()?;
        }
    }
    println!("{}", line); // Still print to stdout
    Ok(())
}

fn get_memory_usage() -> Option<u64> {
    // Print '⌛' to stdout every 30 seconds as a heartbeat (not written to the log file)
    use std::sync::OnceLock as _OnceLock; // Use OnceLock to avoid naming conflict with LOG_FILE
    static LAST_EQ: _OnceLock<Mutex<Instant>> = _OnceLock::new();
    if let Ok(mut t) = LAST_EQ.get_or_init(|| Mutex::new(Instant::now())).lock() {
        if t.elapsed() >= Duration::from_secs(30) {
            print!("⌛");
            let _ = io::stdout().flush();
            *t = Instant::now();
        }
    }
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse().ok());
        }
    }
    None
}
