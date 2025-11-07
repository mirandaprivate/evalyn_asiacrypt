# Evalyn: Scalable zkSNARKs for Matrix Computations

This workspace contains the implementation and experiments for the paper:

**Scalable zkSNARKs for Matrix Computations: A Generic Framework for Verifiable Deep Learning**

It provides building blocks for matrix-oriented zkSNARK protocols and a composite example for verifiable neural networks.

## Quick Start

```bash
# 1. Clone and build the project
cargo build --workspace --release

# 2. Run the neural network experiment (fast demo with depth=16)
RAYON_NUM_THREADS=8 cargo run --release -p composite --example experiment_nn

# 3. Run tests to verify installation
cargo test --workspace --release
```

> Note for Windows users: It is recommended to run the above commands inside WSL (Windows Subsystem for Linux).

### Configuration Options
- **Fast Demo (depth=16)**: ~5 minutes runtime (with 8 threads), ~4GB peak RAM usage
- **Paper Results (depth=1024)**: ~35-37 minutes runtime (with 64 threads), ~103GB peak RAM usage (requires code modification in composite/example/experiment.nn)

## Status and reproducibility

This repository is a frozen snapshot intended for reproducing the paper's experiments. It will not receive further updates. We are continuing active development privately, and a commercial version will be released later. For reproduction, please use this snapshot as-is; issues and PRs may not be triaged.

## Source Code Organization

The codebase is organized as a Rust workspace with the following crates:

### Core Library Crates
- **`atomic_proof/`**: Low-level atomic protocols and gadgets
  - Projection protocols, LiteBullet, and other fundamental building blocks
  - Used by higher-level composite operations
  - Key files: `src/atomic_protocol.rs`, `src/pop/`, `src/protocols/`

- **`mat/`**: Matrix data structures and protocol primitives
  - Matrix operations, R1CS constraint system support (via `r1cs` feature)
  - Utilities for matrix computations in zero-knowledge proofs
  - Key files: `src/protocols/`, `src/utils/matop.rs`

- **`fsproof/`**: Fiat-Shamir transcript and proof infrastructure
  - Non-interactive Fiat-Shamir transformation utilities
  - Shared across all protocol implementations
  - Key files: `src/fs_trans.rs`, `src/batch_r1cs.rs`

- **`poly-commit/`**: Polynomial commitment schemes
  - Fork of arkworks' poly-commit with local modifications
  - Supports KZG, IPA, Marlin, and other commitment schemes
  - Key directories: `src/kzg10/`, `src/ipa_pc/`, `src/marlin/`

### Application Crate
- **`composite/`**: High-level protocols and applications
  - Composite protocol implementations combining atomic operations
  - **Main experiment**: `example/experiment_nn.rs` (neural network verification)
  - Key files: `src/protocols/nn.rs`, `src/protocols/activation.rs`

### Architecture
- All crates target **Rust 2021 edition**
- Default features: `std` and `parallel` for performance
- Modular design allows using individual components independently
- Cross-crate dependencies managed through workspace `Cargo.toml`

### Upstream attribution

The `poly-commit` crate in this workspace is forked from arkworks' implementation:

- Upstream: https://github.com/arkworks-rs/poly-commit

We maintain local changes for experimental features and integration with this workspace.

## Dependencies and Prerequisites

### System Requirements

#### For Fast Demo (depth=16, default configuration)
- Operating System: Linux (recommended) or Windows with WSL2
- Memory: At least 8GB RAM (4GB peak RAM usage observed)
- CPU: Multi-core processor (performance scales with core count)
- Runtime: ~5 minutes with 8 threads

#### For Full Paper Results (depth=1024, requires modification)
- Operating System: Linux (recommended) or Windows with WSL2  
- Memory: At least 256GB RAM for full neural network experiments (103GB peak RAM usage observed)
- CPU: Multi-core processor (64 cores used in paper experiments)
- Runtime: ~35-37 minutes with 64 threads

### Software Dependencies
- **Rust toolchain**: 1.70.0 or later (stable channel)
  - Install via: https://rustup.rs
  - Edition: 2021
- **Required Rust crates** (automatically managed by Cargo):
  - arkworks ecosystem (^0.5.0): `ark-ff`, `ark-std`, `ark-poly`, `ark-relations`, `ark-r1cs-std`, `ark-crypto-primitives`, `ark-groth16`, `ark-serialize`, `ark-ec`, `ark-bls12-381`
  - `rayon` (1.7): For parallel computation
  - `rand` (0.9): Random number generation
  - `chrono` (0.4): Time measurement and logging

### Performance Recommendations
- Build with `--release` flag for optimal performance
- Set `RAYON_NUM_THREADS` environment variable to control parallelism
- Use Linux or WSL2 on Windows for better Rayon parallelism performance

## Building the Artifact

### Basic Build
```bash
# Build all workspace crates in release mode (recommended)
cargo build --workspace --release

# Build in debug mode (for development only)
cargo build --workspace
```

### Verify Build
```bash
# Check that all crates compile successfully
cargo check --workspace --release
```

## Running the Artifact

### Neural Network Example (Main Experiment)

#### Fast Demo (depth=16, default configuration)
The current code is configured for a fast demonstration with 16 layers:

```bash
# Run with optimal settings (recommended)
RAYON_NUM_THREADS=64 cargo run --release -p composite --example experiment_nn

# Run with fewer threads for limited hardware
RAYON_NUM_THREADS=8 cargo run --release -p composite --example experiment_nn
```

**Current default settings:**
- Depth: 16 layers (defined by `const DEPTH: usize = 16;`)  
- Shape: 1024×1024 matrices per layer
- Expected runtime: ~5 minutes (with 8 threads)
- Peak memory: ~4GB

#### Full Paper Results (depth=1024)
To reproduce the exact paper results with 1024 layers, modify the code:

1. **Edit `composite/example/experiment_nn.rs`:**
   ```rust
   // Change line 11 from:
   const DEPTH: usize = 16;
   // To:
   const DEPTH: usize = 1024;
   ```

2. **Rebuild and run:**
   ```bash
  cargo build --workspace --release
  RAYON_NUM_THREADS=64 cargo run --release -p composite --example experiment_nn
   ```

**Paper configuration settings:**
- Depth: 1024 layers
- Shape: 1024×1024 matrices per layer  
- Expected runtime: ~35-37 minutes (with 64 threads)
- Peak memory: ~256GB

### Configuration Options
- **Thread Control**: Set `RAYON_NUM_THREADS` to limit parallel execution
- **Output**: All results are displayed in the console output


## Quantization modes (8-bit vs 16-bit)

The code supports two quantization configurations for the NN example:

- 8-bit (default in this repo snapshot)
	- `mat/src/lib.rs`: `pub type MyShortInt = i8;`
	- `poly-commit/src/lib.rs`: `pub type MyInt = i32;`

- 16-bit
	- Change `mat/src/lib.rs`:
		- from `pub type MyShortInt = i8;`
		- to   `pub type MyShortInt = i16;`
	- Change `poly-commit/src/lib.rs`:
		- from `pub type MyInt = i32;`
		- to   `pub type MyInt = i64;`

Rebuild the workspace after making changes.

## Network Depth Configuration

### Current Default (depth=16)
The repository is configured by default to run a fast demonstration with 16 layers:
- File: `composite/example/experiment_nn.rs`
- Line 11: `const DEPTH: usize = 16;`
- Purpose: Quick validation and demonstration (~5 minutes runtime)

### Paper Results (depth=1024)
To reproduce the full paper results with 1024 layers:

1. **Modify the depth constant:**
   ```rust
   // In composite/example/experiment_nn.rs, line 11
   // Change from:
   const DEPTH: usize = 16;
   // To:
   const DEPTH: usize = 1024;
   ```

2. **Rebuild and run:**
   ```bash
  cargo build --workspace --release
  RAYON_NUM_THREADS=64 cargo run --release -p composite --example experiment_nn
   ```

### System Requirements by Configuration
- **depth=16**: 8GB RAM minimum, ~5 minutes runtime (8 threads)
- **depth=1024**: 256GB RAM minimum, ~35-37 minutes runtime (64 threads)

**Note**: The depth=1024 configuration requires substantial computational resources and is intended for reproducing the exact paper benchmarks.

## Testing

### Run All Tests
```bash
# Run all tests across the workspace (recommended)
cargo test --workspace --release
```

### Run Tests for Specific Crates
```bash
# Test atomic proof protocols
cargo test -p atomic_proof --release

# Test matrix operations and utilities
cargo test -p mat --release

# Test composite protocols
cargo test -p composite --release

# Test Fiat-Shamir transcript functionality
cargo test -p fsproof --release

# Test polynomial commitment schemes
cargo test -p ark-poly-commit --release
```

### Test Output Interpretation
- All tests should pass for a correct setup
- Test failures may indicate environment issues or missing dependencies
- Some tests may be resource-intensive and take several minutes to complete

## Experimental Results

### System and Setup

We conducted experiments on an Alibaba Cloud Elastic Compute Service (ECS) instance equipped with 64 Intel Xeon 6982P-C (x86_64) vCPUs clocked between 0.8–3.9 GHz and 256 GB RAM. The server has a 504 MB shared L3 cache and runs Ubuntu 22.04 LTS (Linux kernel 5.15).

### Model Configurations

#### Fast Demo Configuration (depth=16, default)
- **Layers**: 16 layers (for quick demonstration)
- **Layer transformation**: Y = sigmoid(W X + B)
- **Matrix dimensions**: 1024×1024 per layer
- **Parameters per layer**: 1,049,600 (1024×1024 + 1024)
- **Total parameters**: ~16.8 million
- **Expected performance**:
  - Runtime: ~5 minutes (with RAYON_NUM_THREADS=8)
  - Peak RAM: ~4GB
  - Proof size: ~87KB

#### Paper Results Configuration (depth=1024)
- **Layers**: 1024 layers (full paper experiments)
- **Layer transformation**: Y = sigmoid(W X + B)  
- **Matrix dimensions**: 1024×1024 per layer
- **Parameters per layer**: 1,049,600 (1024×1024 + 1024)
- **Total parameters**: ~2^30 (approximately 1.07 billion)

Where for each layer:
- W: A 1024×1024 weight matrix, quantized to 8-bit integers
- X: A 1024-dimensional input vector, quantized to 8-bit integers  
- Y: A 1024-dimensional output vector
- B: A 1024-dimensional bias vector, quantized to 8-bit integers

### Performance summary

Overall metrics for the NN example under two quantization modes (`RAYON_NUM_THREADS=64`).

| Metric                               | 8-bit (i8/i32) | 16-bit (i16/i64) |
|--------------------------------------|----------------:|-----------------:|
| Commitment time — parameters (s)     | 1276.80         | 1360.50          |
| Commitment time — structure (s)      | 747.07          | 737.30           |
| Commitment size (KB)                 | 42.27           | 42.27            |
| Prover time — total (s)              | 2090.71         | 2210.88          |
| Proof size (KB)                      | 98.84           | 98.84            |
| Verifier time (ms)                   | 173.19          | 172.72           |
| Peak RAM (GB)                        | 87.42           | 103.63           |

### Prover time breakdown

Detailed decomposition of prover wall-clock time.

| Component                           | 8-bit (s) | 16-bit (s) |
|-------------------------------------|----------:|-----------:|
| Auxiliary input commitment          | 733.08    | 755.61     |
| Proof reduction                     | 218.80    | 293.16     |
| PoP proving                         | 775.36    | 765.05     |
| Leaf commitment opening             | 25.46     | 31.23      |
| Fiat–Shamir transform proving       | 338.01    | 356.83     |
| Total                               | 2090.71   | 2210.88    |
| v.s Unverified computation          | 3.53      | 4.64       |

### Notes on measurement

- Build with `--release` and set `RAYON_NUM_THREADS` explicitly.
- All output is displayed directly in the console.
- Both PoP proving and Fiat-Shamir transform proving invokes Groth16 prover, which we havent turn on the parallel feature  

### Notes on RAM usage

- Peak RAM is roughly proportional to the total number of NN parameters.
- Introducing intermediate (per-layer or per-block) commitments partitions a large model into smaller sub-NNs, reducing the prover's peak RAM.
- However, decomposing a large NN into smaller components increases the overall proof size and the verifier time.

## Interpreting Output

### Neural Network Experiment Output
The `experiment_nn` example produces detailed timing and memory usage statistics:

#### Key Metrics
- **Commitment time**: Time to commit to model parameters and structure
- **Prover time**: Total time for zero-knowledge proof generation
- **Proof size**: Size of the generated proof
- **Verifier time**: Time required for proof verification
- **Peak RAM**: Maximum memory consumption during execution

#### Output Format
- **Console output**: Real-time progress and summary statistics displayed directly in the terminal
- **Results**: All timing measurements, memory usage, and performance metrics are printed to stdout

#### Performance Interpretation
- **Baseline comparison**: "v.s Unverified computation" shows overhead factor
- **Component breakdown**: Detailed timing for each proof generation phase
- **Memory usage**: Peak RAM indicates scalability limits

### Platform-Specific Notes

#### Windows/WSL Recommendations
- **WSL2 required**: Native Windows performance is suboptimal for Rayon parallelism
- **Thread control**: Set `RAYON_NUM_THREADS` environment variable in bash:
  ```bash
  # Recommended for reproduction (matches paper experiments)
  RAYON_NUM_THREADS=64 cargo run --release -p composite --example experiment_nn
  
  # For limited hardware
  RAYON_NUM_THREADS=8 cargo run --release -p composite --example experiment_nn
  ```

#### Linux Recommendations
- **Native performance**: Optimal execution environment
- **Memory management**: Monitor system memory usage during large experiments
- **Core scaling**: Performance scales approximately linearly with CPU core count

## License

This project is licensed under either of:

- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Citation

If you use this codebase in academic work, please cite the paper:

Mingshu Cong, Sherman S. M. Chow, Siu Ming Yiu, and Tsz Hon Yuen. Scalable zkSNARKs for Matrix Computations: A Generic Framework for Verifiable Deep Learning. To appear in Asiacrypt 2025 <https://eprint.iacr.org/2025/1646>
