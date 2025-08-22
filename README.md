## Evalyn: Scalable zkSNARKs for Matrix Computations

This workspace contains code for the experiments in the paper:

**Scalable zkSNARKs for Matrix Computations: A Generic Framework for Verifiable Deep Learning**

It provides building blocks for matrix-oriented zkSNARK protocols and a composite example for verifiable neural networks.

## Status and reproducibility

This repository is a frozen snapshot intended for reproducing the paper's experiments. It will not receive further updates. We are continuing active development privately, and a commercial version will be released later. For reproduction, please use this snapshot as-is; issues and PRs may not be triaged.

## Workspace layout

- `atomic_proof/`: Atomic protocols and gadgets (e.g., projection, litebullet) used by higher-level operations.
- `composite/`: Composite protocols and application logic; includes the (mock) neural network example `experiment_nn`.
- `mat/`: Matrix data structures, utilities, and protocol primitives (R1CS support behind the `r1cs` feature).
- `fsproof/`: Fiat–Shamir transcript and proof helpers used across protocols.
- `poly-commit/`: Polynomial commitment schemes (hybrid, KZG, IPA, Marlin, etc.). This crate is a fork of arkworks' poly-commit with local modifications.

All crates target Rust 2021 edition and default to `std` and `parallel` features where available.

### Upstream attribution

The `poly-commit` crate in this workspace is forked from arkworks' implementation:

- Upstream: https://github.com/arkworks-rs/poly-commit

We maintain local changes for experimental features and integration with this workspace.

## Prerequisites

- Rust toolchain (stable). Install via https://rustup.rs
- Recommended: build with `--release` for performance.
- Optional: Linux or WSL2 on Windows for better performance and parallelism.

## Build

```bash
cargo build --workspace --release
```

## Run the NN example

The NN experiment example lives in the `composite` crate.

```bash
cargo run -p composite --example experiment_nn --release
```

Example logs may be written under `composite/example/*.log`.

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

## Test

Run all tests across the workspace:

```bash
cargo test --workspace --release
```

You can also run tests for a specific crate, for example:

```bash
cargo test -p atomic_proof --release
```

## Experimental results

### System and setup

We conducted experiments on an Alibaba Cloud Elastic Compute Service (ECS) instance equipped with 64 ARMv8 (aarch64) cores clocked at 3.0 GHz and 126 GB RAM. The server has 64 MB shared L3 cache and runs Ubuntu 22.04 LTS (Linux kernel 5.15).

### Model/config

Unless otherwise noted, results are for a mock quantized MLP with layers of the form:

Y = W X + B, where W is a 1024 × 1024 matrix per layer, 1024 layers (e.g., total parameters ≈ 2^30).

### Performance summary

Overall metrics for the NN example under two quantization modes.

| Metric                               | 8-bit (i8/i32) | 16-bit (i16/i64) |
|--------------------------------------|----------------:|-----------------:|
| Commitment time — parameters (s)     | TBD             | 1212.14          |
| Commitment time — structure (s)      | TBD             | 958.80           |
| Commitment size (MB)                 | TBD             | TBD              |
| Prover time — total (s)              | TBD             | TBD              |
| Proof size (MB)                      | TBD             | TBD              |
| Verifier time (s)                    | TBD             | TBD              |
| Peak RAM (GB)                        | TBD             | TBD              |

### Prover time breakdown

Detailed decomposition of prover wall-clock time.

| Component                           | 8-bit (s) | 16-bit (s) |
|-------------------------------------|----------:|-----------:|
| Auxiliary input commitment          | TBD       | 60.26      |
| Proof reduction                     | TBD       | TBD        |
| PoP proving                         | TBD       | TBD        |
| Leaf commitment opening             | TBD       | TBD        |
| Fiat–Shamir transform proof         | TBD       | 412.50     |
| Other/overhead                      | TBD       | TBD        |
| Total                               | TBD       | TBD        |

### Notes on measurement

- Build with `--release` and set `RAYON_NUM_THREADS` explicitly.
- Keep raw logs in `composite/example/*.log`.

### Notes on RAM usage

- Peak RAM is roughly proportional to the total number of network parameters (assuming batch size = 1) and also increases with the quantization width.
- Introducing intermediate (per-layer or per-block) commitments partitions a large model into smaller sub-NNs, reducing the prover's peak RAM. This comes at the cost of a roughly proportional increase in proof size (and verifier work).
- In practice, choose a partition granularity (e.g., commit every k layers) that fits the RAM budget while keeping the overall proof size acceptable.

## Windows/WSL notes

- On Windows, running under WSL2 is recommended for better performance with Rayon parallelism.
- To limit threads: set `RAYON_NUM_THREADS` before running (bash examples below).
- For practical prover time, we recommend `RAYON_NUM_THREADS=64`.

```bash
# Suggested for practical prover performance
RAYON_NUM_THREADS=64 cargo run -p composite --example experiment_nn --release
```

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option.

## Citation

If you use this codebase in academic work, please cite the paper:

Scalable zkSNARKs for Matrix Computations: A Generic Framework for Verifiable Deep Learning

An appropriate BibTeX entry can be added here when available.# evalyn_asiacrypt
