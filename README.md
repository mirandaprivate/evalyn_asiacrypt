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
- `poly-commit/`: Polynomial commitment schemes (KZG, IPA, Marlin, etc.). This crate is a fork of arkworks' poly-commit with local modifications.

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
cargo run --release -p composite --features jemalloc --example experiment_nn
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

We conducted experiments on an Alibaba Cloud Elastic Compute Service (ECS) instance equipped with 64 Intel Xeon 6982P-C (x86_64) vCPUs clocked between 0.8–3.9 GHz and 256 GB RAM. The server has a 504 MB shared L3 cache and runs Ubuntu 22.04 LTS (Linux kernel 5.15).

### Model/config: Quantized Multi-Layer Perceptron (MLP)

We evaluated our system using a mock quantized Multi-Layer Perceptron (MLP), a fundamental NN architecture.
The MLP consists of 1024 layers, each defined by the transformation:

	Y = sigmoid(W X + B)

where:
- W: A 1024 × 1024 weight matrix, quantized to 8-bit integers.
- X: A 1024-dimensional input vector, quantized to 8-bit integers.
- Y: A 1024-dimensional output vector.
- B: A 1024-dimensional bias vector, quantized to 8-bit integers.

Each layer contains 1024 x 1024 + 1024 = 1,049,600 parameters, resulting in a total of approximately 2^{30} parameters across 1024 layers.

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
- Keep raw logs in `composite/example/*.log`.
- Both PoP proving and Fiat-Shamir transform proving invokes Groth16 prover, which we havent turn on the parallel feature  

### Notes on RAM usage

- Peak RAM is roughly proportional to the total number of NN parameters.
- Introducing intermediate (per-layer or per-block) commitments partitions a large model into smaller sub-NNs, reducing the prover's peak RAM.
- However, decomposing a large NN into smaller components increases the overall proof size and the verifier time.

## Windows/WSL notes

- On Windows, running under WSL2 is recommended for better performance with Rayon parallelism.
- To limit threads: set `RAYON_NUM_THREADS` before running (bash examples below).
- For practical prover time, we recommend `RAYON_NUM_THREADS=64`.

```bash
# Suggested for practical prover performance
RAYON_NUM_THREADS=64 cargo run --release -p composite --features jemalloc --example experiment_nn
```

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option.

## Citation

If you use this codebase in academic work, please cite the paper:

Mingshu Cong, Sherman S. M. Chow, Siu Ming Yiu, and Tsz Hon Yuen. Scalable zkSNARKs for Matrix Computations: A Generic Framework for Verifiable Deep Learning. To appear in Asiacrypt 2025 <https://eprint.iacr.org/2025/1646>
