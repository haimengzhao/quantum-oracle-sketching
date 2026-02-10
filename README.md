# Quantum Oracle Sketching (QOS)

Code repository for the numerical experiments in our paper "Processing massive classical data on small quantum computers".

This repo contains:
- Core implementations of **quantum oracle sketching** in JAX.
- A benchmark suite for quantum oracle and state sketching.
- Real-dataset experiments (classification and dimension reduction) for demonstrating the exponential memory advantage.

## Overview

**Quantum oracle sketching (QOS)** is a quantum algorithm for loading classical data into a quantum computer.
It instantiates the oracles needed by any quantum query algorithm using only random classical samples, with no full-dataset memory overhead.

This codebase includes two QOS simulation modes:
- `qos_sampling.py`: active random sampling implementation (more direct, heavier in simulation).
- `qos.py`: expected-unitary implementation (used for efficient benchmarking; conservative error upper bound).

They are implemented in JAX, which supports GPU/TPU execution and automatic differentiation.

## Repository Structure

```text
.
├── benchmark.py              # Main benchmark used in the paper
├── data_generation.py        # Random sample generators for vectors/matrices/boolean functions
├── primitives.py             # Shared quantum primitives (e.g., amplitude amplification)
├── qos.py                    # QOS via expected unitaries (main benchmarking path)
├── qos_sampling.py           # QOS via explicit random sampling
├── qsvt.py                   # QSVT utilities + phase generation via pyqsp
├── utils.py                  # Numerical helpers, random instances, block-encoding helpers
├── real_datasets/            # Real-data experiments + plotting scripts
│   ├── *_svm.py              # LS-SVM-style classification accuracy vs machine size
│   ├── *_pca.py              # PCA variance recovery vs machine size
│   ├── *_combine_fig.py      # Combined 2-panel plots for each dataset
│   └── *_size_vs_*.json/.pdf # Precomputed results and figures
└── requirements.txt
```

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `utils.py` enables 64-bit JAX (`jax_enable_x64=True`).
- `json` is part of the Python standard library, so it is not listed in `requirements.txt`.
- If you want GPU JAX wheels, install JAX following official JAX instructions for your CUDA/TPU setup.

### 2) Run the synthetic benchmark (main benchmark figure generation)

From repo root:

```bash
python benchmark.py
```

This generates:
- `benchmark_flat_vector.pdf`
- `benchmark_general_vector.pdf`
- `benchmark_boolean_function.pdf`
- `benchmark_matrix_element.pdf`
- `benchmark_matrix_row_index.pdf`

## Real-Dataset Experiments

`real_datasets/` evaluates machine-size vs performance under feature truncation (via `min_df` or `min_samples`), for:
- IMDb sentiment (text TF-IDF)
- 20 Newsgroups topic data (text TF-IDF)
- PBMC68k single-cell RNA (UMI)
- Dorothea drug-discovery dataset
- Splice dataset (k-mer)

### Run full pipelines from scratch

Run from `real_datasets/` so relative paths and defaults match script expectations.

IMDb:
```bash
python imdb_svm.py
python imdb_pca.py
python imdb_combine_fig.py
```

20 Newsgroups (default averages over 100 random category pairs):
```bash
python 20news_svm.py --n_pairs 100
python 20news_pca.py --n_pairs 100
python 20news_combine_fig.py
```

PBMC68k:
```bash
python pbmc68k_svm.py
python pbmc68k_pca.py
python pbmc68k_combine_fig.py
```

Dorothea:
```bash
python dorothea_svm.py
python dorothea_pca.py
python dorothea_combine_fig.py
```

Splice:
```bash
python splice_svm.py
python splice_pca.py
python splice_combine_fig.py
```

### Dataset source/setup notes

- IMDb: auto-downloaded by `imdb_utils.py` from Stanford ACL IMDb.
- 20 Newsgroups: fetched through `sklearn.datasets.fetch_20newsgroups`.
- PBMC68k: loaded via `scvelo.datasets.pbmc68k` (downloaded/cached automatically).
- Dorothea: download manually from UCI (https://archive.ics.uci.edu/static/public/169/dorothea.zip) and extract to `data_cache/dorothea` (relative to where you run scripts).
- Splice: fetched via `ucimlrepo` (dataset id 69).

## What is Being Compared

The real-dataset scripts compare machine size for three families:
- Classical sparse / QRAM-style storage (lower bounded by nonzeros): `S >= N_nnz`
- Classical streaming (lower bounded by solution vector size): `S >= D`
- Quantum oracle sketching (qubit count + one classical scalar accumulator)

Implemented QOS size formulas in scripts:
- LS-SVM: `S_QOS = 2*ceil(log2(N + 2D)) + ceil(log2(s + 1)) + 3 + 1`
- PCA: `S_QOS = 2*ceil(log2(N + D)) + ceil(log2(s)) + 3 + 1`

Here `N` is sample count, `D` feature dimension, and `s` matrix sparsity.

## Core Files and Roles

- `data_generation.py`: sampling interfaces (`vector_data`, `matrix_data`, `boolean_data`).
- `qos_sampling.py`: explicit sampled-gate assembly; includes oracle/state tests under sampling.
- `qos.py`: expected-unitary assembly (used by `benchmark.py`).
- `qsvt.py`: polynomial angle generation + QSVT application helpers.
- `primitives.py`: amplitude amplification and related utilities.
- `utils.py`: random instance generators, block-encoding helpers, fidelity/infidelity utilities.

## Reproducibility Notes

- Random seeds are fixed in scripts (`np.random.seed(42)` or JAX keys).
- Some full runs are heavy (large sample sweeps and repeated CV/SVD); expect long runtimes.
- Running `qos.py` or `qos_sampling.py` directly executes built-in tests with large default sizes.

## Citation

If you find this repository useful, please consider citing our paper.
