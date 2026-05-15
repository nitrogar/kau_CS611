# Parallel MST Benchmark — Kruskal & Borůvka on SNAP Graphs

> **CS 611 · Design and Analysis of Algorithms**
> Sequential vs Parallel Minimum Spanning Tree algorithms benchmarked on large-scale SNAP datasets in **Rust** (Rayon), **Python** (Numba JIT), and **C++**.

---

## Algorithms

| Key | Name | Type | Strategy |
|-----|------|------|----------|
| `kruskal` | Kruskal (Seq) | Sequential | Sort edges by weight, greedily add non-cycle edges via Union-Find |
| `boruvka_seq` | Borůvka (Seq) | Sequential | Each component finds cheapest outgoing edge → merge → contract |
| `boruvka_par` | Borůvka (Par) | Parallel | Parallel comp-ID + contraction; sequential find-min |
| `boruvka_pooled` | Borůvka (Pooled) | Parallel | Chunked parallel find-min over **edges** with local reduction |
| `boruvka_groups` | Borůvka (Groups) | Parallel | CSR-indexed parallel find-min over **components** (no atomics) |
| `petgraph` | Petgraph | Sequential | Third-party Rust reference ([petgraph](https://docs.rs/petgraph)) |
| `networkx` | NetworkX | Sequential | Third-party Python reference ([NetworkX](https://networkx.org)) |

### C++ Implementation
The C++ implementation provides a sequential **Kruskal's** algorithm using `std::minstd_rand` for edge weights, serving as an independent cross-language reference point.

## Datasets

All datasets are from the [Stanford SNAP](https://snap.stanford.edu/data/) project.

| Dataset | Vertices | Edges (undirected) | Avg Degree | Domain |
|---------|----------|-------------------|------------|--------|
| [amazon0302](https://snap.stanford.edu/data/amazon0302.html) | 262K | 900K | 6.1 | Product co-purchase |
| [roadNet-CA](https://snap.stanford.edu/data/roadNet-CA.html) | 1.97M | 2.77M | 3.1 | California road network |
| [com-Orkut](https://snap.stanford.edu/data/com-Orkut.html) | 3.07M | 117M | 76.3 | Social network |

---

## Prerequisites

### Rust
- [Rust toolchain](https://rustup.rs/) (stable 1.75+)
- Dependencies managed via `Cargo.toml`: `rayon`, `clap`, `csv`, `petgraph`

### Python
- Python 3.12+
- Install dependencies:

```bash
pip install numba numpy matplotlib networkx
```

### C++
- GCC or Clang with C++17 support (`<filesystem>`, `<chrono>`)

---

## Quick Start

### 1. Clone & enter
```bash
git clone https://github.com/nitrogar/kau_CS611.git
cd kau_CS611
```

### 2. Download datasets
```bash
mkdir -p datasets && cd datasets

# Amazon0302 (15 MB)
wget https://snap.stanford.edu/data/amazon0302.txt.gz && gunzip amazon0302.txt.gz

# roadNet-CA (53 MB)
wget https://snap.stanford.edu/data/roadNet-CA.txt.gz && gunzip roadNet-CA.txt.gz

# com-Orkut (1.7 GB) — optional, for large-scale benchmarks
wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz && gunzip com-orkut.ungraph.txt.gz

cd ..
```

### 3. Build

```bash
# Rust (optimized for native CPU)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# C++
g++ -O2 -std=c++17 -o mst_cpp mst_cpp.cpp
```

### 4. Run benchmarks

**Option A — Automated (recommended):**
```bash
chmod +x run_benchmarks.sh

./run_benchmarks.sh              # Everything: Rust + C++ + Python, all datasets
./run_benchmarks.sh rust         # Rust only, all datasets
./run_benchmarks.sh python       # Python only, all datasets
./run_benchmarks.sh cpp          # C++ only, all datasets
./run_benchmarks.sh rust amazon  # Rust + Amazon only
./run_benchmarks.sh rust road    # Rust + roadNet-CA only
./run_benchmarks.sh rust orkut   # Rust + com-Orkut only
./run_benchmarks.sh threads      # Thread scaling sweep (Rust, com-Orkut)
```

**Option B — Individual runs:**

```bash
# Rust — Amazon0302, all 6 algorithms, 8 threads
./target/release/mst-bench \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --algorithms kruskal,boruvka_seq,boruvka_par,boruvka_pooled,boruvka_groups,petgraph \
  --num-threads 8 \
  --experiment scalability \
  --runs 3 \
  --output-dir logs/rust/amazon0302

# Python — Amazon0302, all 6 algorithms
python3.12 mst_python.py \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --algorithms kruskal,boruvka_seq,boruvka_par,boruvka_pooled,boruvka_groups,networkx \
  --default-threads 8 \
  --experiment scalability \
  --runs 3 \
  --output-dir logs/python/amazon0302 \
  --no-plot

# C++ — Amazon0302, Kruskal
./mst_cpp \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --runs 3 \
  --output-dir logs/cpp/amazon0302
```

> **Note:** Use `0` in `--sizes` to load the full dataset (e.g., `--sizes 5000,10000,0`).

---

## CLI Reference

### Rust (`./target/release/mst-bench`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000,10000,...` | Comma-separated vertex counts (`0` = full dataset) |
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to run |
| `--num-threads` | `0` (all cores) | Thread count for Rayon pool |
| `--runs` | `3` | Repetitions per measurement |
| `--experiment` | `both` | `scalability`, `speedup`, or `both` |
| `--output-dir` | `results/rust` | Output directory for CSVs |
| `--seed` | `42` | Random seed for edge weights (LCG PRNG) |

### Python (`python3.12 mst_python.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000,10000,...` | Comma-separated vertex counts (`0` = full dataset) |
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to run |
| `--default-threads` | `0` (all cores) | Thread count for Numba `prange` |
| `--runs` | `5` | Repetitions per measurement |
| `--experiment` | `both` | `scalability`, `speedup`, or `both` |
| `--output-dir` | `results/python` | Output directory for CSVs |
| `--no-validate` | — | Skip NetworkX validation |
| `--no-plot` | — | Skip automatic plot generation |

### C++ (`./mst_cpp`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000` | Comma-separated vertex counts (`0` = full dataset) |
| `--runs` | `3` | Repetitions per measurement |
| `--output-dir` | *(none)* | Output directory for CSV (no output if omitted) |

---

## Project Structure

```
.
├── src/main.rs              # Rust: 6 algorithms + benchmark CLI (Rayon, Petgraph)
├── mst_python.py            # Python: 6 algorithms + benchmark CLI (Numba, NetworkX)
├── mst_cpp.cpp              # C++: Kruskal sequential + CSV output
├── main.py                  # Python benchmark orchestrator (size/thread sweeps)
├── Cargo.toml               # Rust dependencies (rayon, clap, csv, petgraph)
├── pyproject.toml            # Python project config
├── run_benchmarks.sh         # Automated benchmark runner (Rust + Python + C++)
├── generate_ahmed_plots.py   # Publication-quality annotated plots (10 figures)
├── generate_final_plots.py   # Thread-scaling analysis plots
├── generate_plots.py         # Legacy plot generator
├── datasets/                 # SNAP datasets (download separately, git-ignored)
│   ├── amazon0302.txt
│   ├── roadNet-CA.txt
│   └── com-orkut.ungraph.txt
├── logs/                     # Benchmark CSV outputs (git-ignored)
│   ├── rust/
│   ├── python/
│   └── cpp/
└── report/                   # LaTeX report + generated figures
    ├── CS611_Project_Report.tex
    └── figures/              # 10 publication-quality PNG plots
```

---

## How It Works

### Edge Loading & Weight Assignment
All three implementations:
1. Read SNAP tab/space-separated edge lists
2. Filter by vertex ID for size sweeps (`--sizes`)
3. Deduplicate into undirected edges: `(min(u,v), max(u,v))`
4. Remap vertex IDs to contiguous `0..N-1`
5. Assign deterministic random weights using a **portable LCG PRNG** (seed=42)

The Rust and Python implementations use identical LCG constants (`mult=6364136223846793005`, `inc=1442695040888963407`) ensuring **exact MST weight agreement** across both languages. The C++ implementation uses `std::minstd_rand` (different PRNG), so its MST weights differ but are equally valid.

### Borůvka Phases (per round)
1. **Flatten component IDs** — path-splitting find on Union-Find
2. **Find minimum** — each component finds its cheapest outgoing edge
3. **Merge** — Union-Find merge on cheapest edges
4. **Contract** — remove intra-component edges (shrinks edge set ~50% per round)

Rounds repeat until only 1 component remains → `O(log V)` rounds total.

### Parallelization Strategies
| Variant | Phase 1 (Flatten) | Phase 2 (Find-Min) | Phase 3 (Merge) | Phase 4 (Contract) |
|---------|:-:|:-:|:-:|:-:|
| **Par** | ✅ parallel | ❌ sequential | ❌ sequential | ✅ parallel |
| **Pooled** | ✅ parallel | ✅ chunked over edges | ❌ sequential | ❌ sequential |
| **Groups** | ✅ parallel | ✅ CSR per component | ❌ sequential | ❌ sequential |

### Third-Party Validation
- **NetworkX** (Python): Independent Kruskal implementation — validated at all graph sizes with exact weight match
- **Petgraph** (Rust): Independent Kruskal implementation — validated with exact weight match

---

## Generating Plots

After collecting benchmark results in `logs/`, regenerate the 10 publication figures:

```bash
python3.12 generate_ahmed_plots.py
```

Output goes to `report/figures/`:
- 4 scalability plots (Python/Rust × Road/Amazon)
- 2 parallel speedup plots
- 1 cross-language comparison
- 1 speedup ratio (Rust/Python)
- 1 parallel efficiency plot
- 1 validation summary table

---

## Hardware

All benchmarks in the report were run on:

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 3900X (12 cores / 24 threads, up to 4.67 GHz) |
| RAM | 64 GB DDR4-3200 |
| OS | Linux (kernel 6.x) |
| Rust | rustc 1.86+ (release, `-C target-cpu=native`) |
| Python | 3.12 + Numba 0.61 |
| C++ | GCC 14+ (`-O2 -std=c++17`) |

---

## Authors

| Name | Student ID | Email |
|------|-----------|-------|
| Amer Aljohani | 2601471 | aaljohani0677@stu.kau.edu.sa |
| Ahmad Alharbi | 2601464 | AALHARBI3232@stu.kau.edu.sa |
| Abdullah Alluhaibi | 2601468 | AALLUHAIBI0010@stu.kau.edu.sa |

**Instructor:** Prof. Maher Khemakhem · King Abdulaziz University · Spring 2026

---

## License

MIT
