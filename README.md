# Parallel MST Benchmark — Kruskal & Borůvka on SNAP Graphs

> **CS 611 · Design and Analysis of Algorithms**
> Sequential vs Parallel Minimum Spanning Tree algorithms benchmarked on large-scale SNAP datasets in **Rust** (Rayon), **Python** (Numba JIT), and **C++** (`std::thread`).

---

## Algorithms

### Core Variants (benchmarked across all three languages)

| Key | Name | Type | Strategy |
|-----|------|------|----------|
| `kruskal` | Kruskal (Seq) | Sequential | Sort edges by weight, greedily add non-cycle edges via Union-Find. O(E log E) |
| `boruvka_seq` | Borůvka (Seq) | Sequential | Each component finds cheapest outgoing edge → merge → contract. O(E log V) |
| `boruvka_par` | Borůvka (Par) | Parallel | Parallel comp-ID flatten + edge contraction; sequential find-min and merge |

### Ablation Variants (Rust & Python only)

| Key | Name | Purpose |
|-----|------|---------|
| `boruvka_seq_nc` | Borůvka (Seq, No Contraction) | Measures the cost of skipping graph contraction — scans all edges every round |
| `boruvka_par_nc` | Borůvka (Par, No Contraction) | Parallel variant without contraction — demonstrates contraction necessity |

### Experimental Variants (available but not benchmarked by default)

| Key | Name | Description |
|-----|------|-------------|
| `boruvka_par_fr` | Borůvka (Par, Fold-Reduce) | Rayon fold/reduce for find-min (high overhead, experimental) |
| `boruvka_pooled` | Borůvka (Pooled) | Chunked parallel find-min over edges with local reduction |
| `boruvka_groups` | Borůvka (Groups) | CSR-indexed parallel find-min over components |
| `petgraph` | Petgraph Kruskal | Third-party Rust MST ([petgraph](https://docs.rs/petgraph)) |
| `networkx` | NetworkX Kruskal | Third-party Python MST ([NetworkX](https://networkx.org)) |

### Parallelism Models per Language

| Language | Runtime | Parallel Find-Min | Parallel Contraction | Thread Model |
|----------|---------|-------------------|---------------------|-------------|
| **Rust** | Rayon | Sequential (default) | `par_iter` comp-ID flatten + compact | Work-stealing thread pool |
| **Python** | Numba | Sequential (default) | `prange` comp-ID flatten + filter | Static chunk scheduling |
| **C++** | `std::thread` | Fused with find-min (parallel) | No contraction | Manual thread management |

---

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
- Dependencies via `Cargo.toml`: `rayon 1.10`, `clap 4`, `csv 1.3`, `petgraph 0.7`
- Release profile: `opt-level=3`, LTO fat, single codegen unit

### Python
- Python 3.12+
- Dependencies:
```bash
pip install numba numpy matplotlib networkx
```

### C++
- GCC 14+ or Clang with C++17 and pthreads support
- Build flags: `-O2 -std=c++17 -pthread`

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
# Rust (optimized for native CPU, fat LTO)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# C++
g++ -O2 -std=c++17 -pthread -o mst_cpp mst_cpp.cpp
```

### 4. Run benchmarks

**Option A — Automated (recommended):**
```bash
chmod +x run_benchmarks.sh

./run_benchmarks.sh              # Everything: Rust + C++ + Python, all datasets, then plots
./run_benchmarks.sh rust         # Rust only, all datasets
./run_benchmarks.sh python       # Python only, all datasets
./run_benchmarks.sh cpp          # C++ only, all datasets
./run_benchmarks.sh rust amazon  # Rust + Amazon only
./run_benchmarks.sh rust road    # Rust + roadNet-CA only
./run_benchmarks.sh rust orkut   # Rust + com-Orkut only
./run_benchmarks.sh threads      # Thread scaling sweep (Rust)
./run_benchmarks.sh plots        # Regenerate plots from latest run
```

**Option B — Individual runs:**

```bash
# Rust — 5 algorithms, 8 threads, scalability + speedup
./target/release/mst-bench \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --algorithms kruskal,boruvka_seq,boruvka_seq_nc,boruvka_par,boruvka_par_nc \
  --num-threads 8 \
  --experiment both \
  --runs 3 \
  --output-dir logs/rust/amazon0302

# Python — 5 algorithms, Numba JIT
python3.12 mst_python.py \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --algorithms kruskal,boruvka_seq,boruvka_seq_nc,boruvka_par,boruvka_par_nc \
  --default-threads 8 \
  --experiment both \
  --runs 3 \
  --output-dir logs/python/amazon0302 \
  --no-plot

# C++ — 3 algorithms (Kruskal, Borůvka-Seq, Borůvka-Par)
./mst_cpp \
  --dataset datasets/amazon0302.txt \
  --sizes 5000,10000,25000,50000,100000,0 \
  --algorithms kruskal,boruvka_seq,boruvka_par \
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
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to run (comma-separated) |
| `--num-threads` | `0` (all cores) | Thread count for Rayon pool |
| `--runs` | `3` | Repetitions per measurement |
| `--experiment` | `both` | `scalability`, `speedup`, or `both` |
| `--threads` | `2,4,8,12,16` | Thread counts for speedup sweep |
| `--output-dir` | `results/rust` | Output directory for CSVs |
| `--seed` | `42` | Random seed for edge weights (LCG PRNG) |
| `--weight-min` | `1` | Minimum edge weight |
| `--weight-max` | `1000` | Maximum edge weight |

### Python (`python3.12 mst_python.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000,10000,...` | Comma-separated vertex counts (`0` = full dataset) |
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to run (comma-separated) |
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
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to run |
| `--runs` | `3` | Repetitions per measurement |
| `--output-dir` | *(none)* | Output directory for CSV (no output if omitted) |

### Plot Generator (`python3.12 generate_plots.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | latest | Timestamp directory under `logs/` to read data from |
| `--no-errorbars` | — | Disable error bars (min/max range) on all plots |

---

## Benchmark Configuration

The `run_benchmarks.sh` script defines these defaults at the top:

```bash
RUNS=3
NUM_THREADS=8
RUST_ALGORITHMS="kruskal,boruvka_seq,boruvka_seq_nc,boruvka_par,boruvka_par_nc"
PYTHON_ALGORITHMS="kruskal,boruvka_seq,boruvka_seq_nc,boruvka_par,boruvka_par_nc"
CPP_ALGORITHMS="kruskal,boruvka_seq,boruvka_par"
THREAD_COUNTS="2,4,8,12,16"

AMAZON_SIZES="5000,10000,25000,50000,100000,0"
ROAD_SIZES="50000,200000,500000,0"
ORKUT_SIZES="50000,200000,500000,1000000,0"
```

---

## Project Structure

```
.
├── src/main.rs              # Rust: 8 algorithm variants + benchmark CLI (Rayon, Petgraph)
├── mst_python.py            # Python: 8 algorithm variants + benchmark CLI (Numba, NetworkX)
├── mst_cpp.cpp              # C++: Kruskal + Borůvka-Seq + Borůvka-Par (std::thread + mutex)
├── main.py                  # Legacy Python benchmark orchestrator
├── run_benchmarks.sh         # Automated benchmark runner (Rust + Python + C++, all datasets)
├── generate_plots.py         # Publication-quality annotated plots (7 figures)
├── Cargo.toml               # Rust dependencies (rayon, clap, csv, petgraph)
├── pyproject.toml            # Python project config
├── datasets/                 # SNAP datasets (download separately, git-ignored)
│   ├── amazon0302.txt
│   ├── roadNet-CA.txt
│   └── com-orkut.ungraph.txt
├── logs/                     # Benchmark CSV outputs (timestamped runs, git-ignored)
│   └── <YYYY-MM-DD_HH-MM-SS>/
│       ├── rust/<dataset>/scalability_*.csv, speedup_*.csv
│       ├── python/<dataset>/scalability_*.csv, speedup_*.csv
│       ├── cpp/<dataset>/scalability_*.csv
│       └── figures/          # Generated PNG plots
└── report/                   # LaTeX report + slides
    ├── CS611_Project_Report.tex
    ├── CS611_Project_Slides.tex
    └── figures/              # Publication-quality PNG plots
```

---

## How It Works

### Edge Loading & Weight Assignment

All three implementations follow an identical pipeline:
1. Read SNAP tab/space-separated edge lists
2. Filter by vertex ID for size sweeps (`--sizes`)
3. Deduplicate into undirected edges: `(min(u,v), max(u,v))`
4. Sort edges lexicographically for deterministic ordering
5. Remap vertex IDs to contiguous `0..N-1`
6. Assign deterministic random weights using a **portable LCG PRNG** (seed=42)

The Rust and Python implementations use identical LCG constants (`mult=6364136223846793005`, `inc=1442695040888963407`) ensuring **exact MST weight agreement** across both languages at all graph sizes. The C++ implementation uses `std::minstd_rand` (different PRNG), so its MST weights differ but are equally valid.

### Borůvka Phases (per round)

```
┌─────────────────────────────────────────────────┐
│ Round k                                          │
│  1. Flatten comp-IDs  (parallel in boruvka_par)  │
│  2. Find minimum      (sequential — Amdahl's f)  │
│  3. Merge (Union-Find) (sequential)              │
│  4. Contract edges     (parallel in boruvka_par)  │
│     → removes ~50% of edges per round            │
└─────────────────────────────────────────────────┘
     Repeats O(log V) rounds until 1 component
```

### No-Contraction Ablation

The `_nc` variants skip Phase 4 (contraction), scanning ALL original edges every round. This empirically demonstrates that contraction reduces total work from O(E · log V) to O(E) amortized — a critical optimization.

### Parallelization Strategies

| Variant | Phase 1 (Flatten) | Phase 2 (Find-Min) | Phase 3 (Merge) | Phase 4 (Contract) |
|---------|:-:|:-:|:-:|:-:|
| **boruvka_par (Rust)** | ✅ `par_iter_mut` | ❌ sequential | ❌ inline find+union | ✅ `par_iter_mut` |
| **boruvka_par (Python)** | ✅ `prange` | ❌ sequential | ❌ sequential | ✅ `prange` |
| **boruvka_par (C++)** | ✅ fused with find-min | ✅ `std::thread` + `std::mutex` | ❌ sequential | ❌ no contraction |
| **boruvka_par_fr (Rust)** ⚠️ | ✅ `par_iter` | ✅ Rayon `fold`/`reduce` | ❌ sequential | ✅ `par_iter` |
| **boruvka_pooled (Rust)** ⚠️ | ✅ `par_iter` | ✅ chunked `par_iter` + merge | ❌ sequential | ❌ sequential |
| **boruvka_groups (Rust)** ⚠️ | ✅ `par_iter` | ✅ CSR per-component | ❌ sequential | ❌ sequential |

> ⚠️ **Slower in practice:** `boruvka_par_fr`, `boruvka_pooled`, and `boruvka_groups` parallelize find-min but allocate per-chunk `O(n)` arrays each round, making them slower than the default `boruvka_par` which keeps find-min sequential and avoids this overhead.

### Performance Optimizations (Rust)

The Rust implementation includes several performance-critical optimizations:

1. **Struct-of-Arrays (SoA) layout**: Edge data stored as separate `eu[]`, `ev[]`, `ew[]` arrays for SIMD-friendly access
2. **Direct edge sort (Kruskal)**: Sort `(w,u,v)` tuples directly instead of indirect index sort — eliminates cache misses
3. **Zero-clone benchmark loop**: Algorithms borrow `&EdgeArrays` — no outer clone per run
4. **Pre-allocated buffers**: `comp_ids`, `cheapest_w`, `cheapest_idx` allocated once and reused across all Borůvka rounds
5. **Inline find+union**: `boruvka_par` merge phase uses inline path-splitting find and union-by-rank — no `UnionFind` struct clone per round
6. **Unsafe parallel flatten**: `comp_ids` populated via `par_iter_mut` with raw pointer reads from `parent[]` (safe because parent is read-only during flatten phase)
7. **Release profile**: `opt-level=3`, fat LTO, single codegen unit, native CPU target

### Validation

- **Cross-language**: Rust and Python MST weights match exactly at all graph sizes (same LCG PRNG)
- **Third-party**: NetworkX (Python) and Petgraph (Rust) independently compute identical MST weights
- **Intra-language**: All algorithm variants (Kruskal, Borůvka-Seq, Borůvka-Par) produce the same MST weight

---

## Generating Plots

After collecting benchmark results, generate the 7 publication figures:

```bash
# Using latest run
python3.12 generate_plots.py

# Using a specific run
python3.12 generate_plots.py --run-id 2026-05-16_12-30-00

# Without error bars (cleaner figures)
python3.12 generate_plots.py --no-errorbars
```

### Output Figures

| # | Filename | Description |
|---|----------|-------------|
| 1 | `scalability_roadNet-CA.png` | Combined scalability: Python + Rust + C++ on road network |
| 2 | `scalability_amazon0302.png` | Combined scalability: Python + Rust + C++ on Amazon |
| 3 | `scalability_com-orkut.png` | Combined scalability: Python + Rust + C++ on Orkut social network |
| 4 | `parallel_speedup.png` | 2×3 grid: speedup vs thread count (Python/Rust × Road/Amazon/Orkut) |
| 5 | `parallel_efficiency.png` | 2×3 grid: parallel efficiency (Python/Rust × Road/Amazon/Orkut) |
| 6 | `python_vs_rust_comparison.png` | Cross-language overlay on all three datasets |
| 7 | `rust_over_python_ratio.png` | Rust/Python speedup ratio per algorithm |
| 8 | `validation_summary.png` | MST weight correctness across all implementations |

### Visual Encoding

All plots use a consistent visual encoding:

| Dimension | Encodes | Legend |
|-----------|---------|--------|
| **Color** | Language | 🔵 Blue = Python, 🟠 Orange = Rust, 🟢 Green = C++ |
| **Marker shape** | Algorithm | ● = Kruskal, ■ = Borůvka-Seq, ▲ = Borůvka-Par, ◆ = Seq-NC, ▼ = Par-NC |
| **Linestyle** | Algorithm type | Solid = with contraction, Dashed = no contraction |

---

## Hardware

All benchmarks in the report were run on:

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 3900X (12 cores / 24 threads, boost to 4.67 GHz) |
| RAM | 64 GB DDR4-3200 |
| OS | Linux (kernel 6.x) |
| Rust | rustc 1.86+ (`--release`, `-C target-cpu=native`, fat LTO) |
| Python | 3.12 + Numba 0.61 + NumPy |
| C++ | GCC 14+ (`-O2 -std=c++17 -pthread`) |

---

## Report

The LaTeX report is in `report/CS611_Project_Report.tex`. It covers:

1. **Introduction** — MST problem, motivation for parallel Borůvka
2. **Algorithm Design** — Kruskal, Borůvka with/without contraction, parallel strategies
3. **Implementation** — Tri-language architecture, Union-Find, LCG PRNG
4. **Experiments** — 6 experiments (E1–E6): scalability, speedup, efficiency, cross-language, validation
5. **Findings** — Contraction essential, Numba ≈ native for compute-bound loops, Amdahl's Law analysis
6. **Conclusion** — Recommendations for practitioners

Compile with:
```bash
cd report && pdflatex CS611_Project_Report.tex
```

---

## Key Results

| Metric | Finding |
|--------|---------|
| **Contraction** | Borůvka without contraction is O(log V) × slower — empirically confirmed |
| **Numba vs Native** | Numba JIT achieves within 0–50% of Rust/C++ for compute-bound loops |
| **Kruskal: Rust vs Python** | Rust 2–2.4× faster (pdqsort vs introsort) |
| **Peak parallel speedup** | 1.94× at 8 threads (Rust/Rayon on roadNet-CA, 200K vertices) |
| **Amdahl's serial fraction** | f ≈ 0.44 (find-min + merge are sequential) |
| **SMT benefit** | Diminishing returns beyond 12 physical cores |

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
