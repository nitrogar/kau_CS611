# Parallel MST Benchmark — Kruskal & Borůvka on SNAP Graphs

> **CS611 Project** — Sequential vs Parallel Minimum Spanning Tree algorithms benchmarked on large-scale SNAP datasets in both **Rust** (Rayon) and **Python** (Numba).

## Algorithms

| Key | Name | Type | Strategy |
|-----|------|------|----------|
| `kruskal` | Kruskal (Seq) | Sequential | Sort edges by weight, greedily add non-cycle edges |
| `boruvka_seq` | Borůvka (Seq) | Sequential | Each component finds cheapest outgoing edge, merge, contract |
| `boruvka_par` | Borůvka (Par) | Parallel | Parallel comp-ID + contraction; sequential find-min |
| `boruvka_pooled` | Borůvka (Pooled) | Parallel | Chunked parallel find-min over **edges** (atomic CAS) |
| `boruvka_groups` | Borůvka (Groups) | Parallel | CSR-indexed parallel find-min over **components** (no atomics) |
| `petgraph` / `networkx` | Third-party | Sequential | Petgraph (Rust) / NetworkX (Python) reference implementation |

## Datasets

| Dataset | Vertices | Edges | Source |
|---------|----------|-------|--------|
| Amazon0302 | 262K | 900K | [SNAP](https://snap.stanford.edu/data/amazon0302.html) |
| roadNet-CA | 1.97M | 2.77M | [SNAP](https://snap.stanford.edu/data/roadNet-CA.html) |
| com-Orkut | 3.07M | 117M | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) |

---

## Prerequisites

### Rust
- [Rust toolchain](https://rustup.rs/) (stable, 1.75+)

### Python
- Python 3.12+
- Dependencies:

```bash
pip install numba numpy matplotlib networkx
```

---

## Quick Start

### 1. Clone & enter
```bash
git clone https://github.com/<your-username>/parallel-mst-bench.git
cd parallel-mst-bench
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

### 3. Build Rust
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 4. Run benchmarks

**Option A — Run everything with one command:**
```bash
./run_benchmarks.sh            # All datasets, all algorithms, Rust + Python
./run_benchmarks.sh rust       # Rust only
./run_benchmarks.sh python     # Python only
./run_benchmarks.sh rust amazon   # Rust + Amazon only
./run_benchmarks.sh threads    # Thread scaling sweep on com-Orkut
```

**Option B — Run individual benchmarks:**

```bash
# Rust — Amazon0302, all 6 algorithms
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
```

> **Note:** Use `0` in `--sizes` to mean "load full dataset" (e.g., `--sizes 5000,10000,0`).

---

## CLI Reference

### Rust (`./target/release/mst-bench`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000,10000,...` | Comma-separated vertex counts (`0` = full) |
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to benchmark |
| `--num-threads` | `0` (all cores) | Thread count for parallel algorithms |
| `--runs` | `3` | Repetitions per measurement |
| `--experiment` | `both` | `scalability`, `speedup`, or `both` |
| `--output-dir` | `results/rust` | Output directory for CSVs |
| `--seed` | `42` | Random seed for edge weights |

### Python (`python3.12 mst_python.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to SNAP edge-list file |
| `--sizes` | `5000,10000,...` | Comma-separated vertex counts (`0` = full) |
| `--algorithms` | `kruskal,boruvka_seq,boruvka_par` | Algorithms to benchmark |
| `--default-threads` | `0` (all cores) | Thread count for parallel algorithms |
| `--runs` | `5` | Repetitions per measurement |
| `--experiment` | `both` | `scalability`, `speedup`, or `both` |
| `--output-dir` | `results/python` | Output directory for CSVs |
| `--no-validate` | | Skip NetworkX validation |
| `--no-plot` | | Skip automatic plot generation |

---

## Project Structure

```
.
├── src/main.rs              # Rust implementation (all 6 algorithms + benchmarking CLI)
├── mst_python.py            # Python implementation (all 6 algorithms + benchmarking CLI)
├── Cargo.toml               # Rust dependencies (rayon, clap, petgraph)
├── pyproject.toml            # Python project config
├── run_benchmarks.sh         # One-command benchmark runner
├── generate_plots.py         # Plot generation scripts
├── generate_ahmed_plots.py   # Additional plot scripts
├── generate_final_plots.py   # Thread-scaling plot generator
├── datasets/                 # SNAP datasets (download separately)
├── logs/                     # Benchmark CSV outputs
└── report/                   # LaTeX report and figures
    ├── CS611_Project_Report.tex
    └── figures/
```

## How It Works

### Edge Loading
Both implementations read SNAP tab/space-separated edge lists, filter by vertex ID (for size sweeps), deduplicate into undirected edges, remap vertex IDs to contiguous `0..N-1`, and assign deterministic random weights using a portable LCG PRNG (seed=42) for cross-language reproducibility.

### Borůvka Phases (per round)
1. **Flatten component IDs** — path-splitting find on Union-Find
2. **Find minimum** — each component finds its cheapest outgoing edge
3. **Merge** — Union-Find merge on cheapest edges
4. **Contract** — remove intra-component edges (shrinks edge set ~50% per round)

### Parallelization Strategies
- **Par**: Parallelizes phases 1 & 4 (comp-ID and contraction); phase 2 is sequential
- **Pooled**: Parallelizes phase 2 over *edges* using chunked local arrays + merge
- **Groups**: Parallelizes phase 2 over *components* using CSR edge index (no atomics)

---

## License

MIT
