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

## Algorithm Implementations

> All three languages share the same algorithmic structure: Union-Find for component tracking, identical PRNG (LCG seed=42) for reproducible edge weights, and the same sorted-edges convention. Below are the core implementations for grading reference.

### Union-Find (Disjoint Set)

All MST algorithms use **Union-Find** with **path compression** and **union by rank** for O(α(n)) amortized component operations.

<details>
<summary><b>Python (Numba JIT)</b></summary>

```python
@njit
def uf_init(n):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    return parent, rank

@njit
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path splitting
        x = parent[x]
    return x

@njit
def uf_union(parent, rank, a, b):
    a = uf_find(parent, a)
    b = uf_find(parent, b)
    if a == b:
        return False
    if rank[a] < rank[b]:
        a, b = b, a
    parent[b] = a
    if rank[a] == rank[b]:
        rank[a] += 1
    return True
```

</details>

<details>
<summary><b>Rust</b></summary>

```rust
struct UnionFind {
    parent: Vec<i32>,
    rank: Vec<i32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n as i32).collect(), rank: vec![0; n] }
    }

    fn find(&mut self, mut x: i32) -> i32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: i32, b: i32) -> bool {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb { return false; }
        if self.rank[ra as usize] < self.rank[rb as usize] { std::mem::swap(&mut ra, &mut rb); }
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] { self.rank[ra as usize] += 1; }
        true
    }
}
```

</details>

<details>
<summary><b>C++</b></summary>

```cpp
int find(int i, vector<int> &parent) {
    if (parent[i] == i) return i;
    return parent[i] = find(parent[i], parent);
}

bool unionfunction(int sn, int dn, vector<int> &parent, vector<int> &rank) {
    int rootSN = find(sn, parent);
    int rootDN = find(dn, parent);
    if (rootSN != rootDN) {
        if (rank[rootSN] < rank[rootDN]) parent[rootSN] = rootDN;
        else if (rank[rootSN] > rank[rootDN]) parent[rootDN] = rootSN;
        else { parent[rootSN] = rootDN; rank[rootDN]++; }
        return true;
    }
    return false;
}
```

</details>

---

### 1. Kruskal's Algorithm (Sequential) — `kruskal`

**Strategy:** Sort all edges by weight, greedily add lightest edge that doesn't form a cycle. **O(E log E)**

<details>
<summary><b>Python (Numba JIT)</b> — <code>mst_python.py:75</code></summary>

```python
@njit
def kruskal_numba(n, eu, ev, ew):
    order = np.argsort(ew)                          # Step 1: Sort — O(E log E)
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    for idx in range(len(order)):                    # Step 2: Greedy selection — O(E·α(n))
        i = order[idx]
        u, v, w = eu[i], ev[i], ew[i]
        if uf_find(parent, u) != uf_find(parent, v):
            uf_union(parent, rank, u, v)
            mst_weight += w
            mst_count += 1
            if mst_count == n - 1:                   # Step 3: Early termination
                break
    return mst_weight, mst_count
```

</details>

<details>
<summary><b>Rust</b> — <code>src/main.rs:118</code></summary>

```rust
fn kruskal(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let m = edges.len();
    let mut sorted: Vec<(i32, i32, i32)> = (0..m)
        .map(|i| (edges.ew[i], edges.eu[i], edges.ev[i]))
        .collect();
    sorted.sort_unstable();                          // Step 1: Sort — O(E log E)

    let mut uf = UnionFind::new(n);
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    for &(w, u, v) in &sorted {                      // Step 2: Greedy selection
        if uf.find(u) != uf.find(v) {
            uf.union(u, v);
            mst_weight += w as i64;
            mst_count += 1;
            if mst_count == n - 1 { break; }
        }
    }
    (mst_weight, mst_count)
}
```

</details>

<details>
<summary><b>C++</b> — <code>mst_cpp.cpp:108</code></summary>

```cpp
pair<long long, double> run_kruskal(const vector<Edge> &edges, int n) {
    vector<int> parent(n);
    vector<int> rank(n, 0);
    for (int i = 0; i < n; i++) parent[i] = i;

    vector<Edge> edges_copy = edges;
    auto t_start = chrono::high_resolution_clock::now();
    sort(edges_copy.begin(), edges_copy.end());       // Step 1: Sort — O(E log E)

    vector<Edge> mst;
    for (size_t i = 0; i < edges_copy.size(); i++) {  // Step 2: Greedy selection
        const Edge &edge = edges_copy[i];
        if (unionfunction(edge.s, edge.d, parent, rank)) {
            mst.push_back(edge);
        }
    }
    auto t_end = chrono::high_resolution_clock::now();
    double elapsed_s = chrono::duration<double>(t_end - t_start).count();

    long long mst_weight = 0;
    for (const auto &e : mst) mst_weight += e.w;
    return {mst_weight, elapsed_s};
}
```

</details>

---

### 2. Borůvka Sequential (with Contraction) — `boruvka_seq`

**Strategy:** Each round, every component finds its cheapest outgoing edge → merge → contract intra-component edges. **O(E log V)** — edge set shrinks ~50% per round.

<details>
<summary><b>Python (Numba JIT)</b> — <code>mst_python.py:121</code></summary>

```python
@njit
def boruvka_seq_numba(n, eu, ev, ew):
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu)
    cheapest_w = np.empty(n, dtype=np.int32)
    cheapest_idx = np.empty(n, dtype=np.int32)
    n_comp = n

    while n_comp > 1:
        # Phase 1: Find-Min — cheapest outgoing edge per component
        cheapest_w[:] = np.iinfo(np.int32).max
        cheapest_idx[:] = -1
        found = False
        for i in range(m):
            cu = uf_find(parent, eu[i])
            cv = uf_find(parent, ev[i])
            if cu != cv:
                w = ew[i]
                if w < cheapest_w[cu]:
                    cheapest_w[cu] = w; cheapest_idx[cu] = i
                if w < cheapest_w[cv]:
                    cheapest_w[cv] = w; cheapest_idx[cv] = i
                found = True
        if not found: break

        # Phase 2: Merge — union on cheapest edges
        merged = 0
        for c in range(n):
            idx = cheapest_idx[c]
            if idx >= 0:
                if uf_union(parent, rank, eu[idx], ev[idx]):
                    mst_weight += ew[idx]; mst_count += 1; merged += 1
        if merged == 0: break
        n_comp -= merged

        # Phase 3: Contract — remove intra-component edges
        new_m = 0
        for i in range(m):
            if uf_find(parent, eu[i]) != uf_find(parent, ev[i]):
                eu[new_m] = eu[i]; ev[new_m] = ev[i]; ew[new_m] = ew[i]
                new_m += 1
        m = new_m

    return mst_weight, mst_count
```

</details>

<details>
<summary><b>Rust</b> — <code>src/main.rs:160</code></summary>

```rust
fn boruvka_seq(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut uf = UnionFind::new(n);
    let (mut mst_weight, mut mst_count) = (0i64, 0usize);
    let (mut eu, mut ev, mut ew) = (edges.eu.clone(), edges.ev.clone(), edges.ew.clone());
    let mut m = eu.len();
    let mut n_comp = n;
    let mut cheapest_w = vec![i32::MAX; n];
    let mut cheapest_idx: Vec<i32> = vec![-1; n];

    while n_comp > 1 {
        // Phase 1: Find-Min
        cheapest_w.iter_mut().for_each(|x| *x = i32::MAX);
        cheapest_idx.iter_mut().for_each(|x| *x = -1);
        let mut found = false;
        for i in 0..m {
            let (cu, cv) = (uf.find(eu[i]), uf.find(ev[i]));
            if cu != cv {
                let w = ew[i];
                if w < cheapest_w[cu as usize] { cheapest_w[cu as usize] = w; cheapest_idx[cu as usize] = i as i32; }
                if w < cheapest_w[cv as usize] { cheapest_w[cv as usize] = w; cheapest_idx[cv as usize] = i as i32; }
                found = true;
            }
        }
        if !found { break; }

        // Phase 2: Merge
        let mut merged = 0usize;
        for c in 0..n {
            let idx = cheapest_idx[c];
            if idx >= 0 {
                let i = idx as usize;
                if uf.union(eu[i], ev[i]) { mst_weight += ew[i] as i64; mst_count += 1; merged += 1; }
            }
        }
        if merged == 0 { break; }
        n_comp -= merged;

        // Phase 3: Contract
        let mut new_m = 0usize;
        for i in 0..m {
            if uf.find(eu[i]) != uf.find(ev[i]) {
                eu[new_m] = eu[i]; ev[new_m] = ev[i]; ew[new_m] = ew[i]; new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}
```

</details>

> **Note:** C++ uses the No-Contraction variant for `boruvka_seq` (see Section 4 below).

---

### 3. Borůvka Parallel (with Contraction) — `boruvka_par`

**Strategy:** Parallel component-ID flatten + parallel edge contraction. Sequential find-min and merge. Available in **Rust** and **Python**. C++ uses a different parallel approach (see Section 5).

<details>
<summary><b>Python (Numba JIT)</b> — <code>mst_python.py:318</code></summary>

```python
@njit(parallel=True)
def _build_comp_ids(parent, n):
    """Parallel component ID flattening."""
    comp = np.empty(n, dtype=np.int32)
    for i in prange(n):                              # ← PARALLEL (prange)
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        comp[i] = x
    return comp

@njit(parallel=True)
def _contract_edges(eu, ev, ew, parent, m):
    """Parallel edge contraction: mark internal edges."""
    keep = np.empty(m, dtype=np.bool_)
    for i in prange(m):                              # ← PARALLEL (prange)
        cu = uf_find(parent, eu[i])
        cv = uf_find(parent, ev[i])
        keep[i] = (cu != cv)
    return keep

@njit
def boruvka_par_numba(n, eu, ev, ew):
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu); n_comp = n
    cheapest_w = np.empty(n, dtype=np.int32)
    cheapest_idx = np.empty(n, dtype=np.int32)

    while n_comp > 1:
        comp_ids = _build_comp_ids(parent, n)        # ← PARALLEL phase

        # Sequential find-min (avoids per-chunk allocation overhead)
        cheapest_w[:] = np.iinfo(np.int32).max; cheapest_idx[:] = -1
        found = False
        for i in range(m):
            cu, cv = comp_ids[eu[i]], comp_ids[ev[i]]
            if cu != cv:
                w = ew[i]
                if w < cheapest_w[cu]: cheapest_w[cu] = w; cheapest_idx[cu] = i
                if w < cheapest_w[cv]: cheapest_w[cv] = w; cheapest_idx[cv] = i
                found = True
        if not found: break

        # Sequential merge
        merged = 0
        for c in range(n):
            idx = cheapest_idx[c]
            if idx >= 0:
                if uf_union(parent, rank, eu[idx], ev[idx]):
                    mst_weight += ew[idx]; mst_count += 1; merged += 1
        if merged == 0: break
        n_comp -= merged

        keep = _contract_edges(eu, ev, ew, parent, m)  # ← PARALLEL phase
        new_m = 0
        for i in range(m):
            if keep[i]:
                eu[new_m] = eu[i]; ev[new_m] = ev[i]; ew[new_m] = ew[i]; new_m += 1
        m = new_m

    return mst_weight, mst_count
```

</details>

<details>
<summary><b>Rust (Rayon)</b> — <code>src/main.rs:380</code></summary>

```rust
fn boruvka_par(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut parent: Vec<i32> = (0..n as i32).collect();
    let mut rank: Vec<i32> = vec![0; n];
    let (mut mst_weight, mut mst_count) = (0i64, 0usize);
    let (mut eu, mut ev, mut ew) = (edges.eu.clone(), edges.ev.clone(), edges.ew.clone());
    let mut m = eu.len();
    let mut n_comp = n;
    let mut cheapest_w = vec![i32::MAX; n];
    let mut cheapest_idx: Vec<i32> = vec![-1; n];
    let mut comp_ids: Vec<i32> = vec![0; n];

    while n_comp > 1 {
        // PARALLEL: flatten component IDs via Rayon par_iter
        comp_ids.par_iter_mut().enumerate().for_each(|(i, cid)| {
            let mut x = i as i32;
            unsafe {
                let p = parent.as_ptr();
                while *p.add(x as usize) != x { x = *p.add(*p.add(x as usize) as usize); }
            }
            *cid = x;
        });

        // Sequential find-min
        cheapest_w.iter_mut().for_each(|x| *x = i32::MAX);
        cheapest_idx.iter_mut().for_each(|x| *x = -1);
        let mut found = false;
        for i in 0..m {
            let (cu, cv) = (comp_ids[eu[i] as usize] as usize, comp_ids[ev[i] as usize] as usize);
            if cu != cv {
                let w = ew[i];
                if w < cheapest_w[cu] { cheapest_w[cu] = w; cheapest_idx[cu] = i as i32; }
                if w < cheapest_w[cv] { cheapest_w[cv] = w; cheapest_idx[cv] = i as i32; }
                found = true;
            }
        }
        if !found { break; }

        // Sequential merge (inline find + union by rank)
        let mut merged = 0usize;
        for c in 0..n {
            let idx = cheapest_idx[c];
            if idx >= 0 {
                // ... inline find with path splitting + union by rank ...
                // (see full source for details — avoids UnionFind clone overhead)
            }
        }
        if merged == 0 { break; }
        n_comp -= merged;

        // PARALLEL: edge contraction
        // Re-flatten comp_ids, then filter intra-component edges
        let mut new_m = 0usize;
        for i in 0..m {
            if comp_ids[eu[i] as usize] != comp_ids[ev[i] as usize] {
                eu[new_m] = eu[i]; ev[new_m] = ev[i]; ew[new_m] = ew[i]; new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}
```

</details>

---

### 4. Borůvka Sequential — No Contraction — `boruvka_seq_nc`

**Strategy:** Same as `boruvka_seq` but **never removes** intra-component edges. Scans ALL edges every round → O(E·log V). Demonstrates why contraction is essential.

<details>
<summary><b>C++ (used as boruvka_seq)</b> — <code>mst_cpp.cpp:144</code></summary>

```cpp
pair<long long, double> run_boruvka_seq(const vector<Edge> &edges, int n) {
    vector<int> parent(n), rank(n, 0);
    for (int i = 0; i < n; i++) parent[i] = i;
    int components_num = n;
    vector<Edge> mst;

    auto t_start = chrono::high_resolution_clock::now();

    while (components_num > 1) {
        vector<int> cheapest(n, -1);

        // Phase 1: Find-Min (scans ALL edges every round — no contraction)
        for (int i = 0; i < (int)edges.size(); i++) {
            const Edge &edge = edges[i];
            int group1 = find(edge.s, parent);
            int group2 = find(edge.d, parent);
            if (group1 != group2) {
                if (cheapest[group1] == -1 || edge.w < edges[cheapest[group1]].w)
                    cheapest[group1] = i;
                else if (cheapest[group2] == -1 || edge.w < edges[cheapest[group2]].w)
                    cheapest[group2] = i;
            }
        }

        // Phase 2: Merge
        bool merged_any = false;
        for (int i = 0; i < n; i++) {
            if (cheapest[i] != -1) {
                const Edge &edge = edges[cheapest[i]];
                if (unionfunction(edge.s, edge.d, parent, rank)) {
                    mst.push_back(edge);
                    components_num--;
                    merged_any = true;
                }
            }
        }
        if (!merged_any) break;
        // NO Phase 3 — edge set stays at full size
    }
    // ... timing and weight calculation ...
}
```

</details>

<details>
<summary><b>Rust</b> — <code>src/main.rs:249</code> | <b>Python</b> — <code>mst_python.py:205</code></summary>

Same structure as `boruvka_seq` above but without Phase 3 (no `new_m` compaction loop). The edge arrays are never modified.

</details>

---

### 5. Borůvka Parallel — No Contraction — `boruvka_par` (C++) / `boruvka_par_nc` (Rust/Python)

**Strategy:** Parallel find-min with **mutex-guarded** per-component updates. No contraction — scans all edges every round.

<details>
<summary><b>C++ (std::thread + mutex)</b> — <code>mst_cpp.cpp:214</code></summary>

```cpp
pair<long long, double> run_boruvka_par(const vector<Edge> &edges, int n, int num_threads) {
    vector<int> parent(n), rank(n, 0);
    for (int i = 0; i < n; i++) parent[i] = i;
    int components_num = n;
    vector<Edge> mst;

    int par_num_threads = (num_threads > 0) ? num_threads : thread::hardware_concurrency();
    int maximum_edges = edges.size();
    int threads_edge = maximum_edges / par_num_threads;
    vector<mutex> component_locks(n);

    auto t_start = chrono::high_resolution_clock::now();

    while (components_num > 1) {
        vector<int> cheapest(n, -1);
        vector<thread> threads;

        // Phase 1: PARALLEL Find-Min — each thread handles a chunk of edges
        for (int t = 0; t < par_num_threads; t++) {
            int start = t * threads_edge;
            int end = (t == par_num_threads - 1) ? maximum_edges : start + threads_edge;

            threads.push_back(thread([&, start, end]() {
                for (int i = start; i < end; i++) {
                    const Edge &edge = edges[i];
                    int group1 = find(edge.s, parent);
                    int group2 = find(edge.d, parent);
                    if (group1 != group2) {
                        {   // Mutex-guarded update for component group1
                            lock_guard<mutex> lock(component_locks[group1]);
                            if (cheapest[group1] == -1 || edge.w < edges[cheapest[group1]].w)
                                cheapest[group1] = i;
                        }
                        {   // Mutex-guarded update for component group2
                            lock_guard<mutex> lock(component_locks[group2]);
                            if (cheapest[group2] == -1 || edge.w < edges[cheapest[group2]].w)
                                cheapest[group2] = i;
                        }
                    }
                }
            }));
        }
        for (auto &th : threads) if (th.joinable()) th.join();

        // Phase 2: Sequential merge
        bool merged_any = false;
        for (int i = 0; i < n; i++) {
            if (cheapest[i] != -1) {
                if (unionfunction(edges[cheapest[i]].s, edges[cheapest[i]].d, parent, rank)) {
                    mst.push_back(edges[cheapest[i]]);
                    components_num--;
                    merged_any = true;
                }
            }
        }
        if (!merged_any) break;
        // No contraction — scans all edges every round
    }
    // ... timing and weight calculation ...
}
```

</details>

<details>
<summary><b>Rust (Rayon par_iter)</b> — <code>src/main.rs:307</code> | <b>Python (Numba prange)</b> — <code>mst_python.py:386</code></summary>

Same find-min + merge structure but uses **Rayon `par_iter`** (Rust) or **Numba `prange`** (Python) for parallel component-ID flatten. The find-min itself is sequential (avoids per-chunk allocation overhead). No edge contraction.

</details>



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
