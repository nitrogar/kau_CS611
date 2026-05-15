#!/usr/bin/env bash
# ============================================================
# run_benchmarks.sh — Run all MST benchmarks (Rust + Python)
# ============================================================
# Usage:
#   ./run_benchmarks.sh              # Run everything
#   ./run_benchmarks.sh rust         # Rust only
#   ./run_benchmarks.sh python       # Python only
#   ./run_benchmarks.sh rust amazon  # Rust + Amazon only
#   ./run_benchmarks.sh rust orkut   # Rust + Orkut only
#   ./run_benchmarks.sh threads      # Thread scaling only
# ============================================================

set -euo pipefail
set -x  # Print each command before execution

# ─── Configuration ──────────────────────────────────────────
RUNS=3
NUM_THREADS=24
ALGORITHMS="kruskal,boruvka_seq,boruvka_par,boruvka_pooled,boruvka_groups,petgraph"
PYTHON_ALGORITHMS="kruskal,boruvka_seq,boruvka_par,boruvka_pooled,boruvka_groups,networkx"
THREAD_COUNTS="2 4 8 16"

# Datasets
AMAZON="datasets/amazon0302.txt"
ROAD="datasets/roadNet-CA.txt"
ORKUT="datasets/com-orkut.ungraph.txt"

# Vertex sizes per dataset (0 = full dataset)
AMAZON_SIZES="5000,10000,25000,50000,100000,0"
ROAD_SIZES="50000,200000,500000,0"
ORKUT_SIZES="50000,200000,500000,1000000,0"

# Output — each run gets a unique timestamped directory
LOG_DIR="logs"
RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)

# ─── Helpers ────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

header() { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════${RESET}"; echo -e "${BOLD}  $1${RESET}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════${RESET}"; }
done_msg() { echo -e "${GREEN}  ✓ Done → $1${RESET}"; }

# ─── Build ──────────────────────────────────────────────────
build_rust() {
    header "Building Rust (release, native CPU)"
    RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | tail -2
}

build_cpp() {
    header "Building C++ (release, -O2)"
    g++ -O2 -std=c++17 -o mst_cpp mst_cpp.cpp
}

# ─── Rust Benchmarks ────────────────────────────────────────
run_rust_amazon() {
    header "Rust: Amazon0302 (262K V, 900K E)"
    local out="$LOG_DIR/rust/amazon0302/$RUN_ID"
    mkdir -p "$out"
    ./target/release/mst-bench \
        --dataset "$AMAZON" \
        --sizes "$AMAZON_SIZES" \
        --algorithms "$ALGORITHMS" \
        --num-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

run_rust_road() {
    header "Rust: roadNet-CA (1.97M V, 2.77M E)"
    local out="$LOG_DIR/rust/roadNet-CA/$RUN_ID"
    mkdir -p "$out"
    ./target/release/mst-bench \
        --dataset "$ROAD" \
        --sizes "$ROAD_SIZES" \
        --algorithms "$ALGORITHMS" \
        --num-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

run_rust_orkut() {
    header "Rust: com-Orkut (3.07M V, 117M E)"
    local out="$LOG_DIR/rust/com-orkut/$RUN_ID"
    mkdir -p "$out"
    ./target/release/mst-bench \
        --dataset "$ORKUT" \
        --sizes "$ORKUT_SIZES" \
        --algorithms "$ALGORITHMS" \
        --num-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

# ─── Thread Scaling (Rust, Orkut) ───────────────────────────
run_thread_scaling() {
    header "Rust: Thread Scaling on com-Orkut"

    # Sequential baseline
    local out="$LOG_DIR/rust/com-orkut/$RUN_ID/threads/t1_seq"
    mkdir -p "$out"
    echo "  → Sequential baseline (T=1)..."
    ./target/release/mst-bench \
        --dataset "$ORKUT" \
        --sizes "$ORKUT_SIZES" \
        --algorithms boruvka_seq \
        --num-threads 1 \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out"

    # Parallel at each thread count
    for T in $THREAD_COUNTS; do
        out="$LOG_DIR/rust/com-orkut/$RUN_ID/threads/t${T}"
        mkdir -p "$out"
        echo "  → Pooled with T=$T threads..."
        ./target/release/mst-bench \
            --dataset "$ORKUT" \
            --sizes "$ORKUT_SIZES" \
            --algorithms boruvka_pooled \
            --num-threads "$T" \
            --experiment scalability \
            --runs "$RUNS" \
            --output-dir "$out"
    done
    done_msg "$LOG_DIR/rust/com-orkut/$RUN_ID/threads/"
}

# ─── Python Benchmarks ──────────────────────────────────────
run_python_amazon() {
    header "Python: Amazon0302"
    local out="$LOG_DIR/python/amazon0302/$RUN_ID"
    mkdir -p "$out"
    python3.12 mst_python.py \
        --dataset "$AMAZON" \
        --sizes "$AMAZON_SIZES" \
        --algorithms "$PYTHON_ALGORITHMS" \
        --default-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out" \
        --no-plot
    done_msg "$out"
}

run_python_road() {
    header "Python: roadNet-CA"
    local out="$LOG_DIR/python/roadNet-CA/$RUN_ID"
    mkdir -p "$out"
    python3.12 mst_python.py \
        --dataset "$ROAD" \
        --sizes "$ROAD_SIZES" \
        --algorithms "$PYTHON_ALGORITHMS" \
        --default-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out" \
        --no-plot
    done_msg "$out"
}

run_python_orkut() {
    header "Python: com-Orkut (smaller sizes for speed)"
    local out="$LOG_DIR/python/com-orkut/$RUN_ID"
    mkdir -p "$out"
    python3.12 mst_python.py \
        --dataset "$ORKUT" \
        --sizes "50000,200000,500000" \
        --algorithms "$PYTHON_ALGORITHMS" \
        --default-threads "$NUM_THREADS" \
        --experiment scalability \
        --runs "$RUNS" \
        --output-dir "$out" \
        --no-validate --no-plot
    done_msg "$out"
}

# ─── C++ Benchmarks ─────────────────────────────────────────
run_cpp_amazon() {
    header "C++: Amazon0302 (Kruskal only)"
    local out="$LOG_DIR/cpp/amazon0302/$RUN_ID"
    mkdir -p "$out"
    ./mst_cpp \
        --dataset "$AMAZON" \
        --sizes "$AMAZON_SIZES" \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

run_cpp_road() {
    header "C++: roadNet-CA (Kruskal only)"
    local out="$LOG_DIR/cpp/roadNet-CA/$RUN_ID"
    mkdir -p "$out"
    ./mst_cpp \
        --dataset "$ROAD" \
        --sizes "$ROAD_SIZES" \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

run_cpp_orkut() {
    header "C++: com-Orkut (Kruskal only)"
    local out="$LOG_DIR/cpp/com-orkut/$RUN_ID"
    mkdir -p "$out"
    ./mst_cpp \
        --dataset "$ORKUT" \
        --sizes "$ORKUT_SIZES" \
        --runs "$RUNS" \
        --output-dir "$out"
    done_msg "$out"
}

# ─── Main ───────────────────────────────────────────────────
MODE="${1:-all}"
DATASET="${2:-all}"

START=$(date +%s)

case "$MODE" in
    rust)
        build_rust
        case "$DATASET" in
            amazon)  run_rust_amazon ;;
            road)    run_rust_road ;;
            orkut)   run_rust_orkut ;;
            *)       run_rust_amazon; run_rust_road; run_rust_orkut ;;
        esac
        ;;
    python)
        case "$DATASET" in
            amazon)  run_python_amazon ;;
            road)    run_python_road ;;
            orkut)   run_python_orkut ;;
            *)       run_python_amazon; run_python_road; run_python_orkut ;;
        esac
        ;;
    cpp)
        build_cpp
        case "$DATASET" in
            amazon)  run_cpp_amazon ;;
            road)    run_cpp_road ;;
            orkut)   run_cpp_orkut ;;
            *)       run_cpp_amazon; run_cpp_road; run_cpp_orkut ;;
        esac
        ;;
    threads)
        build_rust
        run_thread_scaling
        ;;
    all)
        build_rust
        run_rust_amazon
        run_rust_road
        run_rust_orkut
        run_thread_scaling
        build_cpp
        run_cpp_amazon
        run_cpp_road
        run_cpp_orkut
        run_python_amazon
        run_python_road
        run_python_orkut
        ;;
    *)
        echo "Usage: $0 [rust|python|cpp|threads|all] [amazon|road|orkut]"
        exit 1
        ;;
esac

END=$(date +%s)
ELAPSED=$((END - START))
header "All benchmarks complete in ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
echo "  Results in: $LOG_DIR/"
find "$LOG_DIR" -name "*.csv" -newer /tmp/.bench_start 2>/dev/null | sort || true
