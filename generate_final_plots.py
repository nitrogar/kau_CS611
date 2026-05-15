#!/usr/bin/env python3
"""
Final Benchmark: Thread-Scaling Plots for Parallel MST (Borůvka Pooled)
=======================================================================
Generates two plots:
  1. Time vs Vertices — lines for each (language, thread_count) combo
  2. Speedup vs Vertices — speedup over sequential baseline

Uses com-Orkut (117M edges). Thread count passed via CLI args, not env vars.
"""

import subprocess, os, sys, time, csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── Configuration ──────────────────────────────────────────────────
DATASET = "datasets/com-orkut.ungraph.txt"
DATASET_NAME = "com-orkut"

# Vertex sizes for sweep
RUST_SIZES = [50000, 200000, 500000, 1000000, 3072441]
PYTHON_SIZES = [50000, 200000, 500000]  # Python/Numba is ~20x slower

THREAD_COUNTS = [2, 4, 8, 16]
RUNS = 3
OUTPUT_DIR = "results/final_plots"
FIGURE_DIR = "report/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def run_rust_benchmarks():
    """Run Rust boruvka_pooled + boruvka_seq at each thread count and size."""
    results = []
    sizes_str = ",".join(str(s) for s in RUST_SIZES)

    # Sequential baseline
    print("=" * 65)
    print("RUST: Sequential baseline (boruvka_seq)")
    print("=" * 65)
    cmd = [
        "./target/release/mst-bench",
        "--dataset", DATASET,
        "--sizes", sizes_str,
        "--experiment", "scalability",
        "--runs", str(RUNS),
        "--output-dir", f"{OUTPUT_DIR}/rust_seq",
        "--algorithms", "boruvka_seq",
        "--num-threads", "1",
    ]
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"ERROR: {proc.stderr}")
        return results

    csv_path = f"{OUTPUT_DIR}/rust_seq/scalability_{DATASET_NAME}.ungraph.csv"
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                key = (int(row['n_vertices']),)
                if key not in seen:
                    seen.add(key)
                    results.append({
                        'lang': 'Rust', 'algo': 'boruvka_seq', 'threads': 1,
                        'n_vertices': int(row['n_vertices']),
                        'n_edges': int(row['n_edges']),
                        'median_s': float(row['median_s']),
                    })

    # Parallel: boruvka_pooled at each thread count via --num-threads CLI arg
    for nt in THREAD_COUNTS:
        print(f"\n{'=' * 65}")
        print(f"RUST: boruvka_pooled with {nt} threads (--num-threads {nt})")
        print(f"{'=' * 65}")
        out_dir = f"{OUTPUT_DIR}/rust_t{nt}"
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            "./target/release/mst-bench",
            "--dataset", DATASET,
            "--sizes", sizes_str,
            "--experiment", "scalability",
            "--runs", str(RUNS),
            "--output-dir", out_dir,
            "--algorithms", "boruvka_pooled",
            "--num-threads", str(nt),   # CLI arg, not env var
        ]
        print(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"ERROR: {proc.stderr}")
            continue

        csv_path = f"{out_dir}/scalability_{DATASET_NAME}.ungraph.csv"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                seen = set()
                for row in reader:
                    key = (int(row['n_vertices']),)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            'lang': 'Rust', 'algo': 'boruvka_pooled', 'threads': nt,
                            'n_vertices': int(row['n_vertices']),
                            'n_edges': int(row['n_edges']),
                            'median_s': float(row['median_s']),
                        })

    return results


def run_python_benchmarks():
    """Run Python boruvka_pooled + boruvka_seq at each thread count via --default-threads CLI."""
    results = []

    for nt_config in [1] + THREAD_COUNTS:
        algo = 'boruvka_seq' if nt_config == 1 else 'boruvka_pooled'
        nt = 1 if nt_config == 1 else nt_config
        label = f"Python: {algo} with {nt} threads"

        print(f"\n{'=' * 65}")
        print(label)
        print(f"{'=' * 65}")

        sizes_str = ",".join(str(s) for s in PYTHON_SIZES)
        cmd = [
            sys.executable, "mst_python.py",
            "--dataset", DATASET,
            "--sizes", sizes_str,
            "--experiment", "scalability",
            "--runs", str(RUNS),
            "--output-dir", f"{OUTPUT_DIR}/python_t{nt}_{algo}",
            "--algorithms", algo,
            "--default-threads", str(nt),  # CLI arg, not env var
            "--no-validate",               # skip NetworkX for speed
            "--no-plot",
        ]

        print(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
        if proc.returncode != 0:
            print(f"ERROR: {proc.stderr[-1000:]}")
            continue

        out_dir = f"{OUTPUT_DIR}/python_t{nt}_{algo}"
        csv_path = f"{out_dir}/scalability_{DATASET_NAME}.ungraph.csv"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                seen = set()
                for row in reader:
                    key = (int(row['n_vertices']),)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            'lang': 'Python', 'algo': algo, 'threads': nt,
                            'n_vertices': int(row['n_vertices']),
                            'n_edges': int(row['n_edges']),
                            'median_s': float(row['median_s']),
                        })

    return results


def generate_plots(all_results):
    """Generate the two final plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # Color palette — warm for Rust, cool for Python
    rust_colors = {1: '#C0392B', 2: '#E74C3C', 4: '#E67E22', 8: '#F39C12', 16: '#27AE60'}
    python_colors = {1: '#8E44AD', 2: '#9B59B6', 4: '#2980B9', 8: '#1ABC9C', 16: '#16A085'}
    markers = {1: 'X', 2: 's', 4: 'D', 8: '^', 16: 'v'}

    # ────────── Plot 1: Time vs Vertices ──────────
    fig, ax = plt.subplots(figsize=(12, 7))

    groups = defaultdict(list)
    for r in all_results:
        key = (r['lang'], r['algo'], r['threads'])
        groups[key].append(r)

    for (lang, algo, threads), records in sorted(groups.items()):
        records.sort(key=lambda r: r['n_vertices'])
        xs = [r['n_vertices'] for r in records]
        ys = [r['median_s'] for r in records]
        n_edges_last = records[-1]['n_edges']

        if lang == 'Rust':
            if algo == 'boruvka_seq':
                label = f"Rust Sequential"
                color, ls = '#C0392B', '--'
            else:
                label = f"Rust Pooled (T={threads})"
                color, ls = rust_colors.get(threads, '#333'), '-'
        else:
            if algo == 'boruvka_seq':
                label = f"Python Sequential"
                color, ls = '#8E44AD', '--'
            else:
                label = f"Python Pooled (T={threads})"
                color, ls = python_colors.get(threads, '#666'), '-'

        marker = markers.get(threads, 'o')
        ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                linewidth=2, markersize=7, label=label)

        # Annotate last point
        ax.annotate(f'{ys[-1]:.2f}s', (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    ax.set_xlabel('Number of Vertices (V)', fontsize=13)
    ax.set_ylabel('Median Time (seconds)', fontsize=13)
    ax.set_title(f'Parallel MST Performance: Borůvka Pooled (Atomic CAS)\n'
                 f'Dataset: {DATASET_NAME} | Threads: {THREAD_COUNTS}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    path1 = f"{FIGURE_DIR}/thread_scaling_time.png"
    plt.savefig(path1, bbox_inches='tight')
    print(f"\nSaved: {path1}")
    plt.close()

    # ────────── Plot 2: Speedup vs Vertices ──────────
    fig, ax = plt.subplots(figsize=(12, 7))

    # Build sequential baselines
    seq_baselines = {}
    for r in all_results:
        if r['algo'] == 'boruvka_seq':
            seq_baselines[(r['lang'], r['n_vertices'])] = r['median_s']

    for (lang, algo, threads), records in sorted(groups.items()):
        if algo == 'boruvka_seq':
            continue

        records.sort(key=lambda r: r['n_vertices'])
        xs, ys = [], []
        for r in records:
            baseline = seq_baselines.get((r['lang'], r['n_vertices']))
            if baseline:
                xs.append(r['n_vertices'])
                ys.append(baseline / r['median_s'])

        if not xs:
            continue

        if lang == 'Rust':
            label = f"Rust (T={threads})"
            color = rust_colors.get(threads, '#333')
        else:
            label = f"Python (T={threads})"
            color = python_colors.get(threads, '#666')

        marker = markers.get(threads, 'o')
        ax.plot(xs, ys, marker=marker, color=color, linestyle='-',
                linewidth=2, markersize=7, label=label)

        # Annotate last point
        ax.annotate(f'{ys[-1]:.2f}×', (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color, fontweight='bold')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(min(r['n_vertices'] for r in all_results), 1.02,
            'Sequential baseline (1.0×)', fontsize=8, color='gray', style='italic')

    ax.set_xlabel('Number of Vertices (V)', fontsize=13)
    ax.set_ylabel('Speedup (T_seq / T_parallel)', fontsize=13)
    ax.set_title(f'Parallel Speedup: Borůvka Pooled vs Sequential\n'
                 f'Dataset: {DATASET_NAME} | Threads: {THREAD_COUNTS}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_xscale('log')

    plt.tight_layout()
    path2 = f"{FIGURE_DIR}/thread_scaling_speedup.png"
    plt.savefig(path2, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()

    return path1, path2


if __name__ == "__main__":
    print("=" * 65)
    print("FINAL BENCHMARK: Thread-Scaling Plots for Parallel MST")
    print(f"Dataset: {DATASET} | Threads: {THREAD_COUNTS}")
    print("All thread counts set via CLI args (not environment variables)")
    print("=" * 65)

    t0 = time.time()

    rust_results = run_rust_benchmarks()
    print(f"\nRust: collected {len(rust_results)} data points")

    python_results = run_python_benchmarks()
    print(f"\nPython: collected {len(python_results)} data points")

    all_results = rust_results + python_results

    # Save raw data
    csv_path = f"{OUTPUT_DIR}/all_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lang', 'algo', 'threads',
                                                'n_vertices', 'n_edges', 'median_s'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw data: {csv_path}")

    p1, p2 = generate_plots(all_results)

    elapsed = time.time() - t0
    print(f"\nTotal benchmark time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
