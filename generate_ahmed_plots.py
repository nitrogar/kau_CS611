#!/usr/bin/env python3.12
"""
Generate publication-quality annotated plots for CS611 MST project.
All output files prefixed with ahmed_ for identification.
Reads from results/ CSVs, outputs to report/figures/.

Annotations include:
  - Dataset name, vertex/edge count
  - Thread counts on speedup plots
  - Error bars (min/max range)
  - Hardware info (Ryzen 9 3900X, 12C/24T)
  - Data point labels at key sizes
  - Amdahl's law analysis
"""
import os, csv, platform
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Configuration
# ============================================================
FIGURES_DIR = 'report/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

HARDWARE_LABEL = 'AMD Ryzen 9 3900X · 12 cores / 24 threads'
PHYS_CORES = 12
TOTAL_THREADS = 24

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.9,
    'figure.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

ALGO_COLORS  = {'kruskal': '#1565C0', 'boruvka_seq': '#E65100', 'boruvka_par': '#2E7D32'}
ALGO_MARKERS = {'kruskal': 'o', 'boruvka_seq': 's', 'boruvka_par': '^'}
ALGO_LABELS  = {'kruskal': 'Kruskal (Seq)', 'boruvka_seq': 'Borůvka (Seq)', 'boruvka_par': 'Borůvka (Par)'}

# ============================================================
# Data loading
# ============================================================
def load_csv(path):
    rows = []
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}")
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['n_vertices'] = int(r['n_vertices'])
            r['n_edges'] = int(r['n_edges'])
            r['time_s'] = float(r['time_s'])
            r['mst_weight'] = int(r['mst_weight'])
            r['threads'] = int(r.get('threads', 1))
            rows.append(r)
    return rows

def grouped_stats(rows, key_field, filter_fn=None):
    """Group by key_field, compute median/min/max of time_s."""
    groups = defaultdict(list)
    for r in rows:
        if filter_fn and not filter_fn(r):
            continue
        groups[r[key_field]].append(r['time_s'])
    keys = sorted(groups.keys())
    medians = [np.median(groups[k]) for k in keys]
    mins = [np.min(groups[k]) for k in keys]
    maxs = [np.max(groups[k]) for k in keys]
    return keys, medians, mins, maxs

def load_validation(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            r['n_vertices'] = int(r['n_vertices'])
            r['n_edges'] = int(r['n_edges'])
            r['our_weight'] = int(r['our_weight'])
            r['networkx_weight'] = int(r['networkx_weight'])
            r['match'] = r['match'] == 'True'
            rows.append(r)
    return rows

# ── Load everything ──
py_road = load_csv('logs/python/roadNet-CA/scalability_roadNet-CA.csv')
py_amz  = load_csv('logs/python/amazon0302/scalability_amazon0302.csv')
py_road_sp = load_csv('logs/python/roadNet-CA/speedup_roadNet-CA.csv')
py_amz_sp  = load_csv('logs/python/amazon0302/speedup_amazon0302.csv')

rs_road = load_csv('logs/rust/roadNet-CA/scalability_roadNet-CA.csv')
rs_amz  = load_csv('logs/rust/amazon0302/scalability_amazon0302.csv')
rs_road_sp = load_csv('logs/rust/roadNet-CA/speedup_roadNet-CA.csv')
rs_amz_sp  = load_csv('logs/rust/amazon0302/speedup_amazon0302.csv')

val_road = load_validation('logs/python/roadNet-CA/validation_roadNet-CA.csv')
val_amz  = load_validation('logs/python/amazon0302/validation_amazon0302.csv')

print(f"Loaded data: {len(py_road)+len(py_amz)} Python rows, "
      f"{len(rs_road)+len(rs_amz)} Rust rows")

# ============================================================
# Helper: add hardware annotation footer
# ============================================================
def add_hw_footer(fig, extra=''):
    text = HARDWARE_LABEL
    if extra:
        text += f'  ·  {extra}'
    fig.text(0.99, 0.005, text, ha='right', va='bottom',
             fontsize=7, color='#999999', style='italic')

# ============================================================
# Helper: annotate key data points
# ============================================================
def annotate_last(ax, x, y, fmt='{:.4f}s', offset=(5, 5)):
    """Annotate the last data point with its value."""
    if x and y:
        ax.annotate(fmt.format(y[-1]), (x[-1], y[-1]),
                    textcoords='offset points', xytext=offset,
                    fontsize=7, color='#555', ha='left')

def add_info_box(ax, lines, loc='upper left'):
    """Add a text box with dataset info."""
    text = '\n'.join(lines)
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#ccc', alpha=0.9)
    ax.text(0.02 if 'left' in loc else 0.98, 0.98 if 'upper' in loc else 0.02,
            text, transform=ax.transAxes, fontsize=7,
            va='top' if 'upper' in loc else 'bottom',
            ha='left' if 'left' in loc else 'right',
            bbox=props)

# ============================================================
# 1. Python Road Scalability
# ============================================================
def plot_scalability_annotated(data, title, filename, lang_label, dataset_desc):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    max_edges = 0
    for algo in ['kruskal', 'boruvka_seq', 'boruvka_par']:
        recs = [r for r in data if r['algorithm'] == algo]
        if not recs:
            continue
        sizes, meds, mins, maxs = grouped_stats(recs, 'n_vertices')
        x = [s/1000 for s in sizes]
        yerr_lo = [m - mi for m, mi in zip(meds, mins)]
        yerr_hi = [mx - m for m, mx in zip(meds, maxs)]
        ax.errorbar(x, meds, yerr=[yerr_lo, yerr_hi],
                    fmt=f'{ALGO_MARKERS[algo]}-', color=ALGO_COLORS[algo],
                    capsize=3, capthick=1, label=f'{ALGO_LABELS[algo]} ({lang_label})')
        annotate_last(ax, x, meds)
        # Track max edges
        for r in recs:
            max_edges = max(max_edges, r['n_edges'])

    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    add_info_box(ax, [
        dataset_desc,
        f'Max edges: {max_edges:,}',
        f'Runs: 5 · Metric: median',
        f'Error bars: min–max range'
    ], loc='upper left')
    add_hw_footer(fig, f'{lang_label} · 5 runs per point')

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {filename}")

plot_scalability_annotated(py_road,
    'Road Network Scalability — Python (Numba JIT)',
    'ahmed_python_road_scalability.png', 'Python/Numba',
    'roadNet-CA · up to 200K vertices')

plot_scalability_annotated(py_amz,
    'Amazon Co-Purchase Scalability — Python (Numba JIT)',
    'ahmed_python_amazon_scalability.png', 'Python/Numba',
    'amazon0302 · up to 262K vertices')

plot_scalability_annotated(rs_road,
    'Road Network Scalability — Rust (Rayon + SIMD)',
    'ahmed_rust_road_scalability.png', 'Rust/Rayon',
    'roadNet-CA · up to 200K vertices')

plot_scalability_annotated(rs_amz,
    'Amazon Co-Purchase Scalability — Rust (Rayon + SIMD)',
    'ahmed_rust_amazon_scalability.png', 'Rust/Rayon',
    'amazon0302 · up to 262K vertices')

# ============================================================
# 5-6. Parallel Speedup (Python + Rust, side-by-side)
# ============================================================
def plot_speedup_annotated(data_list, suptitle, filename, lang_label):
    """Plot speedup for multiple sizes side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (data, ds_label, ds_edges) in zip(axes, data_list):
        sizes_set = sorted(set(r['n_vertices'] for r in data))
        cmap = plt.cm.Dark2
        for idx, sz in enumerate(sizes_set):
            recs = [r for r in data if r['n_vertices'] == sz]
            thread_set = sorted(set(r['threads'] for r in recs))

            # Determine baseline: use seq_baseline if available (Python CSV),
            # otherwise use 1-thread median (Rust CSV)
            if 'seq_baseline' in recs[0] and recs[0]['seq_baseline']:
                seq_base = float(recs[0]['seq_baseline'])
            else:
                t1_times = [r['time_s'] for r in recs if r['threads'] == 1]
                seq_base = np.median(t1_times) if t1_times else np.median([r['time_s'] for r in recs])

            speedups = []
            sp_mins = []
            sp_maxs = []
            for tc in thread_set:
                times = [r['time_s'] for r in recs if r['threads'] == tc]
                med = np.median(times)
                speedups.append(seq_base / med)
                sp_mins.append(seq_base / np.max(times))  # worst case
                sp_maxs.append(seq_base / np.min(times))  # best case

            yerr_lo = [s - lo for s, lo in zip(speedups, sp_mins)]
            yerr_hi = [hi - s for s, hi in zip(speedups, sp_maxs)]
            color = cmap(idx / max(len(sizes_set)-1, 1))
            ax.errorbar(thread_set, speedups, yerr=[yerr_lo, yerr_hi],
                        fmt='o-', color=color, capsize=3, capthick=1,
                        label=f'V={sz//1000}K')

            # Annotate peak speedup
            peak_idx = np.argmax(speedups)
            ax.annotate(f'{speedups[peak_idx]:.2f}×',
                        (thread_set[peak_idx], speedups[peak_idx]),
                        textcoords='offset points', xytext=(5, 8),
                        fontsize=7, fontweight='bold', color=color)

        # Reference lines
        ax.axhline(y=1.0, color='gray', ls='--', alpha=0.4, lw=1, label='Baseline (1.0×)')
        ax.axvline(x=PHYS_CORES, color='#E91E63', ls=':', alpha=0.4, lw=1)
        ax.text(PHYS_CORES + 0.3, ax.get_ylim()[1] * 0.95, f'{PHYS_CORES} cores',
                fontsize=7, color='#E91E63', alpha=0.6)

        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup (×)')
        ax.set_title(f'{ds_label}')
        ax.legend(fontsize=8, loc='upper left')

        add_info_box(ax, [
            f'Dataset: {ds_label}',
            f'Edges: {ds_edges:,}',
            f'Baseline: Borůvka-Seq',
        ], loc='lower right')

    plt.suptitle(f'Parallel Speedup — {lang_label}', fontsize=14, fontweight='bold', y=1.01)
    add_hw_footer(fig, f'{lang_label} · Borůvka-Par vs Borůvka-Seq')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {filename}")

plot_speedup_annotated(
    [(py_road_sp, 'roadNet-CA', 273266), (py_amz_sp, 'amazon0302', 899792)],
    'Parallel Speedup — Python (Numba prange)',
    'ahmed_python_parallel_speedup.png', 'Python/Numba')

plot_speedup_annotated(
    [(rs_road_sp, 'roadNet-CA', 273266), (rs_amz_sp, 'amazon0302', 899792)],
    'Parallel Speedup — Rust (Rayon)',
    'ahmed_rust_parallel_speedup.png', 'Rust/Rayon')

# ============================================================
# 7. Cross-Language Comparison (Python vs Rust overlay)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (py_data, rs_data, title, ds_edges) in zip(axes, [
    (py_road, rs_road, 'roadNet-CA', 273266),
    (py_amz, rs_amz, 'amazon0302', 899792),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_recs = [r for r in py_data if r['algorithm'] == algo]
        rs_recs = [r for r in rs_data if r['algorithm'] == algo]
        py_sizes, py_meds, _, _ = grouped_stats(py_recs, 'n_vertices')
        rs_sizes, rs_meds, _, _ = grouped_stats(rs_recs, 'n_vertices')

        ax.plot([s/1000 for s in py_sizes], py_meds, f'{ALGO_MARKERS[algo]}--',
                color=ALGO_COLORS[algo], lw=1.6, ms=5, alpha=0.65,
                label=f'{ALGO_LABELS[algo]} (Python)')
        ax.plot([s/1000 for s in rs_sizes], rs_meds, f'{ALGO_MARKERS[algo]}-',
                color=ALGO_COLORS[algo], lw=2.2, ms=7,
                label=f'{ALGO_LABELS[algo]} (Rust)')

    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=7.5)

    add_info_box(ax, [
        f'Edges at max: {ds_edges:,}',
        f'Dashed = Python, Solid = Rust',
    ], loc='upper left')

plt.suptitle('Python (Numba JIT) vs Rust (Native)', fontsize=14, fontweight='bold', y=1.01)
add_hw_footer(fig, '5 runs · median')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ahmed_python_vs_rust_comparison.png'))
plt.close()
print("  Saved: ahmed_python_vs_rust_comparison.png")

# ============================================================
# 8. Rust/Python Speedup Ratio
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (py_data, rs_data, title) in zip(axes, [
    (py_road, rs_road, 'roadNet-CA'),
    (py_amz, rs_amz, 'amazon0302'),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_sizes, py_meds, _, _ = grouped_stats(
            [r for r in py_data if r['algorithm'] == algo], 'n_vertices')
        rs_sizes, rs_meds, _, _ = grouped_stats(
            [r for r in rs_data if r['algorithm'] == algo], 'n_vertices')

        common = sorted(set(py_sizes) & set(rs_sizes))
        py_map = dict(zip(py_sizes, py_meds))
        rs_map = dict(zip(rs_sizes, rs_meds))
        ratios = [py_map[s] / rs_map[s] for s in common]

        ax.plot([s/1000 for s in common], ratios, f'{ALGO_MARKERS[algo]}-',
                color=ALGO_COLORS[algo], lw=2.2, ms=7, label=ALGO_LABELS[algo])

        # Annotate mean ratio
        mean_ratio = np.mean(ratios)
        ax.annotate(f'avg {mean_ratio:.1f}×', (common[-1]/1000, ratios[-1]),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=7.5, color=ALGO_COLORS[algo], fontweight='bold')

    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Speedup (Python time / Rust time)')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    add_info_box(ax, [
        'Ratio > 1.0 → Rust faster',
        'Ratio = 1.0 → Same speed',
    ], loc='lower right')

plt.suptitle('Rust vs Python Speedup Ratio', fontsize=14, fontweight='bold', y=1.01)
add_hw_footer(fig, 'Kruskal: sort-dominated · Borůvka: compute-dominated')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ahmed_rust_over_python_ratio.png'))
plt.close()
print("  Saved: ahmed_rust_over_python_ratio.png")

# ============================================================
# 9. NEW: Parallel Efficiency Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_configs = [
    (py_road_sp, 'Python/Numba — roadNet-CA', 0, 0),
    (py_amz_sp, 'Python/Numba — amazon0302', 0, 1),
    (rs_road_sp, 'Rust/Rayon — roadNet-CA', 1, 0),
    (rs_amz_sp, 'Rust/Rayon — amazon0302', 1, 1),
]

for data, title, row, col in plot_configs:
    ax = axes[row][col]
    sizes_set = sorted(set(r['n_vertices'] for r in data))
    cmap = plt.cm.Dark2

    for idx, sz in enumerate(sizes_set):
        recs = [r for r in data if r['n_vertices'] == sz]
        thread_set = sorted(set(r['threads'] for r in recs))

        # Determine baseline: use seq_baseline if available, else 1-thread median
        if 'seq_baseline' in recs[0] and recs[0]['seq_baseline']:
            seq_base = float(recs[0]['seq_baseline'])
        else:
            t1_times = [r['time_s'] for r in recs if r['threads'] == 1]
            seq_base = np.median(t1_times) if t1_times else np.median([r['time_s'] for r in recs])

        efficiencies = []
        for tc in thread_set:
            times = [r['time_s'] for r in recs if r['threads'] == tc]
            speedup = seq_base / np.median(times)
            efficiencies.append(speedup / tc * 100)

        color = cmap(idx / max(len(sizes_set)-1, 1))
        ax.plot(thread_set, efficiencies, 'o-', color=color,
                label=f'V={sz//1000}K')

    ax.axhline(y=100, color='gray', ls='--', alpha=0.4, lw=1, label='Ideal (100%)')
    ax.axvline(x=PHYS_CORES, color='#E91E63', ls=':', alpha=0.3)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_ylim(bottom=0)

plt.suptitle('Parallel Efficiency = Speedup / Threads × 100%',
             fontsize=14, fontweight='bold', y=1.01)
add_hw_footer(fig, 'Borůvka-Par · Pink line = 12 physical cores')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ahmed_parallel_efficiency.png'))
plt.close()
print("  Saved: ahmed_parallel_efficiency.png")

# ============================================================
# 10. NEW: Validation Summary Table
# ============================================================
all_val = val_road + val_amz
if all_val:
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(all_val) + 1.5)))
    ax.axis('off')

    headers = ['Dataset', 'Vertices', 'Edges', 'Our MST Weight', 'NetworkX Weight', 'Status']
    table_data = []
    for v in all_val:
        status = '✓ MATCH' if v['match'] else '✗ MISMATCH'
        table_data.append([
            v['dataset'],
            f"{v['n_vertices']:,}",
            f"{v['n_edges']:,}",
            f"{v['our_weight']:,}",
            f"{v['networkx_weight']:,}",
            status,
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j, key in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')

    # Color status cells
    for i, row in enumerate(table_data):
        cell = table[i + 1, 5]
        if '✓' in row[5]:
            cell.set_facecolor('#E8F5E9')
            cell.set_text_props(color='#2E7D32', fontweight='bold')
        else:
            cell.set_facecolor('#FFEBEE')
            cell.set_text_props(color='#C62828', fontweight='bold')

    ax.set_title('NetworkX Third-Party Validation Summary',
                 fontsize=14, fontweight='bold', pad=20)

    all_match = all(v['match'] for v in all_val)
    summary = f"ALL {len(all_val)} TESTS PASSED ✓" if all_match else "SOME TESTS FAILED ✗"
    summary_color = '#2E7D32' if all_match else '#C62828'
    fig.text(0.5, 0.02, summary, ha='center', fontsize=12,
             fontweight='bold', color=summary_color)
    add_hw_footer(fig, 'Kruskal vs Borůvka vs Borůvka-Par vs NetworkX')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'ahmed_validation_summary.png'))
    plt.close()
    print("  Saved: ahmed_validation_summary.png")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*65}")
print(f"All annotated figures saved to {FIGURES_DIR}/")
print(f"{'='*65}")

figures = [
    'ahmed_python_road_scalability.png',
    'ahmed_python_amazon_scalability.png',
    'ahmed_rust_road_scalability.png',
    'ahmed_rust_amazon_scalability.png',
    'ahmed_python_parallel_speedup.png',
    'ahmed_rust_parallel_speedup.png',
    'ahmed_python_vs_rust_comparison.png',
    'ahmed_rust_over_python_ratio.png',
    'ahmed_parallel_efficiency.png',
    'ahmed_validation_summary.png',
]
for f in figures:
    path = os.path.join(FIGURES_DIR, f)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓ {f} ({size_kb:.0f} KB)")
    else:
        print(f"  ✗ {f} MISSING")
