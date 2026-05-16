#!/usr/bin/env python3.12
"""
Generate publication-quality annotated plots for CS611 MST project.
All output files prefixed with  for identification.
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
import argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', default=None, help='Run ID (timestamp dir under logs/). Defaults to latest.')
parser.add_argument('--no-errorbars', action='store_true', help='Disable error bars (min/max range) on all plots.')
args = parser.parse_args()
SHOW_ERRORBARS = not args.no_errorbars

# Find run directory
if args.run_id:
    RUN_DIR = os.path.join('logs', args.run_id)
else:
    # Auto-discover latest run directory
    run_dirs = sorted(glob.glob('logs/????-??-??_??-??-??'))
    if run_dirs:
        RUN_DIR = run_dirs[-1]
        print(f"Auto-selected latest run: {RUN_DIR}")
    else:
        # Fallback: use legacy flat layout
        RUN_DIR = None
        print("No timestamped run dirs found, using legacy flat layout")

# Set up figure output directories
if RUN_DIR:
    FIGURES_DIR = os.path.join(RUN_DIR, 'figures')
else:
    FIGURES_DIR = 'logs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
REPORT_FIGURES_DIR = 'report/figures'
os.makedirs(REPORT_FIGURES_DIR, exist_ok=True)

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

ALGO_COLORS  = {
    'kruskal': '#1565C0', 'boruvka_seq': '#E65100', 'boruvka_par': '#2E7D32',
    'boruvka_seq_nc': '#FF6F00', 'boruvka_par_nc': '#66BB6A',
    'boruvka_par_fr': '#C62828',
    'boruvka_pooled': '#6A1B9A', 'boruvka_groups': '#00838F',
}
ALGO_MARKERS = {
    'kruskal': 'o', 'boruvka_seq': 's', 'boruvka_par': '^',
    'boruvka_seq_nc': 'D', 'boruvka_par_nc': 'v',
    'boruvka_par_fr': 'h',
    'boruvka_pooled': 'P', 'boruvka_groups': 'X',
}
ALGO_LINESTYLES = {
    'kruskal': '-',
    'boruvka_seq': '-',
    'boruvka_par': '-',
    'boruvka_seq_nc': '--',
    'boruvka_par_nc': '--',
    'boruvka_par_fr': '-.',
    'boruvka_pooled': '-.',
    'boruvka_groups': ':',
}
ALGO_LABELS  = {
    'kruskal': 'Kruskal (Seq)',
    'boruvka_seq': 'Borůvka (Seq)',
    'boruvka_seq_nc': 'Borůvka (Seq, NC)',
    'boruvka_par': 'Borůvka (Par)',
    'boruvka_par_nc': 'Borůvka (Par, NC)',
    'boruvka_par_fr': 'Borůvka (Par, FR)',
    'boruvka_pooled': 'Borůvka (Pooled)',
    'boruvka_groups': 'Borůvka (Groups)',
}

# ============================================================
# Data loading
# ============================================================
def find_csv(lang, dataset, filename):
    """Find CSV in logs/<RUN_ID>/<lang>/<dataset>/<filename> or legacy path."""
    if RUN_DIR:
        path = os.path.join(RUN_DIR, lang, dataset, filename)
        if os.path.exists(path):
            return path
    # Fallback: legacy flat layout
    legacy = os.path.join('logs', lang, dataset, filename)
    if os.path.exists(legacy):
        return legacy
    # Return the expected path (will trigger warning)
    return os.path.join(RUN_DIR or 'logs', lang, dataset, filename)

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}")
        return rows
    print(f"  Loading: {path}")
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
    """Group by key_field, compute mean/min/max of time_s."""
    groups = defaultdict(list)
    for r in rows:
        if filter_fn and not filter_fn(r):
            continue
        groups[r[key_field]].append(r['time_s'])
    keys = sorted(groups.keys())
    means = [np.mean(groups[k]) for k in keys]
    mins = [np.min(groups[k]) for k in keys]
    maxs = [np.max(groups[k]) for k in keys]
    return keys, means, mins, maxs

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

def save_fig(fig, filename):
    """Save figure to run dir and report/figures/."""
    for d in [FIGURES_DIR, REPORT_FIGURES_DIR]:
        path = os.path.join(d, filename)
        fig.savefig(path)
    print(f"  Saved: {filename} → {FIGURES_DIR}/ + {REPORT_FIGURES_DIR}/")

# ── Load everything ──
py_road    = load_csv(find_csv('python', 'roadNet-CA', 'scalability_roadNet-CA.csv'))
py_amz     = load_csv(find_csv('python', 'amazon0302', 'scalability_amazon0302.csv'))
py_orkut   = load_csv(find_csv('python', 'com-orkut', 'scalability_com-orkut.ungraph.csv'))
py_road_sp = load_csv(find_csv('python', 'roadNet-CA', 'speedup_roadNet-CA.csv'))
py_amz_sp  = load_csv(find_csv('python', 'amazon0302', 'speedup_amazon0302.csv'))
py_orkut_sp = load_csv(find_csv('python', 'com-orkut', 'speedup_com-orkut.ungraph.csv'))

rs_road    = load_csv(find_csv('rust', 'roadNet-CA', 'scalability_roadNet-CA.csv'))
rs_amz     = load_csv(find_csv('rust', 'amazon0302', 'scalability_amazon0302.csv'))
rs_orkut   = load_csv(find_csv('rust', 'com-orkut', 'scalability_com-orkut.ungraph.csv'))
rs_road_sp = load_csv(find_csv('rust', 'roadNet-CA', 'speedup_roadNet-CA.csv'))
rs_amz_sp  = load_csv(find_csv('rust', 'amazon0302', 'speedup_amazon0302.csv'))
rs_orkut_sp = load_csv(find_csv('rust', 'com-orkut', 'speedup_com-orkut.ungraph.csv'))

cpp_road = load_csv(find_csv('cpp', 'roadNet-CA', 'scalability_roadNet-CA.csv'))
cpp_amz  = load_csv(find_csv('cpp', 'amazon0302', 'scalability_amazon0302.csv'))
cpp_orkut = load_csv(find_csv('cpp', 'com-orkut', 'scalability_com-orkut.ungraph.csv'))

# C++ speedup data: extract boruvka_par + boruvka_seq from scalability CSVs
# (the thread sweep writes to the same scalability CSV with varying thread counts)
cpp_road_sp = [r for r in cpp_road if r['algorithm'] in ('boruvka_par', 'boruvka_seq')]
cpp_amz_sp  = [r for r in cpp_amz  if r['algorithm'] in ('boruvka_par', 'boruvka_seq')]
cpp_orkut_sp = [r for r in cpp_orkut if r['algorithm'] in ('boruvka_par', 'boruvka_seq')]

val_road = load_validation(find_csv('python', 'roadNet-CA', 'validation_roadNet-CA.csv'))
val_amz  = load_validation(find_csv('python', 'amazon0302', 'validation_amazon0302.csv'))

print(f"Loaded data: {len(py_road)+len(py_amz)+len(py_orkut)} Python rows, "
      f"{len(rs_road)+len(rs_amz)+len(rs_orkut)} Rust rows, "
      f"{len(cpp_road)+len(cpp_amz)+len(cpp_orkut)} C++ rows")

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
    # Plot all algorithms present in the data
    algos_in_data = sorted(set(r['algorithm'] for r in data))
    # Order: contraction variants first, then NC variants
    algo_order = ['kruskal', 'boruvka_seq', 'boruvka_par', 'boruvka_pooled', 'boruvka_groups',
                  'boruvka_seq_nc', 'boruvka_par_nc', 'boruvka_par_fr']
    algos_to_plot = [a for a in algo_order if a in algos_in_data]

    for algo in algos_to_plot:
        recs = [r for r in data if r['algorithm'] == algo]
        if not recs:
            continue
        sizes, meds, mins, maxs = grouped_stats(recs, 'n_vertices')
        x = [s/1000 for s in sizes]
        yerr_lo = [m - mi for m, mi in zip(meds, mins)]
        yerr_hi = [mx - m for m, mx in zip(meds, maxs)]
        color = ALGO_COLORS.get(algo, '#888')
        marker = ALGO_MARKERS.get(algo, 'o')
        label = ALGO_LABELS.get(algo, algo)
        ls = '--' if '_nc' in algo else '-'  # dashed for no-contraction
        if SHOW_ERRORBARS:
            ax.errorbar(x, meds, yerr=[yerr_lo, yerr_hi],
                        fmt=f'{marker}{ls}', color=color,
                        capsize=3, capthick=1, label=f'{label} ({lang_label})')
        else:
            ax.plot(x, meds, f'{marker}{ls}', color=color,
                    label=f'{label} ({lang_label})')
        annotate_last(ax, x, meds)
        for r in recs:
            max_edges = max(max_edges, r['n_edges'])

    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    add_info_box(ax, [
        dataset_desc,
        f'Max edges: {max_edges:,}',
        f'Runs: 5 · Metric: average',
        f'Error bars: min–max range'
    ], loc='upper left')
    add_hw_footer(fig, f'{lang_label} · 5 runs per point')

    save_fig(fig, filename)
    plt.close()

# ============================================================
# 3-4. Combined Scalability per Dataset (all languages on one plot)
# ============================================================
LANG_COLORS = {
    'Python': '#1565C0',   # blue
    'Rust':   '#E65100',   # orange
    'C++':    '#2E7D32',   # green
}

def plot_combined_scalability(datasets, title, filename, dataset_desc):
    """Plot all languages on one figure for a single dataset.
    Color = language, marker + linestyle = algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    max_edges = 0

    for lang_label, data, _ls, lw, alpha in datasets:
        if not data:
            continue
        lang_color = LANG_COLORS.get(lang_label, '#888')
        algos = sorted(set(r['algorithm'] for r in data))
        for algo in algos:
            recs = [r for r in data if r['algorithm'] == algo]
            sizes, meds, yerr_lo, yerr_hi = grouped_stats(recs, 'n_vertices')
            if not sizes:
                continue
            x = [s / 1000 for s in sizes]
            marker = ALGO_MARKERS.get(algo, 'o')
            ls = ALGO_LINESTYLES.get(algo, '-')
            algo_label = ALGO_LABELS.get(algo, algo)
            if SHOW_ERRORBARS:
                ax.errorbar(x, meds, yerr=[yerr_lo, yerr_hi],
                            fmt=f'{marker}', ls=ls, color=lang_color,
                            lw=lw, ms=5, alpha=alpha,
                            capsize=3, capthick=1,
                            label=f'{algo_label} ({lang_label})')
            else:
                ax.plot(x, meds, marker=marker, ls=ls, color=lang_color,
                        lw=lw, ms=5, alpha=alpha,
                        label=f'{algo_label} ({lang_label})')
            for r in recs:
                max_edges = max(max_edges, r['n_edges'])

    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(fontsize=6.5, loc='upper left', ncol=2)

    add_info_box(ax, [
        dataset_desc,
        f'Max edges: {max_edges:,}',
        'Color = language · Shape = algorithm',
        'Dashed = no contraction',
    ], loc='center left')
    add_hw_footer(fig, 'average of 3 runs')

    plt.tight_layout()
    save_fig(fig, filename)
    plt.close()

plot_combined_scalability(
    [('Python', py_road, '--', 1.6, 0.7),
     ('Rust',   rs_road, '-',  2.2, 1.0),
     ('C++',    cpp_road, ':', 1.8, 0.8)],
    'roadNet-CA — Scalability (Python vs Rust vs C++)',
    'scalability_roadNet-CA.png',
    'roadNet-CA · sparse planar · avg deg 3.1')

plot_combined_scalability(
    [('Python', py_amz, '--', 1.6, 0.7),
     ('Rust',   rs_amz, '-',  2.2, 1.0),
     ('C++',    cpp_amz, ':', 1.8, 0.8)],
    'amazon0302 — Scalability (Python vs Rust vs C++)',
    'scalability_amazon0302.png',
    'amazon0302 · power-law · avg deg 6.1')

plot_combined_scalability(
    [('Python', py_orkut, '--', 1.6, 0.7),
     ('Rust',   rs_orkut, '-',  2.2, 1.0),
     ('C++',    cpp_orkut, ':', 1.8, 0.8)],
    'com-Orkut — Scalability (Python vs Rust vs C++)',
    'scalability_com-orkut.png',
    'com-Orkut · social network · avg deg 76.3')

# ============================================================
# 5. Combined Parallel Speedup (Python + Rust + C++ on one figure)
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

speedup_datasets = [
    (py_road_sp, 'roadNet-CA (Python)', 273266),
    (py_amz_sp,  'amazon0302 (Python)', 899792),
    (py_orkut_sp, 'com-Orkut (Python)', 117185083),
    (rs_road_sp, 'roadNet-CA (Rust)',   273266),
    (rs_amz_sp,  'amazon0302 (Rust)',   899792),
    (rs_orkut_sp, 'com-Orkut (Rust)',   117185083),
    (cpp_road_sp, 'roadNet-CA (C++)',   273266),
    (cpp_amz_sp,  'amazon0302 (C++)',   899792),
    (cpp_orkut_sp, 'com-Orkut (C++)',   117185083),
]

for ax, (data, ds_label, ds_edges) in zip(axes.flat, speedup_datasets):
    if not data:
        ax.set_title(f'{ds_label} — no data')
        continue
    # For C++ speedup data, compute from boruvka_seq baseline
    par_recs = [r for r in data if r['algorithm'] == 'boruvka_par']
    seq_recs = [r for r in data if r['algorithm'] == 'boruvka_seq']
    # Use par_recs if available, else fall back to all data
    sp_data = par_recs if par_recs else data

    sizes_set = sorted(set(r['n_vertices'] for r in sp_data))
    cmap = plt.cm.Dark2
    for idx, sz in enumerate(sizes_set):
        recs = [r for r in sp_data if r['n_vertices'] == sz]
        thread_set = sorted(set(r['threads'] for r in recs))

        if 'seq_baseline' in recs[0] and recs[0]['seq_baseline']:
            seq_base = float(recs[0]['seq_baseline'])
        else:
            # Use boruvka_seq time at this size as baseline
            seq_times = [r['time_s'] for r in seq_recs if r['n_vertices'] == sz]
            if seq_times:
                seq_base = np.mean(seq_times)
            else:
                t1_times = [r['time_s'] for r in recs if r['threads'] == 1]
                seq_base = np.mean(t1_times) if t1_times else np.mean([r['time_s'] for r in recs])

        speedups = []
        sp_mins = []
        sp_maxs = []
        for tc in thread_set:
            times = [r['time_s'] for r in recs if r['threads'] == tc]
            avg = np.mean(times)
            speedups.append(seq_base / avg)
            sp_mins.append(seq_base / np.max(times))
            sp_maxs.append(seq_base / np.min(times))

        yerr_lo = [s - lo for s, lo in zip(speedups, sp_mins)]
        yerr_hi = [hi - s for s, hi in zip(speedups, sp_maxs)]
        color = cmap(idx / max(len(sizes_set)-1, 1))
        if SHOW_ERRORBARS:
            ax.errorbar(thread_set, speedups, yerr=[yerr_lo, yerr_hi],
                        fmt='o-', color=color, capsize=3, capthick=1,
                        label=f'V={sz//1000}K')
        else:
            ax.plot(thread_set, speedups, 'o-', color=color,
                    label=f'V={sz//1000}K')

        peak_idx = np.argmax(speedups)
        ax.annotate(f'{speedups[peak_idx]:.2f}×',
                    (thread_set[peak_idx], speedups[peak_idx]),
                    textcoords='offset points', xytext=(5, 8),
                    fontsize=7, fontweight='bold', color=color)

    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.4, lw=1)
    ax.axvline(x=PHYS_CORES, color='#E91E63', ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(ds_label, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

plt.suptitle('Parallel Speedup — Python (Numba) vs Rust (Rayon) vs C++ (std::thread)',
             fontsize=14, fontweight='bold', y=1.01)
add_hw_footer(fig, 'Borůvka-Par vs Borůvka-Seq baseline')
plt.tight_layout()
save_fig(fig, 'parallel_speedup.png')
plt.close()

# ============================================================
# 7. Cross-Language Comparison (Python vs Rust vs C++ overlay)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (py_data, rs_data, cpp_data, title, ds_edges) in zip(axes, [
    (py_road, rs_road, cpp_road, 'roadNet-CA', 273266),
    (py_amz, rs_amz, cpp_amz, 'amazon0302', 899792),
    (py_orkut, rs_orkut, cpp_orkut, 'com-Orkut', 117185083),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_recs = [r for r in py_data if r['algorithm'] == algo]
        rs_recs = [r for r in rs_data if r['algorithm'] == algo]
        cpp_recs = [r for r in cpp_data if r['algorithm'] == algo]
        marker = ALGO_MARKERS.get(algo, 'o')
        ls = ALGO_LINESTYLES.get(algo, '-')
        algo_label = ALGO_LABELS.get(algo, algo)

        py_sizes, py_meds, _, _ = grouped_stats(py_recs, 'n_vertices')
        rs_sizes, rs_meds, _, _ = grouped_stats(rs_recs, 'n_vertices')

        ax.plot([s/1000 for s in py_sizes], py_meds, marker=marker, ls=ls,
                color=LANG_COLORS['Python'], lw=1.6, ms=5, alpha=0.65,
                label=f'{algo_label} (Python)')
        ax.plot([s/1000 for s in rs_sizes], rs_meds, marker=marker, ls=ls,
                color=LANG_COLORS['Rust'], lw=2.2, ms=7,
                label=f'{algo_label} (Rust)')

        if cpp_recs:
            cpp_sizes, cpp_meds, _, _ = grouped_stats(cpp_recs, 'n_vertices')
            ax.plot([s/1000 for s in cpp_sizes], cpp_meds, marker=marker, ls=ls,
                    color=LANG_COLORS['C++'], lw=1.8, ms=5, alpha=0.8,
                    label=f'{algo_label} (C++)')

    # Add C++ boruvka_par if present
    cpp_par_recs = [r for r in cpp_data if r['algorithm'] == 'boruvka_par']
    if cpp_par_recs:
        cpp_sizes, cpp_meds, _, _ = grouped_stats(cpp_par_recs, 'n_vertices')
        par_marker = ALGO_MARKERS.get('boruvka_par', '^')
        par_ls = ALGO_LINESTYLES.get('boruvka_par', '-')
        ax.plot([s/1000 for s in cpp_sizes], cpp_meds, marker=par_marker, ls=par_ls,
                color=LANG_COLORS['C++'], lw=1.8, ms=5, alpha=0.8,
                label=f'{ALGO_LABELS["boruvka_par"]} (C++)')

    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=6.5)

    add_info_box(ax, [
        f'Edges at max: {ds_edges:,}',
        'Color = language · Shape = algorithm',
    ], loc='upper left')

plt.suptitle('Python (Numba) vs Rust (Rayon) vs C++', fontsize=14, fontweight='bold', y=1.01)
add_hw_footer(fig, 'average of runs')
plt.tight_layout()
save_fig(fig, 'python_vs_rust_comparison.png')
plt.close()

# ============================================================
# 8. Rust/Python Speedup Ratio
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (py_data, rs_data, title) in zip(axes, [
    (py_road, rs_road, 'roadNet-CA'),
    (py_amz, rs_amz, 'amazon0302'),
    (py_orkut, rs_orkut, 'com-Orkut'),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_sizes, py_meds, _, _ = grouped_stats(
            [r for r in py_data if r['algorithm'] == algo], 'n_vertices')
        rs_sizes, rs_meds, _, _ = grouped_stats(
            [r for r in rs_data if r['algorithm'] == algo], 'n_vertices')

        common = sorted(set(py_sizes) & set(rs_sizes))
        if not common:
            continue
        py_map = dict(zip(py_sizes, py_meds))
        rs_map = dict(zip(rs_sizes, rs_meds))
        ratios = [py_map[s] / rs_map[s] for s in common if rs_map[s] > 0]

        if not ratios:
            continue
        marker = ALGO_MARKERS.get(algo, 'o')
        ls = ALGO_LINESTYLES.get(algo, '-')
        ax.plot([s/1000 for s in common[:len(ratios)]], ratios,
                marker=marker, ls=ls, color=ALGO_COLORS.get(algo, '#888'),
                lw=2.2, ms=7, label=ALGO_LABELS[algo])

        mean_ratio = np.mean(ratios)
        ax.annotate(f'avg {mean_ratio:.1f}×', (common[len(ratios)-1]/1000, ratios[-1]),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=7.5, color=ALGO_COLORS.get(algo, '#888'), fontweight='bold')

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
save_fig(fig, 'rust_over_python_ratio.png')
plt.close()


# ============================================================
# 9. Parallel Efficiency Plot (Python + Rust + C++)
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

plot_configs = [
    (py_road_sp, 'Python/Numba — roadNet-CA', 0, 0),
    (py_amz_sp, 'Python/Numba — amazon0302', 0, 1),
    (py_orkut_sp, 'Python/Numba — com-Orkut', 0, 2),
    (rs_road_sp, 'Rust/Rayon — roadNet-CA', 1, 0),
    (rs_amz_sp, 'Rust/Rayon — amazon0302', 1, 1),
    (rs_orkut_sp, 'Rust/Rayon — com-Orkut', 1, 2),
    (cpp_road_sp, 'C++/std::thread — roadNet-CA', 2, 0),
    (cpp_amz_sp, 'C++/std::thread — amazon0302', 2, 1),
    (cpp_orkut_sp, 'C++/std::thread — com-Orkut', 2, 2),
]

for data, title, row, col in plot_configs:
    ax = axes[row][col]
    if not data:
        ax.set_title(f'{title} — no data')
        continue
    par_recs = [r for r in data if r['algorithm'] == 'boruvka_par']
    seq_recs = [r for r in data if r['algorithm'] == 'boruvka_seq']
    sp_data = par_recs if par_recs else data

    sizes_set = sorted(set(r['n_vertices'] for r in sp_data))
    cmap = plt.cm.Dark2

    for idx, sz in enumerate(sizes_set):
        recs = [r for r in sp_data if r['n_vertices'] == sz]
        thread_set = sorted(set(r['threads'] for r in recs))

        # Determine baseline: use seq_baseline if available, else boruvka_seq, else 1-thread
        if 'seq_baseline' in recs[0] and recs[0]['seq_baseline']:
            seq_base = float(recs[0]['seq_baseline'])
        else:
            seq_times = [r['time_s'] for r in seq_recs if r['n_vertices'] == sz]
            if seq_times:
                seq_base = np.mean(seq_times)
            else:
                t1_times = [r['time_s'] for r in recs if r['threads'] == 1]
                seq_base = np.mean(t1_times) if t1_times else np.mean([r['time_s'] for r in recs])

        efficiencies = []
        for tc in thread_set:
            times = [r['time_s'] for r in recs if r['threads'] == tc]
            speedup = seq_base / np.mean(times)
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
save_fig(fig, 'parallel_efficiency.png')
plt.close()

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
    save_fig(fig, 'validation_summary.png')
    plt.close()

# ============================================================
# Benchmark Summary Table
# ============================================================
def print_summary_table():
    """Print ASCII art summary tables and save LaTeX file."""
    all_datasets = [
        ('roadNet-CA', {'Python': py_road, 'Rust': rs_road, 'C++': cpp_road}),
        ('amazon0302', {'Python': py_amz, 'Rust': rs_amz, 'C++': cpp_amz}),
        ('com-Orkut',  {'Python': py_orkut, 'Rust': rs_orkut, 'C++': cpp_orkut}),
    ]
    algos_order = ['kruskal', 'boruvka_seq', 'boruvka_seq_nc', 'boruvka_par', 'boruvka_par_nc']
    langs_order = ['Python', 'Rust', 'C++']

    # Build rows matching the plot data points
    plot_rows = []
    for ds_name, lang_data in all_datasets:
        for lang in langs_order:
            data = lang_data.get(lang, [])
            if not data:
                continue
            for algo in algos_order:
                recs = [r for r in data if r['algorithm'] == algo]
                if not recs:
                    continue
                for sz in sorted(set(r['n_vertices'] for r in recs)):
                    sz_recs = [r for r in recs if r['n_vertices'] == sz]
                    times = [r['time_s'] for r in sz_recs]
                    plot_rows.append({
                        'dataset': ds_name, 'algorithm': algo, 'language': lang,
                        'vertices': sz, 'edges': sz_recs[0]['n_edges'],
                        'threads': sz_recs[0].get('threads', 1),
                        'runs': len(times),
                        'avg': float(np.mean(times)),
                        'std': float(np.std(times)),
                        'min': float(np.min(times)),
                        'max': float(np.max(times)),
                    })

    if not plot_rows:
        print("\n  No benchmark data found — skipping summary table.")
        return

    # ASCII helpers
    def sep(ws):
        return '+' + '+'.join('-' * (w + 2) for w in ws) + '+'
    def arow(vals, ws):
        cells = []
        for i, (v, w) in enumerate(zip(vals, ws)):
            cells.append(f' {v:<{w}} ' if i < 3 else f' {v:>{w}} ')
        return '|' + '|'.join(cells) + '|'

    # ── Table 1: Detailed scalability data ──
    W = [12, 18, 6, 10, 11, 4, 4, 10, 9, 10, 10]
    headers = ['Dataset', 'Algorithm', 'Lang', 'Vertices', 'Edges', 'Thr', 'Runs',
               'Avg (s)', 'Std (s)', 'Min (s)', 'Max (s)']
    print(f"\n  SCALABILITY DATA (same data points as plots, average of N runs)\n")
    print(sep(W))
    print(arow(headers, W))
    print(sep(W))
    prev_ds = None
    for r in plot_rows:
        if r['dataset'] != prev_ds and prev_ds is not None:
            print(sep(W))
        ds_col = r['dataset'] if r['dataset'] != prev_ds else ""
        prev_ds = r['dataset']
        al = ALGO_LABELS.get(r['algorithm'], r['algorithm'])
        print(arow([ds_col, al, r['language'],
                     f"{r['vertices']:,}", f"{r['edges']:,}",
                     str(r['threads']), str(r['runs']),
                     f"{r['avg']:.6f}", f"{r['std']:.6f}",
                     f"{r['min']:.6f}", f"{r['max']:.6f}"], W))
    print(sep(W))

    # ── Table 2: Cross-language comparison at max size ──
    W2 = [12, 18, 10, 11, 12, 12, 12, 8]
    headers2 = ['Dataset', 'Algorithm', 'Vertices', 'Edges',
                'Python (s)', 'Rust (s)', 'C++ (s)', 'Fastest']
    print(f"\n  CROSS-LANGUAGE COMPARISON (avg time at max graph size, * = fastest)\n")
    print(sep(W2))
    print(arow(headers2, W2))
    print(sep(W2))
    prev_ds2 = None
    for ds_name, lang_data in all_datasets:
        if prev_ds2 is not None:
            print(sep(W2))
        prev_ds2 = ds_name
        for algo in algos_order:
            tbl, meta = {}, {}
            for lang in langs_order:
                data = lang_data.get(lang, [])
                recs = [r for r in data if r['algorithm'] == algo]
                if not recs:
                    continue
                max_v = max(r['n_vertices'] for r in recs)
                tbl[lang] = float(np.mean([r['time_s'] for r in recs if r['n_vertices'] == max_v]))
                meta['vertices'] = max_v
                meta['edges'] = [r for r in recs if r['n_vertices'] == max_v][0]['n_edges']
            if not tbl:
                continue
            fastest = min(tbl, key=tbl.get)
            py_s = f"{tbl['Python']:.4f}" if 'Python' in tbl else "—"
            rs_s = f"{tbl['Rust']:.4f}" if 'Rust' in tbl else "—"
            cpp_s = f"{tbl['C++']:.4f}" if 'C++' in tbl else "—"
            if fastest == 'Python': py_s = f"*{py_s}"
            elif fastest == 'Rust': rs_s = f"*{rs_s}"
            elif fastest == 'C++': cpp_s = f"*{cpp_s}"
            al = ALGO_LABELS.get(algo, algo)
            print(arow([ds_name, al,
                         f"{meta['vertices']:,}", f"{meta['edges']:,}",
                         py_s, rs_s, cpp_s, fastest], W2))
    print(sep(W2))

    # ── Save LaTeX file (silent) ──
    latex_path = os.path.join(FIGURES_DIR, 'benchmark_summary.tex')
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated by generate_plots.py\n")
        f.write("% Requires: \\usepackage{booktabs, float}\n\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Cross-language performance comparison — average execution time (seconds) "
                "at maximum graph size. \\textbf{Bold} = fastest.}\n")
        f.write("\\label{tab:benchmark_summary}\n\\small\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Dataset & Algorithm & Vertices & Python (s) & Rust (s) & C++ (s) \\\\\n\\midrule\n")
        for ds_name, lang_data in all_datasets:
            first_ds = True
            for algo in algos_order:
                tbl, v_max = {}, 0
                for lang in langs_order:
                    data = lang_data.get(lang, [])
                    recs = [r for r in data if r['algorithm'] == algo]
                    if not recs: continue
                    mv = max(r['n_vertices'] for r in recs)
                    tbl[lang] = float(np.mean([r['time_s'] for r in recs if r['n_vertices'] == mv]))
                    v_max = max(v_max, mv)
                if not tbl: continue
                fastest = min(tbl, key=tbl.get)
                ds_col = ds_name if first_ds else ""
                al = ALGO_LABELS.get(algo, algo)
                def fc(lk):
                    if lk not in tbl: return "—"
                    v = f"{tbl[lk]:.4f}"
                    return f"\\textbf{{{v}}}" if lk == fastest else v
                f.write(f"{ds_col} & {al} & {v_max:,} & {fc('Python')} & {fc('Rust')} & {fc('C++')} \\\\\n")
                first_ds = False
            f.write("\\midrule\n")
        f.seek(f.tell() - len("\\midrule\n"))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n  LaTeX table saved to: {latex_path}")

print_summary_table()

# ============================================================
# Figure file summary
# ============================================================
print(f"\n{'='*65}")
print(f"All figures saved to {FIGURES_DIR}/")
print(f"{'='*65}")
expected_figures = [
    'scalability_roadNet-CA.png',
    'scalability_amazon0302.png',
    'scalability_com-orkut.png',
    'parallel_speedup.png',
    'parallel_efficiency.png',
    'python_vs_rust_comparison.png',
    'rust_over_python_ratio.png',
    'validation_summary.png',
    'benchmark_summary.tex',
]
for f in expected_figures:
    for d in [FIGURES_DIR, REPORT_FIGURES_DIR]:
        path = os.path.join(d, f)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓ {d}/{f} ({size_kb:.0f} KB)")
        else:
            print(f"  ✗ {d}/{f} MISSING")

