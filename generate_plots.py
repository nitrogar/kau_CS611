#!/usr/bin/env python3.12
"""
Generate combined publication-quality plots from sweep CSVs.
Outputs to report/figures/ without overwriting existing figure1-5.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = 'report/figures'

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'figure.dpi': 180,
    'axes.grid': True, 'grid.alpha': 0.3,
})

def load_csv(path):
    rows = []
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

def median_by(rows, key_field, val_field, filter_fn=None):
    """Group by key_field, compute median of val_field."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if filter_fn and not filter_fn(r):
            continue
        groups[r[key_field]].append(r[val_field])
    keys = sorted(groups.keys())
    return keys, [np.median(groups[k]) for k in keys]

# ── Load all CSVs ──
py_road_scale = load_csv(f'{FIGURES_DIR}/scalability_roadNet-CA.csv')
py_amz_scale = load_csv(f'{FIGURES_DIR}/scalability_amazon0302.csv')
py_road_speed = load_csv(f'{FIGURES_DIR}/speedup_roadNet-CA.csv')
py_amz_speed = load_csv(f'{FIGURES_DIR}/speedup_amazon0302.csv')

rs_road_scale = load_csv(f'{FIGURES_DIR}/rust/scalability_roadNet-CA.csv')
rs_amz_scale = load_csv(f'{FIGURES_DIR}/rust/scalability_amazon0302.csv')
rs_road_speed = load_csv(f'{FIGURES_DIR}/rust/speedup_roadNet-CA.csv')
rs_amz_speed = load_csv(f'{FIGURES_DIR}/rust/speedup_amazon0302.csv')

colors = {'kruskal': '#1565C0', 'boruvka_seq': '#E65100', 'boruvka_par': '#2E7D32'}
markers = {'kruskal': 'o', 'boruvka_seq': 's', 'boruvka_par': '^'}
labels = {'kruskal': 'Kruskal (Seq)', 'boruvka_seq': 'Borůvka (Seq)', 'boruvka_par': 'Borůvka (Par)'}

# ============================================================
# Figure 6: Python Road Network Scalability (all 3 algos)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ['kruskal', 'boruvka_seq', 'boruvka_par']:
    recs = [r for r in py_road_scale if r['algorithm'] == algo]
    sizes, meds = median_by(recs, 'n_vertices', 'time_s')
    ax.plot([s/1000 for s in sizes], meds, f'{markers[algo]}-',
            color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]} (Python/Numba)')
ax.set_xlabel('Vertices (×1000)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Road Network Scalability — Python (Numba JIT)')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure6_py_road_scalability.png'); plt.close()
print("Saved figure6_py_road_scalability.png")

# ============================================================
# Figure 7: Python Amazon Scalability
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ['kruskal', 'boruvka_seq', 'boruvka_par']:
    recs = [r for r in py_amz_scale if r['algorithm'] == algo]
    sizes, meds = median_by(recs, 'n_vertices', 'time_s')
    ax.plot([s/1000 for s in sizes], meds, f'{markers[algo]}-',
            color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]} (Python/Numba)')
ax.set_xlabel('Vertices (×1000)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Amazon Co-Purchase Scalability — Python (Numba JIT)')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure7_py_amazon_scalability.png'); plt.close()
print("Saved figure7_py_amazon_scalability.png")

# ============================================================
# Figure 8: Rust Road Network Scalability
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ['kruskal', 'boruvka_seq', 'boruvka_par']:
    recs = [r for r in rs_road_scale if r['algorithm'] == algo]
    sizes, meds = median_by(recs, 'n_vertices', 'time_s')
    ax.plot([s/1000 for s in sizes], meds, f'{markers[algo]}-',
            color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]} (Rust/Rayon)')
ax.set_xlabel('Vertices (×1000)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Road Network Scalability — Rust (Rayon + SIMD)')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure8_rs_road_scalability.png'); plt.close()
print("Saved figure8_rs_road_scalability.png")

# ============================================================
# Figure 9: Rust Amazon Scalability
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for algo in ['kruskal', 'boruvka_seq', 'boruvka_par']:
    recs = [r for r in rs_amz_scale if r['algorithm'] == algo]
    sizes, meds = median_by(recs, 'n_vertices', 'time_s')
    ax.plot([s/1000 for s in sizes], meds, f'{markers[algo]}-',
            color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]} (Rust/Rayon)')
ax.set_xlabel('Vertices (×1000)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Amazon Co-Purchase Scalability — Rust (Rayon + SIMD)')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure9_rs_amazon_scalability.png'); plt.close()
print("Saved figure9_rs_amazon_scalability.png")

# ============================================================
# Figure 10: Cross-language comparison (Kruskal) — both datasets
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (py_data, rs_data, title) in zip(axes, [
    (py_road_scale, rs_road_scale, 'roadNet-CA'),
    (py_amz_scale, rs_amz_scale, 'amazon0302'),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_recs = [r for r in py_data if r['algorithm'] == algo]
        rs_recs = [r for r in rs_data if r['algorithm'] == algo]
        py_sizes, py_meds = median_by(py_recs, 'n_vertices', 'time_s')
        rs_sizes, rs_meds = median_by(rs_recs, 'n_vertices', 'time_s')
        ax.plot([s/1000 for s in py_sizes], py_meds, f'{markers[algo]}--',
                color=colors[algo], lw=1.8, ms=6, alpha=0.7, label=f'{labels[algo]} (Python)')
        ax.plot([s/1000 for s in rs_sizes], rs_meds, f'{markers[algo]}-',
                color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]} (Rust)')
    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'{title}')
    ax.legend(fontsize=8)
axes[0].set_title('Road Network: Python vs Rust')
axes[1].set_title('Amazon: Python vs Rust')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure10_cross_language.png'); plt.close()
print("Saved figure10_cross_language.png")

# ============================================================
# Figure 11: Parallel speedup — Python (both datasets)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (data, title) in zip(axes, [
    (py_road_speed, 'roadNet-CA (100K nodes)'),
    (py_amz_speed, 'amazon0302 (262K nodes)'),
]):
    threads, meds = median_by(data, 'threads', 'time_s')
    # Compute speedup relative to 1-thread
    base = meds[0] if meds else 1.0
    speedups = [base / m for m in meds]
    ax.plot(threads, speedups, 'o-', color='#2E7D32', lw=2.2, ms=7, label='Borůvka-Par (Python/Numba)')
    ax.plot(threads, [1.0]*len(threads), ':', color='gray', alpha=0.4, label='Baseline (1.0×)')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(title)
    ax.legend()
plt.suptitle('Parallel Speedup — Python (Numba prange)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure11_py_speedup.png'); plt.close()
print("Saved figure11_py_speedup.png")

# ============================================================
# Figure 12: Parallel speedup — Rust (both datasets)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (data, title) in zip(axes, [
    (rs_road_speed, 'roadNet-CA (100K nodes)'),
    (rs_amz_speed, 'amazon0302 (262K nodes)'),
]):
    # Extract unique thread counts from algorithm names like boruvka_par_t1, boruvka_par_t2, ...
    thread_set = sorted(set(r['threads'] for r in data))
    meds = []
    for tc in thread_set:
        times = [r['time_s'] for r in data if r['threads'] == tc]
        meds.append(np.median(times))
    base = meds[0] if meds else 1.0
    speedups = [base / m for m in meds]
    ax.plot(thread_set, speedups, 'o-', color='#2E7D32', lw=2.2, ms=7, label='Borůvka-Par (Rust/Rayon)')
    ax.plot(thread_set, [1.0]*len(thread_set), ':', color='gray', alpha=0.4, label='Baseline (1.0×)')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(title)
    ax.legend()
plt.suptitle('Parallel Speedup — Rust (Rayon)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure12_rs_speedup.png'); plt.close()
print("Saved figure12_rs_speedup.png")

# ============================================================
# Figure 13: Rust speedup ratio (Rust vs Python) at each size
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (py_data, rs_data, title) in zip(axes, [
    (py_road_scale, rs_road_scale, 'roadNet-CA'),
    (py_amz_scale, rs_amz_scale, 'amazon0302'),
]):
    for algo in ['kruskal', 'boruvka_seq']:
        py_recs = [r for r in py_data if r['algorithm'] == algo]
        rs_recs = [r for r in rs_data if r['algorithm'] == algo]
        py_sizes, py_meds = median_by(py_recs, 'n_vertices', 'time_s')
        rs_sizes, rs_meds = median_by(rs_recs, 'n_vertices', 'time_s')
        # Match sizes
        common = sorted(set(py_sizes) & set(rs_sizes))
        py_map = dict(zip(py_sizes, py_meds))
        rs_map = dict(zip(rs_sizes, rs_meds))
        ratios = [py_map[s] / rs_map[s] for s in common]
        ax.plot([s/1000 for s in common], ratios, f'{markers[algo]}-',
                color=colors[algo], lw=2.2, ms=7, label=f'{labels[algo]}')
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Speedup (Python time / Rust time)')
    ax.set_title(title)
    ax.legend()
plt.suptitle('Rust vs Python Speedup Ratio', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/figure13_rust_vs_python.png'); plt.close()
print("Saved figure13_rust_vs_python.png")

print(f"\nAll new figures saved to {FIGURES_DIR}/")
print("Existing figures (figure1-5) were NOT modified.")
