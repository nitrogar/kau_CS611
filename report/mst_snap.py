#!/usr/bin/env python3
"""
CS611 Project: Sequential vs Parallel MST on SNAP Benchmark Datasets
Domain: Road Networks (roadNet-CA) and E-Commerce Networks (Amazon co-purchase)
Datasets: Stanford Large Network Dataset Collection (SNAP) [9]
"""
import time, random, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Union-Find
# ============================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        return True

# ============================================================
# Algorithms
# ============================================================
def kruskal(n_vertices, edges):
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(n_vertices)
    mst_weight, mst_edges = 0, 0
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges += 1
            if mst_edges == n_vertices - 1: break
    return mst_weight, mst_edges

def boruvka_sequential(n_vertices, edges):
    if not edges: return 0, 0
    uf = UnionFind(n_vertices)
    mst_weight, mst_edges = 0, 0
    n_components = n_vertices
    while n_components > 1:
        min_edge = {}
        for u, v, w in edges:
            cu, cv = uf.find(u), uf.find(v)
            if cu != cv:
                if cu not in min_edge or w < min_edge[cu][0]: min_edge[cu] = (w, u, v)
                if cv not in min_edge or w < min_edge[cv][0]: min_edge[cv] = (w, u, v)
        if not min_edge: break
        added, found_any = set(), False
        for comp_id, (w, u, v) in min_edge.items():
            ek = (min(u, v), max(u, v))
            if ek not in added:
                cu, cv = uf.find(u), uf.find(v)
                if cu != cv:
                    uf.union(cu, cv); added.add(ek)
                    mst_weight += w; mst_edges += 1; found_any = True
        if not found_any: break
        n_components = len(set(uf.find(i) for i in range(n_vertices)))
    return mst_weight, mst_edges

# ============================================================
# SNAP Dataset Loader
# ============================================================
def load_snap(filepath, max_nodes=None):
    """Load SNAP edge list, remap to 0..N-1, add random weights [9]."""
    rng = random.Random(42)
    raw_edges = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u == v: continue
            if max_nodes and (u >= max_nodes or v >= max_nodes): continue
            if u > v: u, v = v, u
            raw_edges.add((u, v))

    # Remap node IDs to 0..N-1
    nodes = set()
    for u, v in raw_edges:
        nodes.add(u); nodes.add(v)
    node_map = {old: new for new, old in enumerate(sorted(nodes))}
    n = len(node_map)
    edges = [(node_map[u], node_map[v], rng.randint(1, 1000)) for u, v in raw_edges]
    return n, edges

# ============================================================
# Experiments
# ============================================================
if __name__ == '__main__':
    # Paths (adjust if running locally)
    road_path = 'roadNet-CA.txt'
    amazon_path = 'amazon0302.txt'

    # Check for uploaded paths
    for p in ['roadNet-CA.txt', 'roadNet-CA.txt']:
        if os.path.exists(p): road_path = p; break
    for p in ['amazon0302.txt', 'amazon0302.txt']:
        if os.path.exists(p): amazon_path = p; break

    os.makedirs('figures', exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'legend.fontsize': 10, 'figure.dpi': 180})

    # ── E1: Road Network Scalability ──
    print("=" * 60)
    print("E1: Road Network (roadNet-CA) Scalability")
    print("=" * 60)
    road_sizes = [5000, 10000, 25000, 50000, 100000]
    road_v, road_e, road_tk, road_tb = [], [], [], []
    for max_n in road_sizes:
        print(f"  Loading roadNet-CA subset (max_node={max_n})...")
        n, edges = load_snap(road_path, max_nodes=max_n)
        print(f"    V={n}, E={len(edges)}")
        road_v.append(n); road_e.append(len(edges))

        t0 = time.perf_counter()
        kw, ke = kruskal(n, edges)
        tk = time.perf_counter() - t0

        t0 = time.perf_counter()
        bw, be = boruvka_sequential(n, edges)
        tb = time.perf_counter() - t0

        assert kw == bw, f"MISMATCH: Kruskal={kw}, Boruvka={bw}"
        print(f"    Kruskal: {tk:.4f}s, Boruvka: {tb:.4f}s, MST weight: {kw}")
        road_tk.append(tk); road_tb.append(tb)

    # ── E2: Amazon Co-Purchase Network ──
    print("\n" + "=" * 60)
    print("E2: Amazon Co-Purchase Network (amazon0302)")
    print("=" * 60)
    amz_sizes = [5000, 10000, 25000, 50000, 100000, 262111]
    amz_v, amz_e, amz_tk, amz_tb = [], [], [], []
    for max_n in amz_sizes:
        label = "full" if max_n >= 262111 else f"max_node={max_n}"
        print(f"  Loading amazon0302 subset ({label})...")
        n, edges = load_snap(amazon_path, max_nodes=None if max_n >= 262111 else max_n)
        print(f"    V={n}, E={len(edges)}")
        amz_v.append(n); amz_e.append(len(edges))

        t0 = time.perf_counter()
        kw, ke = kruskal(n, edges)
        tk = time.perf_counter() - t0

        if n <= 60000:
            t0 = time.perf_counter()
            bw, be = boruvka_sequential(n, edges)
            tb = time.perf_counter() - t0
            assert kw == bw, f"MISMATCH"
        else:
            tb = None  # Too slow for full Boruvka
            print(f"    Boruvka skipped (graph too large for sequential scan)")

        print(f"    Kruskal: {tk:.4f}s, Boruvka: {tb:.4f}s" if tb else f"    Kruskal: {tk:.4f}s")
        amz_tk.append(tk); amz_tb.append(tb)

    # ── E3: Cross-Dataset Comparison (fixed size ~10K nodes) ──
    print("\n" + "=" * 60)
    print("E3: Cross-Dataset Comparison at ~10K nodes")
    print("=" * 60)
    datasets = {
        'roadNet-CA\n(Road Network)': load_snap(road_path, max_nodes=10000),
        'amazon0302\n(E-Commerce)': load_snap(amazon_path, max_nodes=10000),
    }
    cross_names, cross_v, cross_e, cross_tk, cross_tb, cross_avgdeg = [], [], [], [], [], []
    for name, (n, edges) in datasets.items():
        avg_deg = 2 * len(edges) / n if n > 0 else 0
        print(f"  {name.replace(chr(10),' ')}: V={n}, E={len(edges)}, avg_deg={avg_deg:.1f}")
        cross_names.append(name); cross_v.append(n); cross_e.append(len(edges))
        cross_avgdeg.append(avg_deg)

        t0 = time.perf_counter()
        kw, _ = kruskal(n, edges)
        tk = time.perf_counter() - t0

        t0 = time.perf_counter()
        bw, _ = boruvka_sequential(n, edges)
        tb = time.perf_counter() - t0

        assert kw == bw
        print(f"    Kruskal: {tk:.4f}s, Boruvka: {tb:.4f}s")
        cross_tk.append(tk); cross_tb.append(tb)

    # ============================================================
    # FIGURES
    # ============================================================

    # Fig 1: Road network scalability
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([v/1000 for v in road_v], road_tk, 'o-', color='#1565C0', linewidth=2.2,
            markersize=7, label="Kruskal's (Sequential) [2]")
    ax.plot([v/1000 for v in road_v], road_tb, 's-', color='#E65100', linewidth=2.2,
            markersize=7, label="Borůvka's (Sequential) [3]")
    ax.set_xlabel('Number of Vertices (x1000)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('E1: Road Network Scalability (SNAP roadNet-CA) [9]')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig('figures/figure1_road_scalability.png'); plt.close()

    # Fig 2: Amazon scalability (Kruskal only for large, both for small)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([v/1000 for v in amz_v], amz_tk, 'o-', color='#1565C0', linewidth=2.2,
            markersize=7, label="Kruskal's [2]")
    # Plot Boruvka only where available
    bv = [amz_v[i]/1000 for i in range(len(amz_tb)) if amz_tb[i] is not None]
    bt = [amz_tb[i] for i in range(len(amz_tb)) if amz_tb[i] is not None]
    ax.plot(bv, bt, 's-', color='#E65100', linewidth=2.2, markersize=7, label="Borůvka's [3]")
    ax.set_xlabel('Number of Vertices (x1000)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('E2: Amazon Co-Purchase Network (SNAP amazon0302) [9]')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig('figures/figure2_amazon_scalability.png'); plt.close()

    # Fig 3: Speedup ratio (road network)
    speedup = [tb/tk for tk, tb in zip(road_tk, road_tb)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(road_v)), speedup, color='#1565C0', alpha=0.85, edgecolor='#0D47A1')
    ax.set_xticks(range(len(road_v)))
    ax.set_xticklabels([f"{v//1000}K" for v in road_v])
    ax.set_xlabel('Graph Size (vertices from roadNet-CA) [9]')
    ax.set_ylabel("Kruskal's Speedup over Borůvka (x)")
    ax.set_title("Kruskal's Advantage on Real Road Network Data [2,3,9]")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(speedup):
        ax.text(i, v + 0.1, f'{v:.1f}x', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); fig.savefig('figures/figure3_road_speedup.png'); plt.close()

    # Fig 4: Cross-dataset comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cross_names))
    width = 0.35
    ax.bar(x - width/2, cross_tk, width, color='#1565C0', label="Kruskal's [2]")
    ax.bar(x + width/2, cross_tb, width, color='#E65100', label="Borůvka's [3]")
    ax.set_xticks(x)
    ax.set_xticklabels(cross_names, fontsize=9)
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('E3: Algorithm Performance Across SNAP Datasets (~10K nodes) [9]')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); fig.savefig('figures/figure4_cross_dataset.png'); plt.close()

    # Fig 5: Parallel speedup (from literature)
    fig, ax = plt.subplots(figsize=(8, 5))
    cores = np.array([1, 2, 4, 8, 16, 32, 64])
    parallel_frac = 0.70
    amdahl = 1.0 / ((1 - parallel_frac) + parallel_frac / cores)
    ax.plot(cores, amdahl, 'k--', linewidth=1.5, alpha=0.6, label="Amdahl's Law (70% parallel) [4]")
    ax.plot(cores, cores, ':', color='gray', alpha=0.4, label='Ideal linear speedup')
    lit_cores = [1, 2, 4, 6, 8, 12]
    lit_durb = [1.0, 1.45, 1.75, 1.90, 2.03, 2.10]
    lit_parlay = [1.0, 1.8, 3.2, 5.5, 8.0, 10.5]
    ax.plot(lit_cores, lit_durb, 'D-', color='#E65100', linewidth=2, markersize=7,
            label='Durbhakula (2020), C++ CAS [4]')
    ax.plot(lit_cores[:5], lit_parlay[:5], '^-', color='#2E7D32', linewidth=2, markersize=7,
            label='Bhargava & Zaia (2023), ParlayLib [6]')
    ax.set_xlabel('Number of Cores / Threads')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Parallel Borůvka: Theoretical vs. Published Results [4,6]')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 35); ax.set_ylim(0, 14)
    plt.tight_layout(); fig.savefig('figures/figure5_parallel_literature.png'); plt.close()

    print("\n" + "=" * 60)
    print("ALL FIGURES SAVED TO figures/")
    print("=" * 60)
    print("Files:", sorted(os.listdir('figures')))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY TABLE: Road Network (roadNet-CA)")
    print("=" * 60)
    print(f"{'Vertices':>10} {'Edges':>10} {'Kruskal(s)':>12} {'Boruvka(s)':>12} {'Speedup':>10}")
    for i in range(len(road_v)):
        sp = road_tb[i]/road_tk[i]
        print(f"{road_v[i]:>10} {road_e[i]:>10} {road_tk[i]:>12.4f} {road_tb[i]:>12.4f} {sp:>10.1f}x")

    print(f"\nSUMMARY TABLE: Amazon Co-Purchase (amazon0302)")
    print(f"{'Vertices':>10} {'Edges':>10} {'Kruskal(s)':>12} {'Boruvka(s)':>12}")
    for i in range(len(amz_v)):
        bt_str = f"{amz_tb[i]:>12.4f}" if amz_tb[i] else "    skipped "
        print(f"{amz_v[i]:>10} {amz_e[i]:>10} {amz_tk[i]:>12.4f} {bt_str}")
