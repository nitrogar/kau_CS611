#!/usr/bin/env python3.12
"""
CS611 Project: Sequential vs Parallel MST on SNAP Benchmark Datasets
Kruskal (sequential), Borůvka (sequential + parallel) with Numba JIT
All simulation parameters tunable via CLI for parameter sweep plots.

Usage:
  python3.12 mst_python.py --dataset datasets/roadNet-CA.txt --sizes 5000,10000,25000,50000,100000
  python3.12 mst_python.py --dataset datasets/amazon0302.txt --sizes 5000,10000,25000,50000,100000,262111
  python3.12 mst_python.py --dataset datasets/roadNet-CA.txt --sizes 100000 --threads 1,2,4,8,16 --experiment speedup
"""
import argparse, time, os, sys, json
import numpy as np
from numba import njit, prange, config as numba_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Union-Find / Disjoint Set (Numba-compatible, array-based)
# Tracks which vertices belong to the same connected component.
# Used by all MST algorithms to detect cycles.
# ============================================================
@njit
def uf_init(n):
    """Initialize Union-Find for n elements.
    Input:  n (int) — number of vertices
    Output: parent (int32[n]) — parent[i] = i (each vertex is its own root)
            rank   (int32[n]) — rank[i] = 0 (tree height heuristic)
    """
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    return parent, rank

@njit
def uf_find(parent, x):
    """Find the root representative of vertex x with path splitting.
    Input:  parent (int32[n]) — parent array (mutated in-place for compression)
            x      (int)      — vertex to find root of
    Output: root   (int)      — the root representative of x's component
    Side effect: path splitting — parent[x] is updated to grandparent,
                 flattening the tree for O(α(n)) amortized lookups.
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path splitting
        x = parent[x]
    return x

@njit
def uf_union(parent, rank, a, b):
    """Merge the components containing vertices a and b (union by rank).
    Input:  parent (int32[n]) — parent array (mutated in-place)
            rank   (int32[n]) — rank array (mutated in-place)
            a, b   (int)      — vertices to merge
    Output: merged (bool)     — True if a and b were in different components
                                 (i.e., an MST edge was added), False if same.
    """
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

# ============================================================
# Kruskal's Algorithm (Numba JIT) [2]
# Strategy: Sort all edges by weight, greedily add lightest
#           edge that doesn't form a cycle. O(E log E) total.
# ============================================================
@njit
def kruskal_numba(n, eu, ev, ew):
    """Kruskal's MST — sort edges by weight, greedily add non-cycle edges.
    Input:  n  (int)       — number of vertices (0..n-1)
            eu (int32[E])  — source vertex of each edge
            ev (int32[E])  — destination vertex of each edge
            ew (int32[E])  — weight of each edge
    Output: mst_weight (int64) — total weight of the MST
            mst_count  (int)   — number of edges in the MST (should be n-1)
    
    Steps:
      1. Sort edges by weight (O(E log E)) — dominates runtime
      2. For each edge in sorted order:
         - If endpoints are in different components → add to MST
         - If same component → skip (would create cycle)
      3. Stop when n-1 edges added (MST complete)
    """
    # Step 1: Sort — O(E log E), the bottleneck
    order = np.argsort(ew)
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    # Step 2: Greedy selection — O(E · α(n)) ≈ O(E)
    for idx in range(len(order)):
        i = order[idx]
        u, v, w = eu[i], ev[i], ew[i]  # u=source vertex, v=dest vertex, w=edge weight
        # Check if u and v are in different components (adding won't create a cycle)
        if uf_find(parent, u) != uf_find(parent, v):
            # uf_union(u, v): merge the two components containing u and v into one.
            # This means u and v are now "connected" in our growing MST.
            uf_union(parent, rank, u, v)
            mst_weight += w              # add this edge's cost to MST total
            mst_count += 1               # one more edge in our MST
            # Step 3: Early termination — a tree with n vertices has exactly n-1 edges
            if mst_count == n - 1:
                break
        # If u and v are already in the same component, skip —
        # adding it would create a cycle, not allowed in a tree.
    return mst_weight, mst_count

# ============================================================
# Borůvka Sequential (Numba JIT + graph contraction) [3,4]
# Strategy: Each round, every component finds its cheapest
#           outgoing edge and merges. O(E log V) total because
#           components halve each round → O(log V) rounds.
# ============================================================
@njit
def boruvka_seq_numba(n, eu, ev, ew):
    """Borůvka's MST (sequential) with graph contraction.
    Input:  n  (int)       — number of vertices (0..n-1)
            eu (int32[E])  — source vertex of each edge (MUTATED by contraction)
            ev (int32[E])  — destination vertex of each edge (MUTATED)
            ew (int32[E])  — weight of each edge (MUTATED)
    Output: mst_weight (int64) — total weight of the MST
            mst_count  (int)   — number of edges in the MST
    
    Per-round steps:
      Phase 1 (Find-Min):   For each component, find cheapest outgoing edge
      Phase 2 (Merge):      Union-Find merge on all cheapest edges
      Phase 3 (Contract):   Remove intra-component edges (shrinks edge set)
    """
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu)                # active edge count (shrinks each round)
    cheapest_w = np.empty(n, dtype=np.int32)
    cheapest_idx = np.empty(n, dtype=np.int32)
    n_comp = n                 # number of remaining components

    while n_comp > 1:
        # ── Phase 1: Find-Min ──────────────────────────────────
        # Input:  edge arrays eu/ev/ew[0..m), parent array
        # Output: cheapest_w[c]   = min weight of any edge leaving component c
        #         cheapest_idx[c] = index of that edge in eu/ev/ew
        #         found = True if any inter-component edge exists
        cheapest_w[:] = np.iinfo(np.int32).max
        cheapest_idx[:] = -1
        found = False
        for i in range(m):
            cu = uf_find(parent, eu[i])   # component of source
            cv = uf_find(parent, ev[i])   # component of dest
            if cu != cv:                  # inter-component edge
                w = ew[i]
                if w < cheapest_w[cu]:    # update comp cu's cheapest
                    cheapest_w[cu] = w
                    cheapest_idx[cu] = i
                if w < cheapest_w[cv]:    # update comp cv's cheapest
                    cheapest_w[cv] = w
                    cheapest_idx[cv] = i
                found = True
        if not found:
            break

        # ── Phase 2: Merge ─────────────────────────────────────
        # Input:  cheapest_idx[c] for each component c
        # Output: parent/rank updated, mst_weight/mst_count incremented
        #         merged = number of new MST edges this round
        merged = 0
        for c in range(n):
            idx = cheapest_idx[c]
            if idx >= 0:
                u, v, w = eu[idx], ev[idx], ew[idx]
                if uf_union(parent, rank, u, v):
                    mst_weight += w
                    mst_count += 1
                    merged += 1
        if merged == 0:
            break
        n_comp -= merged

        # ── Phase 3: Contract (edge filtering) ─────────────────
        # Input:  eu/ev/ew[0..m), updated parent
        # Output: eu/ev/ew[0..new_m) — only inter-component edges remain
        #         m = new_m (edge count shrinks ~50% per round)
        new_m = 0
        for i in range(m):
            if uf_find(parent, eu[i]) != uf_find(parent, ev[i]):
                eu[new_m] = eu[i]
                ev[new_m] = ev[i]
                ew[new_m] = ew[i]
                new_m += 1
        m = new_m

    return mst_weight, mst_count

# ============================================================
# Borůvka Parallel (Numba prange + contraction) [3,4]
# The find-minimum phase is sequential (indexed reductions
# are not race-free under prange). Parallelism is in the
# comp-ID flattening and edge contraction phases, which are
# embarrassingly parallel and shrink the edge set each round.
# ============================================================
@njit(parallel=True)
def _build_comp_ids(parent, n):
    """Parallel Phase: Flatten component IDs via path-splitting.
    Input:  parent (int32[n]) — Union-Find parent array (mutated for compression)
            n      (int)      — number of vertices
    Output: comp   (int32[n]) — comp[i] = root representative of vertex i
    Parallelism: each vertex independently walks to its root (prange over V).
    """
    comp = np.empty(n, dtype=np.int32)
    for i in prange(n):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path splitting
            x = parent[x]
        comp[i] = x
    return comp

@njit(parallel=True)
def _contract_edges_par(eu, ev, ew, comp_ids, m):
    """Parallel Phase: Mark which edges cross component boundaries.
    Input:  eu, ev   (int32[E]) — edge endpoint arrays
            ew       (int32[E]) — edge weights (unused but passed for consistency)
            comp_ids (int32[n]) — component ID of each vertex
            m        (int)      — number of active edges
    Output: keep     (bool[m])  — keep[i] = True if edge i connects two
                                   different components (inter-component)
    Parallelism: each edge independently checks its endpoints (prange over E).
    """
    keep = np.empty(m, dtype=np.bool_)
    for i in prange(m):
        keep[i] = (comp_ids[eu[i]] != comp_ids[ev[i]])
    return keep

@njit
def boruvka_par_numba(n, eu, ev, ew):
    """Borůvka's MST (partially parallel) — prange for comp-ID and contraction.
    Input:  n  (int)       — number of vertices (0..n-1)
            eu (int32[E])  — source vertex of each edge (MUTATED)
            ev (int32[E])  — destination vertex of each edge (MUTATED)
            ew (int32[E])  — weight of each edge (MUTATED)
    Output: mst_weight (int64) — total weight of the MST
            mst_count  (int)   — number of edges in the MST
    
    What's parallel vs sequential:
      ✅ PARALLEL: comp-ID flattening (prange over V)  — _build_comp_ids
      ❌ SEQUENTIAL: find-minimum (data race on cheapest_w[comp_id])
      ❌ SEQUENTIAL: merge (Union-Find is inherently serial)
      ✅ PARALLEL: edge contraction mask (prange over E) — _contract_edges_par
    """
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu)
    n_comp = n

    cheapest_w = np.empty(n, dtype=np.int32)
    cheapest_idx = np.empty(n, dtype=np.int32)

    while n_comp > 1:
        # Parallel: flatten component IDs
        comp_ids = _build_comp_ids(parent, n)

        # Sequential: find min outgoing edge per component
        cheapest_w[:] = np.iinfo(np.int32).max
        cheapest_idx[:] = -1
        found = False
        for i in range(m):
            cu = comp_ids[eu[i]]
            cv = comp_ids[ev[i]]
            if cu != cv:
                w = ew[i]
                if w < cheapest_w[cu]:
                    cheapest_w[cu] = w
                    cheapest_idx[cu] = i
                if w < cheapest_w[cv]:
                    cheapest_w[cv] = w
                    cheapest_idx[cv] = i
                found = True
        if not found:
            break

        # Sequential: merge components
        merged = 0
        for c in range(n):
            idx = cheapest_idx[c]
            if idx >= 0:
                u, v, w = eu[idx], ev[idx], ew[idx]
                if uf_union(parent, rank, u, v):
                    mst_weight += w
                    mst_count += 1
                    merged += 1
        if merged == 0:
            break
        n_comp -= merged

        # Parallel: filter internal edges (contraction)
        comp_ids2 = _build_comp_ids(parent, n)
        keep = _contract_edges_par(eu, ev, ew, comp_ids2, m)
        new_m = 0
        for i in range(m):
            if keep[i]:
                eu[new_m] = eu[i]
                ev[new_m] = ev[i]
                ew[new_m] = ew[i]
                new_m += 1
        m = new_m

    return mst_weight, mst_count

# ============================================================
# Borůvka "Pooled" Parallel — Reduced-overhead variant
# Key improvements over boruvka_par_numba:
#   1. Find-minimum is PARALLELIZED via chunked local reduction
#      (was fully sequential in boruvka_par — the biggest bottleneck)
#   2. Only 1 parallel region per round instead of 3-4
#   3. Adaptive: falls back to sequential when edges < threshold
#   4. Fewer barrier synchronizations = lower per-round overhead
# ============================================================
@njit(parallel=True)
def _pooled_round(eu, ev, ew, parent, n, m, nchunks):
    """Single fused parallel round: comp-ID build + chunked find-min + contract.
    
    Returns (cheapest_w, cheapest_idx, new_comp_ids, keep_mask, found).
    Uses ONE parallel region for comp-ID build, then ONE for chunked find-min,
    instead of the 3-4 separate parallel regions in boruvka_par.
    """
    # Phase A: Build comp IDs (parallel over vertices)
    comp = np.empty(n, dtype=np.int32)
    for i in prange(n):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        comp[i] = x

    # Phase B: Chunked parallel find-minimum
    # Each chunk processes a slice of edges with LOCAL cheapest arrays,
    # avoiding the data-race issue that forced the original to be sequential.
    chunk_sz = max(1, (m + nchunks - 1) // nchunks)
    actual_chunks = min(nchunks, max(1, (m + chunk_sz - 1) // chunk_sz))

    # Per-chunk local arrays: shape (actual_chunks, n)
    all_w = np.full((actual_chunks, n), np.iinfo(np.int32).max, dtype=np.int32)
    all_idx = np.full((actual_chunks, n), -1, dtype=np.int32)

    for c in prange(actual_chunks):
        lo = c * chunk_sz
        hi = min(lo + chunk_sz, m)
        for i in range(lo, hi):
            cu = comp[eu[i]]
            cv = comp[ev[i]]
            if cu != cv:
                w = ew[i]
                if w < all_w[c, cu]:
                    all_w[c, cu] = w
                    all_idx[c, cu] = i
                if w < all_w[c, cv]:
                    all_w[c, cv] = w
                    all_idx[c, cv] = i

    # Phase C: Merge per-chunk results (sequential but Numba-compiled)
    cheapest_w = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    cheapest_idx = np.full(n, -1, dtype=np.int32)
    found = False
    for c in range(actual_chunks):
        for i in range(n):
            if all_w[c, i] < cheapest_w[i]:
                cheapest_w[i] = all_w[c, i]
                cheapest_idx[i] = all_idx[c, i]
                found = True

    return cheapest_w, cheapest_idx, comp, found


@njit
def boruvka_pooled_numba(n, eu, ev, ew, nchunks=4):
    """Borůvka with reduced parallel overhead via chunked find-minimum.
    
    Args:
        nchunks: Number of parallel chunks for find-min phase.
                 Lower = less merge overhead but less parallelism.
                 Optimal is typically 4-8 for sparse graphs (E ≈ 3-6×V).
    """
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu)
    n_comp = n
    PAR_THRESHOLD = 10000  # Only use parallel when enough edges

    while n_comp > 1:
        if m >= PAR_THRESHOLD:
            # Parallel path: fused comp-ID + chunked find-min
            cheapest_w, cheapest_idx, comp_ids, found = \
                _pooled_round(eu, ev, ew, parent, n, m, nchunks)
        else:
            # Sequential fallback for small edge sets (late rounds)
            comp_ids = np.empty(n, dtype=np.int32)
            for i in range(n):
                x = i
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                comp_ids[i] = x
            cheapest_w = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
            cheapest_idx = np.full(n, -1, dtype=np.int32)
            found = False
            for i in range(m):
                cu = comp_ids[eu[i]]
                cv = comp_ids[ev[i]]
                if cu != cv:
                    w = ew[i]
                    if w < cheapest_w[cu]:
                        cheapest_w[cu] = w
                        cheapest_idx[cu] = i
                    if w < cheapest_w[cv]:
                        cheapest_w[cv] = w
                        cheapest_idx[cv] = i
                    found = True

        if not found:
            break

        # Merge components (always sequential — Union-Find is inherently serial)
        merged = 0
        for c in range(n):
            idx = cheapest_idx[c]
            if idx >= 0:
                u, v, w = eu[idx], ev[idx], ew[idx]
                if uf_union(parent, rank, u, v):
                    mst_weight += w
                    mst_count += 1
                    merged += 1
        if merged == 0:
            break
        n_comp -= merged

        # Contract: sequential edge filter (edges shrink rapidly, no parallel needed)
        new_m = 0
        for i in range(m):
            xu = eu[i]
            while parent[xu] != xu:
                parent[xu] = parent[parent[xu]]
                xu = parent[xu]
            xv = ev[i]
            while parent[xv] != xv:
                parent[xv] = parent[parent[xv]]
                xv = parent[xv]
            if xu != xv:
                eu[new_m] = eu[i]
                ev[new_m] = ev[i]
                ew[new_m] = ew[i]
                new_m += 1
        m = new_m

    return mst_weight, mst_count


# ============================================================
# Portable LCG PRNG (matches Rust implementation exactly)
# Knuth's 64-bit LCG constants for cross-language reproducibility
# ============================================================
class LcgRng:
    """Simple 64-bit LCG matching the Rust implementation."""
    MULT = 6364136223846793005
    INC  = 1442695040888963407
    MOD  = (1 << 64)

    def __init__(self, seed):
        self.state = seed % self.MOD

    def next_u32(self):
        self.state = (self.state * self.MULT + self.INC) % self.MOD
        return (self.state >> 33) & 0x7FFFFFFF

    def randint(self, lo, hi):
        """Return a random int in [lo, hi] inclusive."""
        span = hi - lo + 1
        return lo + (self.next_u32() % span)

# ============================================================
# SNAP Dataset Loader
# ============================================================
def load_snap(filepath, max_nodes=None, weight_min=1, weight_max=1000, seed=42):
    """Load SNAP edge list, remap to 0..N-1, add random weights [9]."""
    raw_edges = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            if max_nodes and (u >= max_nodes or v >= max_nodes):
                continue
            if u > v:
                u, v = v, u
            raw_edges.add((u, v))
    # Sort edges for deterministic weight assignment across languages
    sorted_edges = sorted(raw_edges)
    nodes = set()
    for u, v in sorted_edges:
        nodes.add(u)
        nodes.add(v)
    node_map = {old: new for new, old in enumerate(sorted(nodes))}
    n = len(node_map)
    rng = LcgRng(seed)
    eu = np.array([node_map[u] for u, v in sorted_edges], dtype=np.int32)
    ev = np.array([node_map[v] for u, v in sorted_edges], dtype=np.int32)
    ew = np.array([rng.randint(weight_min, weight_max) for _ in sorted_edges], dtype=np.int32)
    return n, eu, ev, ew

# ============================================================
# NetworkX Third-Party Validation
# ============================================================
def validate_with_networkx(n, eu, ev, ew, our_weight):
    """Validate MST weight against NetworkX's independent implementation."""
    try:
        import networkx as nx
    except ImportError:
        print("    [SKIP] NetworkX not installed, skipping validation")
        return None, False

    G = nx.Graph()
    for i in range(len(eu)):
        G.add_edge(int(eu[i]), int(ev[i]), weight=int(ew[i]))

    # NetworkX uses Kruskal's by default — independent reference implementation
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    nx_weight = sum(d['weight'] for _, _, d in mst.edges(data=True))

    match = (nx_weight == our_weight)
    return nx_weight, match

# ============================================================
# JIT Warmup
# ============================================================
def warmup_jit():
    print("Warming up Numba JIT compilation...")
    _eu = np.array([0,1,2], dtype=np.int32)
    _ev = np.array([1,2,3], dtype=np.int32)
    _ew = np.array([1,2,3], dtype=np.int32)
    kruskal_numba(10, _eu.copy(), _ev.copy(), _ew.copy())
    boruvka_seq_numba(10, _eu.copy(), _ev.copy(), _ew.copy())
    boruvka_par_numba(10, _eu.copy(), _ev.copy(), _ew.copy())
    boruvka_pooled_numba(10, _eu.copy(), _ev.copy(), _ew.copy())
    boruvka_groups_numba(10, _eu.copy(), _ev.copy(), _ew.copy())
    print("JIT warmup done.\n")

# ============================================================
# Borůvka "Groups" — Per-Component Parallel Find-Min (CSR)
# Key idea: Instead of parallelizing over EDGES (like pooled),
#   parallelize over COMPONENTS. Build a CSR index mapping each
#   component to its inter-component edges, then each thread
#   scans only its assigned components.
#   NO atomics/contention — each thread exclusively owns its components.
# ============================================================

@njit(parallel=True)
def _groups_find_min(offsets, csr_edges, ew, n):
    """Parallel find-min per component using CSR edge index.
    Input:
        offsets   (int32[n+1]) — CSR offset array
        csr_edges (int32[T])   — flat array of edge indices per component
        ew        (int32[m])   — edge weights
        n         (int)        — number of vertices
    Output:
        best_w   (int32[n]) — cheapest edge weight per component
        best_idx (int32[n]) — index of cheapest edge per component
    """
    best_w = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    best_idx = np.full(n, -1, dtype=np.int32)
    for c in prange(n):
        start = offsets[c]
        end = offsets[c + 1]
        for p in range(start, end):
            i = csr_edges[p]
            if ew[i] < best_w[c]:
                best_w[c] = ew[i]
                best_idx[c] = i
    return best_w, best_idx


@njit
def boruvka_groups_numba(n, eu, ev, ew):
    """Borůvka with per-component parallel find-min via CSR edge index.
    Input:  n (int), eu/ev/ew (int32[m]) — edge arrays
    Output: (mst_weight: int64, mst_count: int)
    """
    parent, rank = uf_init(n)
    mst_weight = np.int64(0)
    mst_count = 0
    m = len(eu)
    n_comp = n

    while n_comp > 1:
        # Phase 1: Flatten component IDs
        comp_ids = np.empty(n, dtype=np.int32)
        for i in range(n):
            x = np.int32(i)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            comp_ids[i] = x

        # Phase 2: Build CSR — count edges per component
        counts = np.zeros(n, dtype=np.int32)
        inter_flags = np.zeros(m, dtype=np.int32)  # 1 if inter-component

        for i in range(m):
            cu = comp_ids[eu[i]]
            cv = comp_ids[ev[i]]
            if cu != cv:
                inter_flags[i] = 1
                counts[cu] += 1
                counts[cv] += 1

        total_entries = 0
        for c in range(n):
            total_entries += counts[c]

        if total_entries == 0:
            break

        # Prefix sum for CSR offsets
        offsets = np.zeros(n + 1, dtype=np.int32)
        for c in range(n):
            offsets[c + 1] = offsets[c] + counts[c]

        # Fill CSR edge indices
        csr_edges = np.zeros(total_entries, dtype=np.int32)
        pos = offsets[:n].copy()  # write cursors

        for i in range(m):
            if inter_flags[i] == 1:
                cu = comp_ids[eu[i]]
                cv = comp_ids[ev[i]]
                csr_edges[pos[cu]] = np.int32(i)
                pos[cu] += 1
                csr_edges[pos[cv]] = np.int32(i)
                pos[cv] += 1

        # Phase 3: Parallel find-min per component (via CSR)
        best_w, best_idx = _groups_find_min(offsets, csr_edges, ew, n)

        # Phase 4: Merge
        merged = 0
        for c in range(n):
            idx = best_idx[c]
            if idx >= 0:
                u, v, w = eu[idx], ev[idx], ew[idx]
                if uf_union(parent, rank, u, v):
                    mst_weight += w
                    mst_count += 1
                    merged += 1
        if merged == 0:
            break
        n_comp -= merged

        # Phase 5: Contract
        new_m = 0
        for i in range(m):
            xu = eu[i]
            while parent[xu] != xu:
                parent[xu] = parent[parent[xu]]
                xu = parent[xu]
            xv = ev[i]
            while parent[xv] != xv:
                parent[xv] = parent[parent[xv]]
                xv = parent[xv]
            if xu != xv:
                eu[new_m] = eu[i]
                ev[new_m] = ev[i]
                ew[new_m] = ew[i]
                new_m += 1
        m = new_m

    return mst_weight, mst_count


# ============================================================
# NetworkX Kruskal — Third-party library benchmark
# Uses NetworkX's built-in minimum_spanning_tree (Kruskal's).
# NOT Numba-compiled — runs in pure Python.
# ============================================================
def networkx_kruskal_wrapper(n, eu, ev, ew):
    """Wrapper matching the (n, eu, ev, ew) -> (weight, count) interface."""
    import networkx as nx
    G = nx.Graph()
    for i in range(len(eu)):
        G.add_edge(int(eu[i]), int(ev[i]), weight=int(ew[i]))
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    total_weight = sum(d['weight'] for _, _, d in mst.edges(data=True))
    edge_count = mst.number_of_edges()
    return total_weight, edge_count


# ============================================================
# Benchmark runner
# ============================================================
ALGORITHMS = {
    'kruskal': ('Kruskal (Seq)', kruskal_numba),
    'boruvka_seq': ('Borůvka (Seq)', boruvka_seq_numba),
    'boruvka_par': ('Borůvka (Par)', boruvka_par_numba),
    'boruvka_pooled': ('Borůvka (Pooled)', boruvka_pooled_numba),
    'boruvka_groups': ('Borůvka (Groups)', boruvka_groups_numba),
    'networkx': ('NetworkX (Kruskal)', networkx_kruskal_wrapper),
}

def run_benchmark(n, eu, ev, ew, algo_key, runs=5):
    """Run a single algorithm multiple times, return rich stats."""
    _, func = ALGORITHMS[algo_key]
    times = []
    mst_w = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        w, _ = func(n, eu.copy(), ev.copy(), ew.copy())
        t1 = time.perf_counter()
        times.append(t1 - t0)
        mst_w = int(w)
    times_arr = np.array(times)
    stats = {
        'median': float(np.median(times_arr)),
        'mean': float(np.mean(times_arr)),
        'std': float(np.std(times_arr)),
        'min': float(np.min(times_arr)),
        'max': float(np.max(times_arr)),
    }
    return stats, mst_w, times

# ============================================================
# Experiment: Scalability sweep
# ============================================================
def experiment_scalability(args):
    results = []
    sizes = [int(s) for s in args.sizes.split(',')]
    algos = [a.strip() for a in args.algorithms.split(',')]

    validation_results = []

    for sz in sizes:
        # sz=0 means "load full dataset"; any other value filters to vertex IDs < sz
        max_n = None if sz == 0 else sz
        label = "full" if max_n is None else f"max_node={sz}"
        print(f"  Loading {os.path.basename(args.dataset)} ({label})...")
        n, eu, ev, ew = load_snap(args.dataset, max_nodes=max_n,
                                   weight_min=args.weight_min, weight_max=args.weight_max,
                                   seed=args.seed)
        print(f"    V={n}, E={len(eu)}")

        ref_weight = None
        for algo in algos:
            if algo in ('boruvka_par', 'boruvka_pooled'):
                numba_config.NUMBA_NUM_THREADS = args.default_threads
                os.environ['NUMBA_NUM_THREADS'] = str(args.default_threads)
            stats, mst_w, all_times = run_benchmark(n, eu, ev, ew, algo, runs=args.runs)
            if ref_weight is None:
                ref_weight = mst_w
            else:
                assert mst_w == ref_weight, f"WEIGHT MISMATCH: {algo}={mst_w}, expected={ref_weight}"
            name, _ = ALGORITHMS[algo]
            print(f"    {name}: median={stats['median']:.6f}s  mean={stats['mean']:.6f}s  "
                  f"std={stats['std']:.6f}s  (MST weight={mst_w})")
            for r, t in enumerate(all_times):
                results.append({
                    'dataset': os.path.basename(args.dataset),
                    'algorithm': algo, 'algo_name': name,
                    'n_vertices': n, 'n_edges': len(eu),
                    'threads': args.default_threads if algo in ('boruvka_par', 'boruvka_pooled') else 1,
                    'run': r, 'time_s': t, 'mst_weight': mst_w,
                    'median_s': stats['median'], 'mean_s': stats['mean'],
                    'std_s': stats['std'], 'min_s': stats['min'], 'max_s': stats['max'],
                })

        # Validate against NetworkX
        if args.validate:
            nx_w, match = validate_with_networkx(n, eu, ev, ew, ref_weight)
            if nx_w is not None:
                status = "✓ MATCH" if match else "✗ MISMATCH"
                print(f"    [NetworkX Validation] nx_weight={nx_w}, our_weight={ref_weight} → [{status}]")
                validation_results.append({
                    'dataset': os.path.basename(args.dataset),
                    'n_vertices': n, 'n_edges': len(eu),
                    'our_weight': ref_weight, 'networkx_weight': nx_w,
                    'match': match,
                })

    return results, validation_results

# ============================================================
# Experiment: Parallel speedup sweep
# ============================================================
def experiment_speedup(args):
    results = []
    sizes = [int(s) for s in args.sizes.split(',')]
    thread_counts = [int(t) for t in args.threads.split(',')]

    for sz in sizes:
        max_n = None if sz == 0 else sz
        n, eu, ev, ew = load_snap(args.dataset, max_nodes=max_n,
                                   weight_min=args.weight_min, weight_max=args.weight_max,
                                   seed=args.seed)
        print(f"  V={n}, E={len(eu)}")

        # Sequential baseline
        stats_seq, mst_w, _ = run_benchmark(n, eu, ev, ew, 'boruvka_seq', runs=args.runs)
        med_seq = stats_seq['median']
        print(f"    Borůvka-Seq baseline: median={med_seq:.6f}s")

        for nt in thread_counts:
            numba_config.NUMBA_NUM_THREADS = nt
            os.environ['NUMBA_NUM_THREADS'] = str(nt)
            os.environ['OMP_NUM_THREADS'] = str(nt)
            stats_par, mst_w_p, all_times = run_benchmark(n, eu, ev, ew, 'boruvka_par', runs=args.runs)
            assert mst_w_p == mst_w, f"WEIGHT MISMATCH at threads={nt}"
            med_par = stats_par['median']
            speedup = med_seq / med_par if med_par > 0 else 1.0
            efficiency = speedup / nt * 100
            print(f"    Threads={nt:>2d}: median={med_par:.6f}s, Speedup={speedup:.3f}x, "
                  f"Efficiency={efficiency:.1f}%")
            for r, t in enumerate(all_times):
                results.append({
                    'dataset': os.path.basename(args.dataset),
                    'algorithm': 'boruvka_par', 'algo_name': f'Borůvka-Par (t={nt})',
                    'n_vertices': n, 'n_edges': len(eu),
                    'threads': nt, 'run': r, 'time_s': t, 'mst_weight': mst_w,
                    'seq_baseline': med_seq,
                    'median_s': stats_par['median'], 'mean_s': stats_par['mean'],
                    'std_s': stats_par['std'], 'min_s': stats_par['min'], 'max_s': stats_par['max'],
                })
    return results

# ============================================================
# CSV Output
# ============================================================
def save_csv(results, filepath):
    import csv
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved CSV: {filepath}")

def save_validation_csv(results, filepath):
    import csv
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved validation CSV: {filepath}")

# ============================================================
# Plot generation (basic — detailed plots in generate_ahmed_plots.py)
# ============================================================
def plot_scalability(results, output_dir, dataset_name):
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'legend.fontsize': 9, 'figure.dpi': 180})
    # Group by algorithm, compute median per size
    algos = sorted(set(r['algorithm'] for r in results))
    colors = {'kruskal': '#1565C0', 'boruvka_seq': '#E65100', 'boruvka_par': '#2E7D32'}
    markers = {'kruskal': 'o', 'boruvka_seq': 's', 'boruvka_par': '^'}

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in algos:
        recs = [r for r in results if r['algorithm'] == algo]
        sizes_set = sorted(set(r['n_vertices'] for r in recs))
        medians = []
        for sz in sizes_set:
            times = [r['time_s'] for r in recs if r['n_vertices'] == sz]
            medians.append(np.median(times))
        name = ALGORITHMS[algo][0]
        ax.plot([s/1000 for s in sizes_set], medians,
                f'{markers.get(algo,"o")}-', color=colors.get(algo,'#333'),
                lw=2.2, ms=7, label=name)
    ax.set_xlabel('Vertices (×1000)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Scalability: {dataset_name}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'scalability_{dataset_name}.png')
    fig.savefig(path); plt.close()
    print(f"  Saved plot: {path}")

def plot_speedup(results, output_dir, dataset_name):
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 180})
    if not results:
        return
    thread_counts = sorted(set(r['threads'] for r in results))
    sizes_set = sorted(set(r['n_vertices'] for r in results))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sz in sizes_set:
        recs = [r for r in results if r['n_vertices'] == sz]
        seq_base = recs[0].get('seq_baseline', 1.0)
        speedups = []
        tcs = []
        for tc in thread_counts:
            times = [r['time_s'] for r in recs if r['threads'] == tc]
            if times:
                speedups.append(seq_base / np.median(times))
                tcs.append(tc)
        ax.plot(tcs, speedups, 'o-', lw=2.2, ms=7, label=f'V={sz//1000}K')
    ax.plot(thread_counts, thread_counts, ':', color='gray', alpha=0.4, label='Ideal linear')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(f'Parallel Speedup: {dataset_name}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'speedup_{dataset_name}.png')
    fig.savefig(path); plt.close()
    print(f"  Saved plot: {path}")

# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='MST Benchmark (Python + Numba)')
    p.add_argument('--dataset', required=True, help='Path to SNAP edge-list file')
    p.add_argument('--sizes', default='5000,10000,25000,50000,100000',
                   help='Comma-separated max_nodes values for scalability sweep')
    p.add_argument('--threads', default='1,2,4,8,16',
                   help='Comma-separated thread counts for parallel sweep')
    p.add_argument('--default-threads', type=int, default=0,
                   help='Default thread count for parallel algo (0=all cores)')
    p.add_argument('--weight-min', type=int, default=1, help='Min random edge weight')
    p.add_argument('--weight-max', type=int, default=1000, help='Max random edge weight')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--runs', type=int, default=5, help='Repetitions per measurement')
    p.add_argument('--output-dir', default='results/python', help='Output directory')
    p.add_argument('--algorithms', default='kruskal,boruvka_seq,boruvka_par',
                   help='Algorithms to run (comma-separated)')
    p.add_argument('--experiment', choices=['scalability', 'speedup', 'both'], default='both',
                   help='Which experiment to run')
    p.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    p.add_argument('--validate', action='store_true', default=True,
                   help='Validate MST against NetworkX (default: True)')
    p.add_argument('--no-validate', dest='validate', action='store_false',
                   help='Skip NetworkX validation')
    return p.parse_args()

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    args = parse_args()
    if args.default_threads <= 0:
        args.default_threads = os.cpu_count() or 4
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    warmup_jit()

    all_validation = []

    if args.experiment in ('scalability', 'both'):
        print("=" * 65)
        print(f"SCALABILITY EXPERIMENT: {dataset_name}")
        print("=" * 65)
        scale_results, val_results = experiment_scalability(args)
        all_validation.extend(val_results)
        save_csv(scale_results, os.path.join(args.output_dir, f'scalability_{dataset_name}.csv'))
        if not args.no_plot:
            plot_scalability(scale_results, args.output_dir, dataset_name)

    if args.experiment in ('speedup', 'both'):
        print("\n" + "=" * 65)
        print(f"PARALLEL SPEEDUP EXPERIMENT: {dataset_name}")
        print("=" * 65)
        speed_results = experiment_speedup(args)
        save_csv(speed_results, os.path.join(args.output_dir, f'speedup_{dataset_name}.csv'))
        if not args.no_plot:
            plot_speedup(speed_results, args.output_dir, dataset_name)

    if all_validation:
        save_validation_csv(all_validation, os.path.join(args.output_dir, f'validation_{dataset_name}.csv'))
        print("\n" + "=" * 65)
        print("NETWORKX VALIDATION SUMMARY")
        print("=" * 65)
        all_match = all(v['match'] for v in all_validation)
        for v in all_validation:
            status = "✓" if v['match'] else "✗"
            print(f"  [{status}] V={v['n_vertices']}, E={v['n_edges']}: "
                  f"ours={v['our_weight']}, nx={v['networkx_weight']}")
        if all_match:
            print("  ══════════════════════════════════════")
            print("  ALL VALIDATIONS PASSED ✓")
            print("  ══════════════════════════════════════")
        else:
            print("  !! VALIDATION FAILURES DETECTED !!")

    print("\nDone. Results in:", args.output_dir)
