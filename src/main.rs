//! CS611 Project: Sequential vs Parallel MST on SNAP Benchmark Datasets
//! Kruskal (sequential), Borůvka (sequential + parallel) with Rayon
//! SIMD-friendly layout via struct-of-arrays and native target compilation.
//!
//! Usage:
//!   cargo run --release -- --dataset datasets/roadNet-CA.txt --sizes 5000,10000,25000,50000,100000
//!   cargo run --release -- --dataset datasets/amazon0302.txt --sizes 5000,10000,50000,262111
//!   cargo run --release -- --dataset datasets/roadNet-CA.txt --sizes 100000 --threads 1,2,4,8,16 --experiment speedup

use clap::Parser;
use csv::Writer;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;
use petgraph::graph::UnGraph;
use petgraph::algo::min_spanning_tree;
use petgraph::data::FromElements;

// ============================================================
// Union-Find / Disjoint Set (array-based)
// Tracks which vertices belong to the same connected component.
// Used by all MST algorithms to detect cycles.
//
// Input:  n (usize) — number of vertices
// State:  parent[i] = parent of vertex i (root if parent[i]==i)
//         rank[i]   = tree height heuristic for union-by-rank
// ============================================================
struct UnionFind {
    parent: Vec<i32>,   // parent[i] = parent of vertex i
    rank: Vec<i32>,     // rank[i] = tree height upper bound
}

impl UnionFind {
    /// Create a new Union-Find with n elements.
    /// Output: each vertex is its own root (parent[i] = i, rank[i] = 0)
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as i32).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the root representative of vertex x with path splitting.
    /// Input:  x (i32) — vertex to query
    /// Output: root (i32) — root representative of x's component
    /// Side effect: path splitting — parent[x] updated to grandparent,
    ///              flattening the tree for O(α(n)) amortized lookups.
    #[inline]
    fn find(&mut self, mut x: i32) -> i32 {
        while self.parent[x as usize] != x {
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;  // path splitting
            x = gp;
        }
        x
    }

    /// Merge components containing vertices a and b (union by rank).
    /// Input:  a, b (i32) — vertices to merge
    /// Output: true if a and b were in different components (MST edge added),
    ///         false if they were already in the same component.
    #[inline]
    fn union(&mut self, a: i32, b: i32) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
        true
    }
}

// ============================================================
// Edge arrays (struct-of-arrays for SIMD-friendly access)
//
// Each edge i is defined by three parallel arrays:
//   eu[i] = u = source vertex       (one endpoint of the edge)
//   ev[i] = v = destination vertex   (other endpoint of the edge)
//   ew[i] = w = weight/cost of the edge (integer, 1..1000)
//
// Example: edge 5 connects vertex eu[5] to vertex ev[5]
//          with cost ew[5]. The graph is undirected, so
//          (u→v) and (v→u) are the same edge.
// ============================================================
#[derive(Clone)]
struct EdgeArrays {
    eu: Vec<i32>,   // eu[i] = u = source vertex of edge i
    ev: Vec<i32>,   // ev[i] = v = destination vertex of edge i
    ew: Vec<i32>,   // ew[i] = w = weight (cost) of edge i
}

impl EdgeArrays {
    fn len(&self) -> usize {
        self.eu.len()
    }
}

// ============================================================
// Kruskal's Algorithm (Sequential) [2]
// Strategy: Sort all edges by weight, greedily add lightest
//           edge that doesn't form a cycle. O(E log E) total.
//
// Input:  n (usize)      — number of vertices (0..n-1)
//         edges (&EdgeArrays) — eu[i], ev[i], ew[i] = edge i
// Output: (mst_weight: i64, mst_count: usize)
//         mst_weight = total weight of the MST
//         mst_count  = number of edges in MST (should be n-1)
// ============================================================
fn kruskal(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let m = edges.len();
    // Step 1: Sort edge indices by weight — O(E log E), the bottleneck
    // Uses pdqsort (pattern-defeating quicksort) via sort_unstable
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by_key(|&i| edges.ew[i]);

    let mut uf = UnionFind::new(n);
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    // Step 2: Greedy selection — O(E · α(n)) ≈ O(E)
    for &i in &order {
        let u = edges.eu[i];   // u = source vertex of edge i
        let v = edges.ev[i];   // v = destination vertex of edge i
        let w = edges.ew[i];   // w = weight (cost) of edge i
        // Check if u and v are in different components (adding this edge won't create a cycle)
        if uf.find(u) != uf.find(v) {
            // union(u, v): merge the two components containing u and v into one.
            // This means u and v are now "connected" in our growing MST.
            // Returns true because they were in different components.
            uf.union(u, v);
            mst_weight += w as i64;  // add this edge's cost to MST total
            mst_count += 1;          // one more edge in our MST
            // Step 3: Early termination — a tree with n vertices has exactly n-1 edges
            if mst_count == n - 1 {
                break;
            }
        }
        // If u and v are already in the same component, skip this edge —
        // adding it would create a cycle, which is not allowed in a tree.
    }
    (mst_weight, mst_count)
}

// ============================================================
// Borůvka Sequential (with graph contraction) [3,4]
// Strategy: Each round, every component finds its cheapest
//           outgoing edge and merges. O(E log V) total because
//           components halve each round → O(log V) rounds.
//
// Input:  n (usize)          — number of vertices (0..n-1)
//         edges (&EdgeArrays) — eu[i], ev[i], ew[i] = edge i
// Output: (mst_weight: i64, mst_count: usize)
//
// Per-round phases:
//   Phase 1 (Find-Min):  For each component, find cheapest outgoing edge
//   Phase 2 (Merge):     Union-Find merge on all cheapest edges
//   Phase 3 (Contract):  Remove intra-component edges (shrinks edge set)
// ============================================================
fn boruvka_seq(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut uf = UnionFind::new(n);
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    let mut eu = edges.eu.clone();   // working copy (mutated by contraction)
    let mut ev = edges.ev.clone();
    let mut ew = edges.ew.clone();
    let mut m = eu.len();            // active edge count (shrinks each round)
    let mut n_comp = n;              // remaining components

    let mut cheapest_w = vec![i32::MAX; n];
    let mut cheapest_idx: Vec<i32> = vec![-1; n];

    while n_comp > 1 {
        // ── Phase 1: Find-Min ──────────────────────────────────
        // Input:  eu/ev/ew[0..m), Union-Find uf
        // Output: cheapest_w[c]   = min weight of edge leaving component c
        //         cheapest_idx[c] = index of that edge
        //         found = true if any inter-component edge exists
        cheapest_w.iter_mut().for_each(|x| *x = i32::MAX);
        cheapest_idx.iter_mut().for_each(|x| *x = -1);

        let mut found = false;
        for i in 0..m {
            let cu = uf.find(eu[i]);    // component of source vertex
            let cv = uf.find(ev[i]);    // component of dest vertex
            if cu != cv {               // inter-component edge
                let w = ew[i];
                if w < cheapest_w[cu as usize] {
                    cheapest_w[cu as usize] = w;
                    cheapest_idx[cu as usize] = i as i32;
                }
                if w < cheapest_w[cv as usize] {
                    cheapest_w[cv as usize] = w;
                    cheapest_idx[cv as usize] = i as i32;
                }
                found = true;
            }
        }
        if !found {
            break;
        }

        // ── Phase 2: Merge ─────────────────────────────────────
        // Input:  cheapest_idx[c] for each component c
        // Output: uf updated, mst_weight/mst_count incremented
        //         merged = number of new MST edges this round
        let mut merged = 0usize;
        for c in 0..n {
            let idx = cheapest_idx[c];
            if idx >= 0 {
                let i = idx as usize;
                if uf.union(eu[i], ev[i]) {
                    mst_weight += ew[i] as i64;
                    mst_count += 1;
                    merged += 1;
                }
            }
        }
        if merged == 0 {
            break;
        }
        n_comp -= merged;

        // ── Phase 3: Contract (edge filtering) ─────────────────
        // Input:  eu/ev/ew[0..m), updated uf
        // Output: eu/ev/ew[0..new_m) — only inter-component edges remain
        //         m = new_m (edge count shrinks ~50% per round)
        let mut new_m = 0usize;
        for i in 0..m {
            if uf.find(eu[i]) != uf.find(ev[i]) {
                eu[new_m] = eu[i];
                ev[new_m] = ev[i];
                ew[new_m] = ew[i];
                new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}

// Borůvka Parallel (Rayon + contraction) [3,4]
// SIMD-friendly: contiguous i32 arrays processed in parallel chunks.
// Uses explicit chunking with pre-allocated buffers per thread to avoid
// allocation overhead in the fold/reduce pattern.
// ============================================================
fn boruvka_par(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut parent: Vec<i32> = (0..n as i32).collect();
    let mut rank: Vec<i32> = vec![0; n];
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    let mut eu = edges.eu.clone();
    let mut ev = edges.ev.clone();
    let mut ew = edges.ew.clone();
    let mut m = eu.len();
    let mut n_comp = n;

    let mut cheapest_w = vec![i32::MAX; n];
    let mut cheapest_idx: Vec<i32> = vec![-1; n];

    while n_comp > 1 {
        // Parallel: flatten component IDs
        let comp_ids: Vec<i32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut x = i as i32;
                while parent[x as usize] != x {
                    x = parent[parent[x as usize] as usize];
                }
                x
            })
            .collect();

        // Sequential find-minimum (avoids allocation overhead for thread-local
        // arrays which dominates at small-medium graph sizes). The edge set
        // shrinks each round via contraction, so this is fast.
        cheapest_w.iter_mut().for_each(|x| *x = i32::MAX);
        cheapest_idx.iter_mut().for_each(|x| *x = -1);

        let mut found = false;
        for i in 0..m {
            let cu = comp_ids[eu[i] as usize] as usize;
            let cv = comp_ids[ev[i] as usize] as usize;
            if cu != cv {
                let w = ew[i];
                if w < cheapest_w[cu] {
                    cheapest_w[cu] = w;
                    cheapest_idx[cu] = i as i32;
                }
                if w < cheapest_w[cv] {
                    cheapest_w[cv] = w;
                    cheapest_idx[cv] = i as i32;
                }
                found = true;
            }
        }
        if !found {
            break;
        }

        // Serial merge phase
        let mut merged = 0usize;
        let mut uf_local = UnionFind {
            parent: parent.clone(),
            rank: rank.clone(),
        };

        for c in 0..n {
            let idx = cheapest_idx[c];
            if idx >= 0 {
                let i = idx as usize;
                if uf_local.union(eu[i], ev[i]) {
                    mst_weight += ew[i] as i64;
                    mst_count += 1;
                    merged += 1;
                }
            }
        }
        parent = uf_local.parent;
        rank = uf_local.rank;

        if merged == 0 {
            break;
        }
        n_comp -= merged;

        // Parallel: compute keep mask for edge contraction
        let keep: Vec<bool> = eu[..m]
            .par_iter()
            .zip(ev[..m].par_iter())
            .map(|(&u, &v)| {
                let mut xu = u;
                while parent[xu as usize] != xu {
                    xu = parent[parent[xu as usize] as usize];
                }
                let mut xv = v;
                while parent[xv as usize] != xv {
                    xv = parent[parent[xv as usize] as usize];
                }
                xu != xv
            })
            .collect();

        // Compact (serial, but simple memcpy-like)
        let mut new_m = 0usize;
        for i in 0..m {
            if keep[i] {
                eu[new_m] = eu[i];
                ev[new_m] = ev[i];
                ew[new_m] = ew[i];
                new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}

// ============================================================
// Borůvka "Pooled" Parallel — Reduced-overhead variant
// Key improvements over boruvka_par:
//   1. Find-minimum is PARALLELIZED via chunked local reduction
//      (was fully sequential in boruvka_par)
//   2. Adaptive: falls back to sequential for small edge sets
//   3. Uses Rayon's par_chunks instead of per-element par_iter,
//      reducing work-stealing overhead for fine-grained tasks
// ============================================================
fn boruvka_pooled(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut parent: Vec<i32> = (0..n as i32).collect();
    let mut rank: Vec<i32> = vec![0; n];
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    let mut eu = edges.eu.clone();
    let mut ev = edges.ev.clone();
    let mut ew = edges.ew.clone();
    let mut m = eu.len();
    let mut n_comp = n;

    const PAR_THRESHOLD: usize = 10000;
    let nchunks = rayon::current_num_threads().max(1).min(8); // Cap chunks

    while n_comp > 1 {
        // Parallel: flatten component IDs
        let comp_ids: Vec<i32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut x = i as i32;
                while parent[x as usize] != x {
                    x = parent[parent[x as usize] as usize];
                }
                x
            })
            .collect();

        let (cheapest_w, cheapest_idx, found) = if m >= PAR_THRESHOLD {
            // Chunked parallel find-minimum: each chunk gets LOCAL arrays
            let chunk_sz = (m + nchunks - 1) / nchunks;
            let chunk_results: Vec<(Vec<i32>, Vec<i32>)> = (0..nchunks)
                .into_par_iter()
                .map(|c| {
                    let lo = c * chunk_sz;
                    let hi = (lo + chunk_sz).min(m);
                    let mut local_w = vec![i32::MAX; n];
                    let mut local_idx: Vec<i32> = vec![-1; n];
                    for i in lo..hi {
                        let cu = comp_ids[eu[i] as usize] as usize;
                        let cv = comp_ids[ev[i] as usize] as usize;
                        if cu != cv {
                            let w = ew[i];
                            if w < local_w[cu] {
                                local_w[cu] = w;
                                local_idx[cu] = i as i32;
                            }
                            if w < local_w[cv] {
                                local_w[cv] = w;
                                local_idx[cv] = i as i32;
                            }
                        }
                    }
                    (local_w, local_idx)
                })
                .collect();

            // Merge per-chunk results (sequential)
            let mut cw = vec![i32::MAX; n];
            let mut ci: Vec<i32> = vec![-1; n];
            let mut f = false;
            for (lw, li) in &chunk_results {
                for j in 0..n {
                    if lw[j] < cw[j] {
                        cw[j] = lw[j];
                        ci[j] = li[j];
                        f = true;
                    }
                }
            }
            (cw, ci, f)
        } else {
            // Sequential fallback for small edge sets (late rounds)
            let mut cw = vec![i32::MAX; n];
            let mut ci: Vec<i32> = vec![-1; n];
            let mut f = false;
            for i in 0..m {
                let cu = comp_ids[eu[i] as usize] as usize;
                let cv = comp_ids[ev[i] as usize] as usize;
                if cu != cv {
                    let w = ew[i];
                    if w < cw[cu] {
                        cw[cu] = w;
                        ci[cu] = i as i32;
                    }
                    if w < cw[cv] {
                        cw[cv] = w;
                        ci[cv] = i as i32;
                    }
                    f = true;
                }
            }
            (cw, ci, f)
        };

        if !found {
            break;
        }

        // Serial merge phase
        let mut merged = 0usize;
        let mut uf_local = UnionFind {
            parent: parent.clone(),
            rank: rank.clone(),
        };

        for c in 0..n {
            let idx = cheapest_idx[c];
            if idx >= 0 {
                let i = idx as usize;
                if uf_local.union(eu[i], ev[i]) {
                    mst_weight += ew[i] as i64;
                    mst_count += 1;
                    merged += 1;
                }
            }
        }
        parent = uf_local.parent;
        rank = uf_local.rank;

        if merged == 0 {
            break;
        }
        n_comp -= merged;

        // Sequential contraction (edges shrink rapidly, parallel overhead not worth it)
        let mut new_m = 0usize;
        for i in 0..m {
            let mut xu = eu[i];
            while parent[xu as usize] != xu {
                xu = parent[parent[xu as usize] as usize];
            }
            let mut xv = ev[i];
            while parent[xv as usize] != xv {
                xv = parent[parent[xv as usize] as usize];
            }
            if xu != xv {
                eu[new_m] = eu[i];
                ev[new_m] = ev[i];
                ew[new_m] = ew[i];
                new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}

// ============================================================
// Borůvka "Groups" — Per-Component Parallel Find-Min (CSR)
// Key idea: Instead of parallelizing over EDGES (like pooled),
//   parallelize over COMPONENTS. Build a CSR (Compressed Sparse
//   Row) index mapping each component to its inter-component
//   edges, then each thread scans only its assigned components.
//   NO atomics needed — each thread exclusively owns its components.
//
// Input:  n (usize)          — number of vertices
//         edges (&EdgeArrays) — edge arrays
// Output: (mst_weight: i64, mst_count: usize)
//
// Per-round phases:
//   Phase 1: Flatten component IDs (parallel)
//   Phase 2: Build CSR — count edges per component, prefix-sum,
//            fill edge index array
//   Phase 3: Find-min per component (parallel over components)
//   Phase 4: Merge via Union-Find (sequential)
//   Phase 5: Contract — remove intra-component edges
// ============================================================
fn boruvka_groups(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut parent: Vec<i32> = (0..n as i32).collect();
    let mut rank: Vec<i32> = vec![0; n];
    let mut mst_weight: i64 = 0;
    let mut mst_count: usize = 0;

    let mut eu = edges.eu.clone();
    let mut ev = edges.ev.clone();
    let mut ew = edges.ew.clone();
    let mut m = eu.len();
    let mut n_comp = n;

    while n_comp > 1 {
        // ── Phase 1: Parallel flatten component IDs ──
        let comp_ids: Vec<i32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut x = i as i32;
                while parent[x as usize] != x {
                    x = parent[parent[x as usize] as usize];
                }
                x
            })
            .collect();

        // ── Phase 2: Build CSR edge index per component ──
        // First pass: count inter-component edges per component
        let mut counts = vec![0u32; n];
        let mut inter_edge_flags = vec![false; m]; // which edges are inter-component

        for i in 0..m {
            let cu = comp_ids[eu[i] as usize];
            let cv = comp_ids[ev[i] as usize];
            if cu != cv {
                inter_edge_flags[i] = true;
                counts[cu as usize] += 1;
                counts[cv as usize] += 1;
            }
        }

        // Check if any inter-component edges exist
        let total_entries: usize = counts.iter().map(|&c| c as usize).sum();
        if total_entries == 0 {
            break;
        }

        // Prefix sum for CSR offsets
        let mut offsets = vec![0u32; n + 1];
        for c in 0..n {
            offsets[c + 1] = offsets[c] + counts[c];
        }

        // Second pass: fill CSR edge indices
        let mut csr_edges = vec![0u32; total_entries];
        let mut pos = offsets[..n].to_vec(); // write cursors

        for i in 0..m {
            if inter_edge_flags[i] {
                let cu = comp_ids[eu[i] as usize] as usize;
                let cv = comp_ids[ev[i] as usize] as usize;
                csr_edges[pos[cu] as usize] = i as u32;
                pos[cu] += 1;
                csr_edges[pos[cv] as usize] = i as u32;
                pos[cv] += 1;
            }
        }

        // ── Phase 3: Parallel find-min per component ──
        // Each component scans ONLY its own edges — no contention, no atomics
        let cheapest: Vec<(i32, i32)> = (0..n)
            .into_par_iter()
            .map(|c| {
                let start = offsets[c] as usize;
                let end = offsets[c + 1] as usize;
                if start == end {
                    return (i32::MAX, -1i32);
                }
                let mut best_w = i32::MAX;
                let mut best_idx = -1i32;
                for pos in start..end {
                    let i = csr_edges[pos] as usize;
                    if ew[i] < best_w {
                        best_w = ew[i];
                        best_idx = i as i32;
                    }
                }
                (best_w, best_idx)
            })
            .collect();

        // ── Phase 4: Sequential merge ──
        let mut merged = 0usize;
        let mut uf_local = UnionFind {
            parent: parent.clone(),
            rank: rank.clone(),
        };

        for c in 0..n {
            let (_, idx) = cheapest[c];
            if idx >= 0 {
                let i = idx as usize;
                if uf_local.union(eu[i], ev[i]) {
                    mst_weight += ew[i] as i64;
                    mst_count += 1;
                    merged += 1;
                }
            }
        }
        parent = uf_local.parent;
        rank = uf_local.rank;

        if merged == 0 {
            break;
        }
        n_comp -= merged;

        // ── Phase 5: Contract — remove intra-component edges ──
        let mut new_m = 0usize;
        for i in 0..m {
            let mut xu = eu[i];
            while parent[xu as usize] != xu {
                xu = parent[parent[xu as usize] as usize];
            }
            let mut xv = ev[i];
            while parent[xv as usize] != xv {
                xv = parent[parent[xv as usize] as usize];
            }
            if xu != xv {
                eu[new_m] = eu[i];
                ev[new_m] = ev[i];
                ew[new_m] = ew[i];
                new_m += 1;
            }
        }
        m = new_m;
    }
    (mst_weight, mst_count)
}

// ============================================================
// Petgraph Kruskal — Third-party library benchmark
// Uses petgraph's built-in min_spanning_tree (Kruskal's algorithm).
// This serves as an independent reference implementation.
// ============================================================
fn petgraph_kruskal(n: usize, edges: &EdgeArrays) -> (i64, usize) {
    let mut graph = UnGraph::<(), i32>::with_capacity(n, edges.len());

    // Add nodes
    for _ in 0..n {
        graph.add_node(());
    }

    // Add edges
    for i in 0..edges.len() {
        graph.add_edge(
            petgraph::graph::NodeIndex::new(edges.eu[i] as usize),
            petgraph::graph::NodeIndex::new(edges.ev[i] as usize),
            edges.ew[i],
        );
    }

    // Run petgraph's min_spanning_tree (Kruskal's)
    let mst: UnGraph<(), i32> = UnGraph::from_elements(min_spanning_tree(&graph));

    let mut total_weight: i64 = 0;
    let mut edge_count: usize = 0;
    for edge in mst.edge_indices() {
        total_weight += *mst.edge_weight(edge).unwrap() as i64;
        edge_count += 1;
    }

    (total_weight, edge_count)
}

// ============================================================
// Portable LCG PRNG (matches Python implementation exactly)
// Knuth's 64-bit LCG constants for cross-language reproducibility
// ============================================================
struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MULT: u64 = 6364136223846793005;
    const INC: u64 = 1442695040888963407;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(Self::INC);
        ((self.state >> 33) & 0x7FFFFFFF) as u32
    }

    fn randint(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }
}

// ============================================================
// SNAP Dataset Loader
// ============================================================
fn load_snap(
    path: &str,
    max_nodes: Option<usize>,
    weight_min: i32,
    weight_max: i32,
    seed: u64,
) -> (usize, EdgeArrays) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let reader = BufReader::new(file);

    let mut raw_edges: HashSet<(i32, i32)> = HashSet::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim().to_string();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let u: i32 = parts[0].parse().unwrap();
        let v: i32 = parts[1].parse().unwrap();
        if u == v {
            continue;
        }
        if let Some(max) = max_nodes {
            if u >= max as i32 || v >= max as i32 {
                continue;
            }
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        raw_edges.insert((a, b));
    }

    // Sort edges for deterministic weight assignment across languages
    let mut sorted_edges: Vec<(i32, i32)> = raw_edges.into_iter().collect();
    sorted_edges.sort_unstable();

    // Remap node IDs to 0..N-1
    let mut nodes: HashSet<i32> = HashSet::new();
    for &(u, v) in &sorted_edges {
        nodes.insert(u);
        nodes.insert(v);
    }
    let mut sorted_nodes: Vec<i32> = nodes.into_iter().collect();
    sorted_nodes.sort_unstable();
    let node_map: HashMap<i32, i32> = sorted_nodes
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new as i32))
        .collect();
    let n = node_map.len();

    let mut rng = LcgRng::new(seed);
    let mut eu = Vec::with_capacity(sorted_edges.len());
    let mut ev = Vec::with_capacity(sorted_edges.len());
    let mut ew = Vec::with_capacity(sorted_edges.len());
    for &(u, v) in &sorted_edges {
        eu.push(node_map[&u]);
        ev.push(node_map[&v]);
        ew.push(rng.randint(weight_min, weight_max));
    }

    (n, EdgeArrays { eu, ev, ew })
}

// ============================================================
// Result record
// ============================================================
#[derive(Clone)]
struct BenchResult {
    dataset: String,
    algorithm: String,
    n_vertices: usize,
    n_edges: usize,
    threads: usize,
    run: usize,
    time_s: f64,
    mst_weight: i64,
    median_s: f64,
    mean_s: f64,
    std_s: f64,
    min_s: f64,
    max_s: f64,
}

fn compute_stats(times: &[f64]) -> (f64, f64, f64, f64, f64) {
    let med = median(times);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std = variance.sqrt();
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (med, mean, std, min, max)
}

fn save_csv(results: &[BenchResult], path: &str) {
    let mut wtr = Writer::from_path(path).unwrap();
    wtr.write_record(["dataset","algorithm","n_vertices","n_edges","threads","run","time_s","mst_weight","median_s","mean_s","std_s","min_s","max_s"]).unwrap();
    for r in results {
        wtr.write_record([
            &r.dataset, &r.algorithm,
            &r.n_vertices.to_string(), &r.n_edges.to_string(),
            &r.threads.to_string(), &r.run.to_string(),
            &format!("{:.6}", r.time_s), &r.mst_weight.to_string(),
            &format!("{:.6}", r.median_s), &format!("{:.6}", r.mean_s),
            &format!("{:.6}", r.std_s), &format!("{:.6}", r.min_s),
            &format!("{:.6}", r.max_s),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("  Saved CSV: {path}");
}

fn median(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 { (v[n/2 - 1] + v[n/2]) / 2.0 } else { v[n/2] }
}

// ============================================================
// CLI
// ============================================================
#[derive(Parser)]
#[command(name = "mst-bench", about = "MST Benchmark (Rust + Rayon)")]
struct Cli {
    /// Path to SNAP edge-list file
    #[arg(long)]
    dataset: String,

    /// Comma-separated max_nodes values for scalability sweep
    #[arg(long, default_value = "5000,10000,25000,50000,100000")]
    sizes: String,

    /// Comma-separated thread counts for parallel sweep
    #[arg(long, default_value = "1,2,4,8,16")]
    threads: String,

    /// Min random edge weight
    #[arg(long, default_value_t = 1)]
    weight_min: i32,

    /// Max random edge weight
    #[arg(long, default_value_t = 1000)]
    weight_max: i32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Repetitions per measurement
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Output directory
    #[arg(long, default_value = "results/rust")]
    output_dir: String,

    /// Algorithms to run (kruskal, boruvka_seq, boruvka_par)
    #[arg(long, default_value = "kruskal,boruvka_seq,boruvka_par")]
    algorithms: String,

    /// Experiment: scalability, speedup, or both
    #[arg(long, default_value = "both")]
    experiment: String,

    /// Number of threads for parallel algorithms (overrides --threads for scalability)
    #[arg(long, default_value_t = 0)]
    num_threads: usize,
}

// ============================================================
// Main
// ============================================================
fn main() {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir).unwrap();

    let dataset_name = std::path::Path::new(&cli.dataset)
        .file_stem().unwrap().to_str().unwrap().to_string();
    let sizes: Vec<usize> = cli.sizes.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let thread_counts: Vec<usize> = cli.threads.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let algos: Vec<&str> = cli.algorithms.split(',').map(|s| s.trim()).collect();

    // Set up Rayon thread pool globally ONCE at startup.
    // --num-threads overrides the default (0 = use all available cores).
    let num_threads = if cli.num_threads > 0 { cli.num_threads } else {
        *thread_counts.last().unwrap_or(&0)
    };
    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to build Rayon thread pool");
    }
    println!("Rayon thread pool: {} threads", rayon::current_num_threads());

    if cli.experiment == "scalability" || cli.experiment == "both" {
        println!("{}", "=".repeat(65));
        println!("SCALABILITY EXPERIMENT: {dataset_name} (Rust)");
        println!("{}", "=".repeat(65));

        let mut results: Vec<BenchResult> = Vec::new();

        for &sz in &sizes {
            // sz=0 means "load full dataset"; any other value filters to vertex IDs < sz
            let max_n = if sz == 0 { None } else { Some(sz) };
            let label = if max_n.is_none() { "full".to_string() } else { format!("max_node={sz}") };
            println!("  Loading {dataset_name} ({label})...");
            let (n, edges) = load_snap(&cli.dataset, max_n, cli.weight_min, cli.weight_max, cli.seed);
            println!("    V={n}, E={}", edges.len());

            let mut ref_weight: Option<i64> = None;

            for &algo in &algos {
                let mut times = Vec::new();
                let mut mst_w: i64 = 0;

                // Thread pool already configured globally at startup via --num-threads

                for _r in 0..cli.runs {
                    let e = edges.clone();
                    let t0 = Instant::now();
                    let (w, _) = match algo {
                        "kruskal" => kruskal(n, &e),
                        "boruvka_seq" => boruvka_seq(n, &e),
                        "boruvka_par" => boruvka_par(n, &e),
                        "boruvka_pooled" => boruvka_pooled(n, &e),
                        "boruvka_groups" => boruvka_groups(n, &e),
                        "petgraph" => petgraph_kruskal(n, &e),
                        _ => panic!("Unknown algorithm: {algo}"),
                    };
                    let elapsed = t0.elapsed().as_secs_f64();
                    times.push(elapsed);
                    mst_w = w;
                }

                let (med, mean, std, min_t, max_t) = compute_stats(&times);

                // Push per-run records with aggregate stats
                for (r, &t) in times.iter().enumerate() {
                    results.push(BenchResult {
                        dataset: dataset_name.clone(),
                        algorithm: algo.to_string(),
                        n_vertices: n, n_edges: edges.len(),
                        threads: if algo.contains("par") || algo.contains("pooled") || algo.contains("groups") { rayon::current_num_threads() } else { 1 },
                        run: r, time_s: t, mst_weight: mst_w,
                        median_s: med, mean_s: mean, std_s: std, min_s: min_t, max_s: max_t,
                    });
                }

                if let Some(rw) = ref_weight {
                    assert_eq!(mst_w, rw, "WEIGHT MISMATCH: {algo}={mst_w}, expected={rw}");
                } else {
                    ref_weight = Some(mst_w);
                }

                let algo_label = match algo {
                    "kruskal" => "Kruskal (Seq)",
                    "boruvka_seq" => "Borůvka (Seq)",
                    "boruvka_par" => "Borůvka (Par)",
                    "boruvka_pooled" => "Borůvka (Pooled)",
                    "boruvka_groups" => "Borůvka (Groups)",
                    "petgraph" => "Petgraph (Kruskal)",
                    _ => algo,
                };
                println!("    {algo_label}: median={med:.6}s mean={mean:.6}s std={std:.6}s (MST weight={mst_w})");
            }
        }

        let csv_path = format!("{}/scalability_{dataset_name}.csv", cli.output_dir);
        save_csv(&results, &csv_path);
    }

    if cli.experiment == "speedup" || cli.experiment == "both" {
        println!("\n{}", "=".repeat(65));
        println!("PARALLEL SPEEDUP EXPERIMENT: {dataset_name} (Rust)");
        println!("{}", "=".repeat(65));

        let mut results: Vec<BenchResult> = Vec::new();

        for &sz in &sizes {
            let max_n = if sz == 0 { None } else { Some(sz) };
            let (n, edges) = load_snap(&cli.dataset, max_n, cli.weight_min, cli.weight_max, cli.seed);
            println!("  V={n}, E={}", edges.len());

            // Sequential baseline
            let mut seq_times = Vec::new();
            for _ in 0..cli.runs {
                let e = edges.clone();
                let t0 = Instant::now();
                boruvka_seq(n, &e);
                seq_times.push(t0.elapsed().as_secs_f64());
            }
            let seq_med = median(&seq_times);
            println!("    Borůvka-Seq baseline: {seq_med:.5}s");

            for &nt in &thread_counts {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
                let mut par_times = Vec::new();
                let mut last_w: i64 = 0;

                for _r in 0..cli.runs {
                    let e = edges.clone();
                    let t0 = Instant::now();
                    let (w, _) = pool.install(|| boruvka_par(n, &e));
                    let elapsed = t0.elapsed().as_secs_f64();
                    par_times.push(elapsed);
                    last_w = w;
                }

                let (med, mean, std, min_t, max_t) = compute_stats(&par_times);

                for (r, &t) in par_times.iter().enumerate() {
                    results.push(BenchResult {
                        dataset: dataset_name.clone(),
                        algorithm: format!("boruvka_par_t{nt}"),
                        n_vertices: n, n_edges: edges.len(),
                        threads: nt, run: r, time_s: t, mst_weight: last_w,
                        median_s: med, mean_s: mean, std_s: std, min_s: min_t, max_s: max_t,
                    });
                }

                let speedup = seq_med / med;
                let efficiency = speedup / nt as f64 * 100.0;
                println!("    Threads={nt:>2}: median={med:.6}s, Speedup={speedup:.3}x, Efficiency={efficiency:.1}%");
            }
        }

        let csv_path = format!("{}/speedup_{dataset_name}.csv", cli.output_dir);
        save_csv(&results, &csv_path);
    }

    println!("\nDone. Results in: {}", cli.output_dir);
}
