#!/usr/bin/env python3.12
"""
Verify generate_plots.py helpers compute correct values using random pandas DataFrames.
No files on disk — everything is in-memory with random data + expected values from pandas.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# ── Replicate the exact helpers from generate_plots.py ──
def select_rows(df, lang=None, ds_key=None, algo=None):
    result = df
    if lang:   result = result[result['lang'] == lang]
    if ds_key: result = result[result['ds_key'] == ds_key]
    if algo:   result = result[result['algorithm'] == algo]
    return result

def grouped_time_stats(df):
    if df.empty: return [], [], [], []
    agg = df.groupby('n_vertices')['time_s'].agg(['mean', 'min', 'max']).sort_index()
    return agg.index.tolist(), agg['mean'].tolist(), agg['min'].tolist(), agg['max'].tolist()

def get_speedup_baseline(df):
    seq_rows = df[df['algorithm'] == 'boruvka_seq']
    par_rows = df[df['algorithm'] == 'boruvka_par']
    if not seq_rows.empty and not par_rows.empty:
        return par_rows, lambda vc: seq_rows.loc[seq_rows['n_vertices'] == vc, 'time_s'].mean() or None
    def min_thread_baseline(vc):
        size_rows = df[df['n_vertices'] == vc]
        if size_rows.empty: return None
        return size_rows.loc[size_rows['threads'] == size_rows['threads'].min(), 'time_s'].mean()
    return df, min_thread_baseline

# ── Generate random scalability DataFrame ──
VERTEX_COUNTS = [1000, 5000, 10000]
RUNS_PER_CONFIG = 5
LANGUAGES = ['python', 'rust', 'cpp']
ALGORITHMS = ['kruskal', 'boruvka_seq']

scal_records = []
for lang in LANGUAGES:
    for algo in ALGORITHMS:
        for vertex_count in VERTEX_COUNTS:
            times = np.random.uniform(0.01, 10.0, size=RUNS_PER_CONFIG)
            for time_val in times:
                scal_records.append({
                    'dataset': 'random_ds', 'algorithm': algo,
                    'n_vertices': vertex_count, 'n_edges': vertex_count * 5,
                    'mst_weight': 99999, 'time_s': time_val, 'threads': 1,
                    'lang': lang, 'ds_key': 'test-dataset',
                })

df_scalability = pd.DataFrame(scal_records)

# ── Generate random speedup DataFrames ──
THREAD_COUNTS = [2, 4, 8, 16]

# Rust/Python format: all rows are boruvka_par with varying threads
rust_speed_records = []
for vertex_count in VERTEX_COUNTS:
    for thread_count in THREAD_COUNTS:
        times = np.random.uniform(0.1, 5.0, size=RUNS_PER_CONFIG)
        for time_val in times:
            rust_speed_records.append({
                'dataset': 'random_ds', 'algorithm': f'boruvka_par_t{thread_count}',
                'n_vertices': vertex_count, 'n_edges': vertex_count * 5,
                'mst_weight': 99999, 'time_s': time_val, 'threads': thread_count,
                'lang': 'rust', 'ds_key': 'test-dataset',
            })
df_rust_speed = pd.DataFrame(rust_speed_records)

# C++ format: boruvka_seq (T=1) + boruvka_par (T=2,4,8,16)
cpp_speed_records = []
for vertex_count in VERTEX_COUNTS:
    seq_times = np.random.uniform(1.0, 10.0, size=RUNS_PER_CONFIG)
    for time_val in seq_times:
        cpp_speed_records.append({
            'dataset': 'random_ds', 'algorithm': 'boruvka_seq',
            'n_vertices': vertex_count, 'n_edges': vertex_count * 5,
            'mst_weight': 99999, 'time_s': time_val, 'threads': 1,
            'lang': 'cpp', 'ds_key': 'test-dataset',
        })
    for thread_count in THREAD_COUNTS:
        par_times = np.random.uniform(0.1, 5.0, size=RUNS_PER_CONFIG)
        for time_val in par_times:
            cpp_speed_records.append({
                'dataset': 'random_ds', 'algorithm': 'boruvka_par',
                'n_vertices': vertex_count, 'n_edges': vertex_count * 5,
                'mst_weight': 99999, 'time_s': time_val, 'threads': thread_count,
                'lang': 'cpp', 'ds_key': 'test-dataset',
            })
df_cpp_speed = pd.DataFrame(cpp_speed_records)

# ====================================================================
# TESTS
# ====================================================================
print("=== TEST 1: grouped_time_stats matches raw pandas groupby ===")
for lang in LANGUAGES:
    for algo in ALGORITHMS:
        subset = select_rows(df_scalability, lang=lang, algo=algo)
        vertex_counts, avg_times, min_times, max_times = grouped_time_stats(subset)

        # Compute expected directly from pandas
        expected = subset.groupby('n_vertices')['time_s'].agg(['mean', 'min', 'max']).sort_index()

        assert vertex_counts == expected.index.tolist(), f"Vertex counts mismatch for {lang}/{algo}"
        for i, vc in enumerate(vertex_counts):
            assert abs(avg_times[i] - expected.loc[vc, 'mean']) < 1e-12, \
                f"Avg mismatch {lang}/{algo} V={vc}: {avg_times[i]} != {expected.loc[vc, 'mean']}"
            assert abs(min_times[i] - expected.loc[vc, 'min']) < 1e-12, \
                f"Min mismatch {lang}/{algo} V={vc}"
            assert abs(max_times[i] - expected.loc[vc, 'max']) < 1e-12, \
                f"Max mismatch {lang}/{algo} V={vc}"
        print(f"  ✓ {lang}/{algo}: avg/min/max correct for {len(vertex_counts)} vertex sizes")

print("\n=== TEST 2: select_rows isolates data correctly ===")
for lang in LANGUAGES:
    selected = select_rows(df_scalability, lang=lang)
    assert (selected['lang'] == lang).all(), f"Lang filter leaked: {selected['lang'].unique()}"
    expected_count = len(ALGORITHMS) * len(VERTEX_COUNTS) * RUNS_PER_CONFIG
    assert len(selected) == expected_count, f"{lang}: got {len(selected)} rows, expected {expected_count}"
    print(f"  ✓ {lang}: {len(selected)} rows, no cross-language contamination")

for algo in ALGORITHMS:
    selected = select_rows(df_scalability, algo=algo)
    assert (selected['algorithm'] == algo).all(), f"Algo filter leaked"
    print(f"  ✓ {algo}: {len(selected)} rows, no cross-algo contamination")

print("\n=== TEST 3: Rust speedup baseline (min-thread format) ===")
for vertex_count in VERTEX_COUNTS:
    parallel_data, baseline_fn = get_speedup_baseline(df_rust_speed)
    baseline_time = baseline_fn(vertex_count)

    # Expected: average time at the minimum thread count for this vertex count
    vc_data = df_rust_speed[df_rust_speed['n_vertices'] == vertex_count]
    min_threads = vc_data['threads'].min()
    expected_baseline = vc_data.loc[vc_data['threads'] == min_threads, 'time_s'].mean()
    assert abs(baseline_time - expected_baseline) < 1e-12, \
        f"Rust baseline V={vertex_count}: {baseline_time} != {expected_baseline}"

    # Check speedup at each thread count
    for thread_count in THREAD_COUNTS:
        thread_avg = vc_data.loc[vc_data['threads'] == thread_count, 'time_s'].mean()
        expected_speedup = expected_baseline / thread_avg
        expected_efficiency = expected_speedup / thread_count * 100

        # Recompute how generate_plots.py would
        computed_speedup = baseline_time / parallel_data.loc[
            (parallel_data['n_vertices'] == vertex_count) &
            (parallel_data['threads'] == thread_count), 'time_s'].mean()

        assert abs(computed_speedup - expected_speedup) < 1e-12, \
            f"Rust speedup V={vertex_count} T={thread_count}: {computed_speedup} != {expected_speedup}"

    print(f"  ✓ V={vertex_count}: baseline={baseline_time:.4f}s (T={min_threads}), all thread speedups correct")

print("\n=== TEST 4: C++ speedup baseline (boruvka_seq format) ===")
for vertex_count in VERTEX_COUNTS:
    parallel_data, baseline_fn = get_speedup_baseline(df_cpp_speed)
    baseline_time = baseline_fn(vertex_count)

    # Expected: average of boruvka_seq times at this vertex count
    seq_data = df_cpp_speed[(df_cpp_speed['algorithm'] == 'boruvka_seq') &
                            (df_cpp_speed['n_vertices'] == vertex_count)]
    expected_baseline = seq_data['time_s'].mean()
    assert abs(baseline_time - expected_baseline) < 1e-12, \
        f"C++ baseline V={vertex_count}: {baseline_time} != {expected_baseline}"

    # Parallel data should only contain boruvka_par rows
    assert (parallel_data['algorithm'] == 'boruvka_par').all(), "C++ parallel_data contains non-par rows"

    for thread_count in THREAD_COUNTS:
        par_data = parallel_data[(parallel_data['n_vertices'] == vertex_count) &
                                 (parallel_data['threads'] == thread_count)]
        thread_avg = par_data['time_s'].mean()
        expected_speedup = expected_baseline / thread_avg
        expected_efficiency = expected_speedup / thread_count * 100

        computed_speedup = baseline_time / thread_avg
        assert abs(computed_speedup - expected_speedup) < 1e-12, \
            f"C++ speedup V={vertex_count} T={thread_count}: {computed_speedup} != {expected_speedup}"

    print(f"  ✓ V={vertex_count}: baseline={baseline_time:.4f}s (seq), all thread speedups correct")

print("\n=== TEST 5: Cross-language ratio computation ===")
for algo in ALGORITHMS:
    py_sizes, py_avgs, _, _ = grouped_time_stats(select_rows(df_scalability, lang='python', algo=algo))
    rs_sizes, rs_avgs, _, _ = grouped_time_stats(select_rows(df_scalability, lang='rust', algo=algo))

    py_map = dict(zip(py_sizes, py_avgs))
    rs_map = dict(zip(rs_sizes, rs_avgs))
    common_sizes = sorted(set(py_sizes) & set(rs_sizes))

    for vertex_count in common_sizes:
        # Expected ratio from raw pandas
        py_raw_avg = df_scalability[(df_scalability['lang'] == 'python') &
                                    (df_scalability['algorithm'] == algo) &
                                    (df_scalability['n_vertices'] == vertex_count)]['time_s'].mean()
        rs_raw_avg = df_scalability[(df_scalability['lang'] == 'rust') &
                                    (df_scalability['algorithm'] == algo) &
                                    (df_scalability['n_vertices'] == vertex_count)]['time_s'].mean()
        expected_ratio = py_raw_avg / rs_raw_avg
        computed_ratio = py_map[vertex_count] / rs_map[vertex_count]
        assert abs(computed_ratio - expected_ratio) < 1e-12, \
            f"Ratio {algo} V={vertex_count}: {computed_ratio} != {expected_ratio}"

    print(f"  ✓ {algo}: all {len(common_sizes)} ratios match raw pandas computation")

print("\n=== TEST 6: Cross-language vertex count mismatch detection ===")
# Simulate what the old bug was: Python has max V=500K, Rust/C++ have max V=3M
# A naive "each lang's own max" comparison would compare different graph sizes
mismatched_records = []
for time_val in np.random.uniform(1.0, 5.0, size=3):
    mismatched_records.append({
        'dataset': 'test_ds', 'algorithm': 'kruskal',
        'n_vertices': 500000, 'n_edges': 2000000,
        'mst_weight': 99999, 'time_s': time_val, 'threads': 1,
        'lang': 'python', 'ds_key': 'test-dataset',
    })
for time_val in np.random.uniform(5.0, 15.0, size=3):
    mismatched_records.append({
        'dataset': 'test_ds', 'algorithm': 'kruskal',
        'n_vertices': 3000000, 'n_edges': 100000000,
        'mst_weight': 99999, 'time_s': time_val, 'threads': 1,
        'lang': 'rust', 'ds_key': 'test-dataset',
    })
df_mismatched = pd.DataFrame(mismatched_records)

# The max vertex counts should differ
py_max = select_rows(df_mismatched, lang='python')['n_vertices'].max()
rs_max = select_rows(df_mismatched, lang='rust')['n_vertices'].max()
assert py_max != rs_max, "Test setup error: vertex counts should differ"
assert py_max == 500000 and rs_max == 3000000, f"Unexpected maxes: {py_max}, {rs_max}"

# A correct comparison would only compare at V=500K (not mix V=500K Python vs V=3M Rust)
# The old bug: comparing Python avg at V=500K with Rust avg at V=3M (totally different sizes!)
py_avg_at_max = df_mismatched[(df_mismatched['lang'] == 'python') &
                              (df_mismatched['n_vertices'] == py_max)]['time_s'].mean()
rs_avg_at_max = df_mismatched[(df_mismatched['lang'] == 'rust') &
                              (df_mismatched['n_vertices'] == rs_max)]['time_s'].mean()
# Python would look faster because it's running on a SMALLER graph
assert py_avg_at_max < rs_avg_at_max, "Test expects Python faster on smaller graph"
print(f"  ✓ Detected mismatch: Python max V={py_max:,} vs Rust max V={rs_max:,}")
print(f"    Python avg={py_avg_at_max:.2f}s (V=500K) vs Rust avg={rs_avg_at_max:.2f}s (V=3M)")
print(f"    Comparing these would be WRONG — different graph sizes!")

# Verify that with matching vertex counts, comparison is fair
common_v = sorted(set(select_rows(df_mismatched, lang='python')['n_vertices'].unique()) &
                  set(select_rows(df_mismatched, lang='rust')['n_vertices'].unique()))
assert len(common_v) == 0, "No common vertex counts in this mismatched scenario"
print(f"  ✓ No common vertex counts → comparison should be skipped or warned")

print(f"\n{'=' * 50}")
print("ALL TESTS PASSED ✓")
