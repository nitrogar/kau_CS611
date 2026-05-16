#!/usr/bin/env python3.12
"""Generate plots for CS611 MST benchmarks.  Reads from logs/, outputs to report/figures/."""
import os, csv, glob, argparse, platform
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI & paths ──
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', default=None)
parser.add_argument('--no-errorbars', action='store_true')
args = parser.parse_args()
SHOW_ERRORBARS = not args.no_errorbars

if args.run_id:
    RUN_DIR = os.path.join('logs', args.run_id)
else:
    runs = sorted(glob.glob('logs/????-??-??_??-??-??'))
    RUN_DIR = runs[-1] if runs else None
    print(f"Auto-selected latest run: {RUN_DIR}" if RUN_DIR else "No run dirs found")

FIGURES_DIR = os.path.join(RUN_DIR or 'logs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
REPORT_FIGURES_DIR = 'report/figures'
os.makedirs(REPORT_FIGURES_DIR, exist_ok=True)

HARDWARE_LABEL = 'AMD Ryzen 9 3900X · 12 cores / 24 threads'
PHYS_CORES = 12

# ── Matplotlib style ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.labelsize': 11, 'legend.fontsize': 8.5,
    'legend.framealpha': 0.9, 'figure.dpi': 200, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'lines.linewidth': 2.0,
    'lines.markersize': 6, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

# ── Algorithm display config ──
ALGOS = {  # key → (label, color, marker, linestyle)
    'kruskal':        ('Kruskal (Seq)',     '#1565C0', 'o', '-'),
    'boruvka_seq':    ('Borůvka (Seq)',     '#E65100', 's', '-'),
    'boruvka_par':    ('Borůvka (Par)',     '#2E7D32', '^', '-'),
    'boruvka_seq_nc': ('Borůvka (Seq, NC)','#FF6F00', 'D', '--'),
    'boruvka_par_nc': ('Borůvka (Par, NC)','#66BB6A', 'v', '--'),
    'boruvka_par_fr': ('Borůvka (Par, FR)','#C62828', 'h', '-.'),
    'boruvka_pooled': ('Borůvka (Pooled)', '#6A1B9A', 'P', '-.'),
    'boruvka_groups': ('Borůvka (Groups)', '#00838F', 'X', ':'),
}
def algo_label(algo):     return ALGOS.get(algo, (algo,))[0]
def algo_marker(algo):    return ALGOS.get(algo, (None, '', 'o'))[2]
def algo_linestyle(algo): return ALGOS.get(algo, (None, '', '', '-'))[3]
def algo_color(algo):     return ALGOS.get(algo, (None, '#888'))[1]

# ── Language / dataset config ──
LANGS = {  # key → (display_label, color, linewidth, alpha)
    'python': ('Python', '#1565C0', 1.6, 0.7),
    'rust':   ('Rust',   '#E65100', 2.2, 1.0),
    'cpp':    ('C++',    '#2E7D32', 1.8, 0.8),
}
DATASETS = [  # (key, display_name, scalability_csv, speedup_csv, max_edges, description)
    ('roadNet-CA', 'roadNet-CA', 'scalability_roadNet-CA.csv',
     'speedup_roadNet-CA.csv', 273266, 'sparse planar · avg deg 3.1'),
    ('amazon0302', 'amazon0302', 'scalability_amazon0302.csv',
     'speedup_amazon0302.csv', 899792, 'power-law · avg deg 6.1'),
    ('com-orkut',  'com-Orkut',  'scalability_com-orkut.ungraph.csv',
     'speedup_com-orkut.ungraph.csv', 117185083, 'social network · avg deg 76.3'),
]

def lang_label(lang): return LANGS[lang][0]
def lang_color(lang): return LANGS[lang][1]

# ── Data loading ──
def find_csv(lang, dataset_key, filename):
    for base_dir in [RUN_DIR, 'logs']:
        if base_dir:
            path = os.path.join(base_dir, lang, dataset_key, filename)
            if os.path.exists(path): return path
    return os.path.join(RUN_DIR or 'logs', lang, dataset_key, filename)

def load_csv(path, lang, dataset_key):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['n_vertices', 'n_edges', 'mst_weight']:
        if col in df.columns: df[col] = df[col].astype(np.int64)
    df['time_s'] = df['time_s'].astype(float)
    if 'threads' not in df.columns: df['threads'] = 1
    df['threads'] = df['threads'].astype(int)
    df['lang'], df['ds_key'] = lang, dataset_key
    return df

scalability_parts, speedup_parts = [], []
for lang in LANGS:
    for dataset_key, _, scalability_file, speedup_file, *_ in DATASETS:
        for part_list, csv_file in [(scalability_parts, scalability_file), (speedup_parts, speedup_file)]:
            df = load_csv(find_csv(lang, dataset_key, csv_file), lang, dataset_key)
            if not df.empty: part_list.append(df)

df_scalability = pd.concat(scalability_parts, ignore_index=True) if scalability_parts else pd.DataFrame()
df_speedup     = pd.concat(speedup_parts, ignore_index=True) if speedup_parts else pd.DataFrame()

# Validation data (only used for validation table plot)
def load_validation(path):
    if not os.path.exists(path): return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row.update({k: int(row[k]) for k in ['n_vertices', 'n_edges', 'our_weight', 'networkx_weight']})
            row['match'] = row['match'] == 'True'
            rows.append(row)
    return rows
validation_data = load_validation(find_csv('python', 'roadNet-CA', 'validation_roadNet-CA.csv')) + \
                  load_validation(find_csv('python', 'amazon0302', 'validation_amazon0302.csv'))

# ── Helpers ──
def select_rows(df, lang=None, ds_key=None, algo=None):
    """Filter DataFrame by language, dataset key, and/or algorithm."""
    result = df
    if lang:   result = result[result['lang'] == lang]
    if ds_key: result = result[result['ds_key'] == ds_key]
    if algo:   result = result[result['algorithm'] == algo]
    return result

def grouped_time_stats(df):
    """Group by n_vertices → (vertex_counts, avg_times, min_times, max_times)."""
    if df.empty: return [], [], [], []
    agg = df.groupby('n_vertices')['time_s'].agg(['mean', 'min', 'max']).sort_index()
    return agg.index.tolist(), agg['mean'].tolist(), agg['min'].tolist(), agg['max'].tolist()

def save_figure(fig, filename):
    for output_dir in [FIGURES_DIR, REPORT_FIGURES_DIR]:
        fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")

def add_hardware_footer(fig, extra=''):
    text = f"{HARDWARE_LABEL}  ·  {extra}" if extra else HARDWARE_LABEL
    fig.text(0.99, 0.005, text, ha='right', va='bottom', fontsize=7, color='#999', style='italic')

def add_info_box(ax, lines, loc='upper left'):
    x_pos = 0.02 if 'left' in loc else 0.98
    y_pos = 0.98 if 'upper' in loc else 0.02
    ax.text(x_pos, y_pos, '\n'.join(lines), transform=ax.transAxes, fontsize=7,
            va='top' if 'upper' in loc else 'bottom',
            ha='left' if 'left' in loc else 'right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))

for label, df in [('scalability', df_scalability), ('speedup', df_speedup)]:
    if not df.empty:
        print(f"  {label}: {dict(df.groupby('lang').size())} rows")

# ====================================================================
# PLOT 1: Scalability (one per dataset, all languages overlaid)
# ====================================================================
for dataset_key, dataset_name, _, _, max_edges, description in DATASETS:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for lang in LANGS:
        lang_data = select_rows(df_scalability, lang=lang, ds_key=dataset_key)
        if lang_data.empty: continue
        _, _, linewidth, alpha = LANGS[lang]
        for algo in lang_data['algorithm'].unique():
            vertex_counts, avg_times, min_times, max_times = grouped_time_stats(lang_data[lang_data['algorithm'] == algo])
            if not vertex_counts: continue
            x_vertices = [v / 1000 for v in vertex_counts]
            plot_kwargs = dict(marker=algo_marker(algo), ls=algo_linestyle(algo), color=lang_color(lang),
                               lw=linewidth, ms=5, alpha=alpha, label=f'{algo_label(algo)} ({lang_label(lang)})')
            if SHOW_ERRORBARS:
                err_lo = [avg - lo for avg, lo in zip(avg_times, min_times)]
                err_hi = [hi - avg for avg, hi in zip(avg_times, max_times)]
                ax.errorbar(x_vertices, avg_times, yerr=[err_lo, err_hi],
                            fmt=plot_kwargs.pop('marker'), capsize=3, capthick=1, **plot_kwargs)
            else:
                ax.plot(x_vertices, avg_times, **plot_kwargs)
    ax.set(xlabel='Vertices (×1000)', ylabel='Time (seconds)')
    ax.set_title(f'{dataset_name} — Scalability', fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper left', ncol=2)
    add_info_box(ax, [f'{dataset_name} · {description}', f'Max edges: {max_edges:,}'], loc='center left')
    add_hardware_footer(fig, 'average of 3 runs')
    plt.tight_layout(); save_figure(fig, f'scalability_{dataset_key}.png'); plt.close()

# ====================================================================
# PLOT 2 & 3: Speedup + Efficiency (3×3 grid each)
# ====================================================================
def get_speedup_baseline(df):
    """Return (parallel_df, baseline_time_fn). C++ has boruvka_seq; Rust/Python use min-thread."""
    seq_rows = df[df['algorithm'] == 'boruvka_seq']
    par_rows = df[df['algorithm'] == 'boruvka_par']
    if not seq_rows.empty and not par_rows.empty:
        return par_rows, lambda vertex_count: seq_rows.loc[seq_rows['n_vertices'] == vertex_count, 'time_s'].mean() or None
    def min_thread_baseline(vertex_count):
        size_rows = df[df['n_vertices'] == vertex_count]
        if size_rows.empty: return None
        min_threads = size_rows['threads'].min()
        return size_rows.loc[size_rows['threads'] == min_threads, 'time_s'].mean()
    return df, min_thread_baseline

for mode in ['speedup', 'efficiency']:
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    grid_cells = [(lang, ds[0]) for lang in LANGS for ds in DATASETS]
    for ax, (lang, dataset_key) in zip(axes.flat, grid_cells):
        speed_data = select_rows(df_speedup, lang=lang, ds_key=dataset_key)
        dataset_name = next(ds[1] for ds in DATASETS if ds[0] == dataset_key)
        subplot_title = f"{lang_label(lang)} — {dataset_name}"
        if speed_data.empty:
            ax.set_title(f'{subplot_title} — no data'); continue

        parallel_data, baseline_fn = get_speedup_baseline(speed_data)
        colormap = plt.cm.Dark2
        unique_sizes = sorted(parallel_data['n_vertices'].unique())
        for size_idx, vertex_count in enumerate(unique_sizes):
            size_data = parallel_data[parallel_data['n_vertices'] == vertex_count]
            thread_counts = sorted(size_data['threads'].unique())
            baseline_time = baseline_fn(vertex_count)
            if not baseline_time or baseline_time <= 0: continue
            plot_values = []
            for thread_count in thread_counts:
                speedup = baseline_time / size_data.loc[size_data['threads'] == thread_count, 'time_s'].mean()
                plot_values.append(speedup if mode == 'speedup' else speedup / thread_count * 100)
            line_color = colormap(size_idx / max(len(unique_sizes) - 1, 1))
            ax.plot(thread_counts, plot_values, 'o-', color=line_color, label=f'V={vertex_count // 1000}K')
            if mode == 'speedup':
                peak_idx = np.argmax(plot_values)
                ax.annotate(f'{plot_values[peak_idx]:.2f}×', (thread_counts[peak_idx], plot_values[peak_idx]),
                            textcoords='offset points', xytext=(5, 8),
                            fontsize=7, fontweight='bold', color=line_color)

        ax.axhline(y=1 if mode == 'speedup' else 100, color='gray', ls='--', alpha=0.4)
        ax.axvline(x=PHYS_CORES, color='#E91E63', ls=':', alpha=0.4)
        ax.set(xlabel='Threads', ylabel='Speedup (×)' if mode == 'speedup' else 'Efficiency (%)')
        if mode == 'efficiency': ax.set_ylim(bottom=0)
        ax.set_title(subplot_title, fontweight='bold', fontsize=11)
        ax.legend(fontsize=7, loc='upper left' if mode == 'speedup' else 'upper right')

    plt.suptitle('Parallel Speedup' if mode == 'speedup' else 'Parallel Efficiency = Speedup/Threads × 100%',
                 fontsize=14, fontweight='bold', y=1.01)
    add_hardware_footer(fig, 'Borůvka-Par baseline')
    plt.tight_layout(); save_figure(fig, f'parallel_{mode}.png'); plt.close()

# ====================================================================
# PLOT 4: Cross-language comparison (1×3)
# ====================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (dataset_key, dataset_name, _, _, max_edges, _) in zip(axes, DATASETS):
    for algo in ['kruskal', 'boruvka_seq']:
        for lang in LANGS:
            vertex_counts, avg_times, _, _ = grouped_time_stats(select_rows(df_scalability, lang=lang, ds_key=dataset_key, algo=algo))
            if vertex_counts:
                ax.plot([v / 1000 for v in vertex_counts], avg_times, marker=algo_marker(algo), ls=algo_linestyle(algo),
                        color=lang_color(lang), lw=LANGS[lang][2], ms=5, alpha=LANGS[lang][3],
                        label=f'{algo_label(algo)} ({lang_label(lang)})')
    # C++ boruvka_par
    cpp_par_data = select_rows(df_scalability, lang='cpp', ds_key=dataset_key, algo='boruvka_par')
    if not cpp_par_data.empty:
        vertex_counts, avg_times, _, _ = grouped_time_stats(cpp_par_data)
        ax.plot([v / 1000 for v in vertex_counts], avg_times, marker=algo_marker('boruvka_par'),
                ls=algo_linestyle('boruvka_par'), color=lang_color('cpp'), lw=1.8, ms=5, alpha=0.8,
                label=f'{algo_label("boruvka_par")} (C++)')
    ax.set(xlabel='Vertices (×1000)', ylabel='Time (seconds)')
    ax.set_title(dataset_name, fontweight='bold'); ax.legend(fontsize=6.5)
    add_info_box(ax, [f'Edges: {max_edges:,}'])
plt.suptitle('Python vs Rust vs C++', fontsize=14, fontweight='bold', y=1.01)
add_hardware_footer(fig); plt.tight_layout(); save_figure(fig, 'python_vs_rust_comparison.png'); plt.close()

# ====================================================================
# PLOT 5: Rust/Python speedup ratio (1×3)
# ====================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (dataset_key, dataset_name, *_) in zip(axes, DATASETS):
    for algo in ['kruskal', 'boruvka_seq']:
        py_sizes, py_avgs, _, _ = grouped_time_stats(select_rows(df_scalability, lang='python', ds_key=dataset_key, algo=algo))
        rs_sizes, rs_avgs, _, _ = grouped_time_stats(select_rows(df_scalability, lang='rust', ds_key=dataset_key, algo=algo))
        common_sizes = sorted(set(py_sizes) & set(rs_sizes))
        if not common_sizes: continue
        py_time_map = dict(zip(py_sizes, py_avgs))
        rs_time_map = dict(zip(rs_sizes, rs_avgs))
        ratios = [py_time_map[size] / rs_time_map[size] for size in common_sizes if rs_time_map[size] > 0]
        if not ratios: continue
        ax.plot([size / 1000 for size in common_sizes[:len(ratios)]], ratios,
                marker=algo_marker(algo), ls=algo_linestyle(algo), color=algo_color(algo),
                lw=2.2, ms=7, label=algo_label(algo))
        ax.annotate(f'avg {np.mean(ratios):.1f}×', (common_sizes[-1] / 1000, ratios[-1]),
                    textcoords='offset points', xytext=(5, 5), fontsize=7.5,
                    color=algo_color(algo), fontweight='bold')
    ax.axhline(y=1, color='gray', ls='--', alpha=0.5)
    ax.set(xlabel='Vertices (×1000)', ylabel='Python / Rust time')
    ax.set_title(dataset_name, fontweight='bold'); ax.legend()
    add_info_box(ax, ['> 1 → Rust faster'], loc='lower right')
plt.suptitle('Rust vs Python Speedup Ratio', fontsize=14, fontweight='bold', y=1.01)
add_hardware_footer(fig); plt.tight_layout(); save_figure(fig, 'rust_over_python_ratio.png'); plt.close()

# ====================================================================
# PLOT 6: Validation table
# ====================================================================
if validation_data:
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(validation_data) + 1.5)))
    ax.axis('off')
    headers = ['Dataset', 'Vertices', 'Edges', 'Our Weight', 'NetworkX Weight', 'Status']
    table_rows = [[v['dataset'], f"{v['n_vertices']:,}", f"{v['n_edges']:,}",
                    f"{v['our_weight']:,}", f"{v['networkx_weight']:,}",
                    '✓ MATCH' if v['match'] else '✗ MISMATCH'] for v in validation_data]
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.6)
    for col_idx in range(len(headers)):
        table[0, col_idx].set_facecolor('#1565C0')
        table[0, col_idx].set_text_props(color='white', fontweight='bold')
    for row_idx, row in enumerate(table_rows):
        status_cell = table[row_idx + 1, 5]
        is_match = '✓' in row[5]
        status_cell.set_facecolor('#E8F5E9' if is_match else '#FFEBEE')
        status_cell.set_text_props(color='#2E7D32' if is_match else '#C62828', fontweight='bold')
    ax.set_title('Validation Summary', fontsize=14, fontweight='bold', pad=20)
    all_passed = all(v['match'] for v in validation_data)
    fig.text(0.5, 0.02, f"ALL {len(validation_data)} TESTS PASSED ✓" if all_passed else "SOME FAILED ✗",
             ha='center', fontsize=12, fontweight='bold', color='#2E7D32' if all_passed else '#C62828')
    add_hardware_footer(fig); plt.tight_layout(); save_figure(fig, 'validation_summary.png'); plt.close()

# ====================================================================
# Summary table (console + LaTeX)
# ====================================================================
if not df_scalability.empty:
    LANG_DISPLAY = {lang: LANGS[lang][0] for lang in LANGS}
    DATASET_ORDER = [(ds[0], ds[1]) for ds in DATASETS]
    ALGOS_ORDER = ['kruskal', 'boruvka_seq', 'boruvka_seq_nc', 'boruvka_par', 'boruvka_par_nc']

    print(f"\n  CROSS-LANGUAGE COMPARISON (max graph size)\n")
    for dataset_key, dataset_name in DATASET_ORDER:
        for algo in ALGOS_ORDER:
            time_by_lang, max_vertices = {}, 0
            for lang in LANGS:
                sub = select_rows(df_scalability, lang=lang, ds_key=dataset_key, algo=algo)
                if sub.empty: continue
                max_v = sub['n_vertices'].max()
                time_by_lang[LANG_DISPLAY[lang]] = sub.loc[sub['n_vertices'] == max_v, 'time_s'].mean()
                max_vertices = max(max_vertices, max_v)
            if not time_by_lang: continue
            fastest_lang = min(time_by_lang, key=time_by_lang.get)
            parts = [f"{LANG_DISPLAY[lang]}={time_by_lang.get(LANG_DISPLAY[lang], 0):.4f}{'*' if LANG_DISPLAY[lang] == fastest_lang else ''}"
                     for lang in LANGS if LANG_DISPLAY[lang] in time_by_lang]
            print(f"    {dataset_name:12s} {algo_label(algo):20s} V={max_vertices:>10,}  {' | '.join(parts)}")

    # LaTeX table
    latex_path = os.path.join(FIGURES_DIR, 'benchmark_summary.tex')
    with open(latex_path, 'w') as tex_file:
        tex_file.write("% Auto-generated\n\\begin{table}[H]\\centering\\small\n")
        tex_file.write("\\caption{Performance at max graph size. \\textbf{Bold}=fastest.}\n")
        tex_file.write("\\label{tab:benchmark_summary}\n\\begin{tabular}{llrrrr}\\toprule\n")
        tex_file.write("Dataset & Algorithm & V & Python & Rust & C++ \\\\\\midrule\n")
        for dataset_key, dataset_name in DATASET_ORDER:
            first_row = True
            for algo in ALGOS_ORDER:
                time_by_lang, max_vertices = {}, 0
                for lang in LANGS:
                    sub = select_rows(df_scalability, lang=lang, ds_key=dataset_key, algo=algo)
                    if sub.empty: continue
                    max_v = sub['n_vertices'].max()
                    time_by_lang[LANG_DISPLAY[lang]] = sub.loc[sub['n_vertices'] == max_v, 'time_s'].mean()
                    max_vertices = max(max_vertices, max_v)
                if not time_by_lang: continue
                fastest_lang = min(time_by_lang, key=time_by_lang.get)
                def format_cell(lang_name):
                    if lang_name not in time_by_lang: return "—"
                    val = f"{time_by_lang[lang_name]:.4f}"
                    return f"\\textbf{{{val}}}" if lang_name == fastest_lang else val
                tex_file.write(f"{dataset_name if first_row else ''} & {algo_label(algo)} & {max_vertices:,} & "
                               f"{format_cell('Python')} & {format_cell('Rust')} & {format_cell('C++')} \\\\\n")
                first_row = False
            tex_file.write("\\midrule\n")
        tex_file.seek(tex_file.tell() - len("\\midrule\n"))
        tex_file.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX → {latex_path}")

# ── Final summary ──
print(f"\n{'=' * 60}\nAll figures → {FIGURES_DIR}/\n{'=' * 60}")
for filename in ['scalability_roadNet-CA.png', 'scalability_amazon0302.png', 'scalability_com-orkut.png',
                  'parallel_speedup.png', 'parallel_efficiency.png', 'python_vs_rust_comparison.png',
                  'rust_over_python_ratio.png', 'validation_summary.png', 'benchmark_summary.tex']:
    for output_dir in [FIGURES_DIR, REPORT_FIGURES_DIR]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {output_dir}/{filename} ({os.path.getsize(filepath) / 1024:.0f} KB)")
        else:
            print(f"  ✗ {output_dir}/{filename} MISSING")
