#!/usr/bin/env python3
"""
Unified runner for MST benchmarks (Python + Rust).

Usage:
  python3 main.py                        # Run both Python & Rust on both datasets
  python3 main.py --lang python          # Python only
  python3 main.py --lang rust            # Rust only
  python3 main.py --experiment speedup   # Parallel speedup only
"""
import argparse
import subprocess
import sys
import os

DATASETS = {
    'road': 'datasets/roadNet-CA.txt',
    'amazon': 'datasets/amazon0302.txt',
}

# Comprehensive size sweeps for richer data
ROAD_SIZES = '1000,2500,5000,10000,25000,50000,100000,200000'
AMAZON_SIZES = '1000,2500,5000,10000,25000,50000,100000,262111'

# Speedup experiment sizes (use large graphs only)
ROAD_SPEEDUP_SIZES = '100000,200000'
AMAZON_SPEEDUP_SIZES = '100000,262111'


def run_python(dataset_path, sizes, experiment, extra_args):
    cmd = [
        'python3.12', 'mst_python.py',
        '--dataset', dataset_path,
        '--sizes', sizes,
        '--experiment', experiment,
    ] + extra_args
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_rust(dataset_path, sizes, experiment, extra_args):
    # Build first
    print("\n>>> RUSTFLAGS='-C target-cpu=native' cargo build --release")
    env = os.environ.copy()
    env['RUSTFLAGS'] = '-C target-cpu=native'
    subprocess.run(['cargo', 'build', '--release'], check=True, env=env)

    cmd = [
        './target/release/mst-bench',
        '--dataset', dataset_path,
        '--sizes', sizes,
        '--experiment', experiment,
    ] + extra_args
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description='Unified MST benchmark runner')
    p.add_argument('--lang', choices=['python', 'rust', 'both'], default='both')
    p.add_argument('--data', choices=['road', 'amazon', 'both'], default='both')
    p.add_argument('--experiment', choices=['scalability', 'speedup', 'both'], default='both')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--threads', default='1,2,4,6,8,12,16,24')
    args, extra = p.parse_known_args()

    extra_args = ['--runs', str(args.runs), '--threads', args.threads] + extra

    # Define datasets with separate sizes for scalability vs speedup
    datasets = []
    if args.data in ('road', 'both'):
        datasets.append(('road', DATASETS['road'], ROAD_SIZES, ROAD_SPEEDUP_SIZES))
    if args.data in ('amazon', 'both'):
        datasets.append(('amazon', DATASETS['amazon'], AMAZON_SIZES, AMAZON_SPEEDUP_SIZES))

    for name, path, scale_sizes, speed_sizes in datasets:
        if not os.path.exists(path):
            print(f"ERROR: Dataset not found: {path}")
            print("Run: python3 download_data.py")
            sys.exit(1)

        if args.experiment in ('scalability', 'both'):
            if args.lang in ('python', 'both'):
                run_python(path, scale_sizes, 'scalability',
                           extra_args + ['--output-dir', f'results/python/{name}',
                                         '--validate'])
            if args.lang in ('rust', 'both'):
                run_rust(path, scale_sizes, 'scalability',
                         extra_args + ['--output-dir', f'results/rust/{name}'])

        if args.experiment in ('speedup', 'both'):
            if args.lang in ('python', 'both'):
                run_python(path, speed_sizes, 'speedup',
                           extra_args + ['--output-dir', f'results/python/{name}',
                                         '--no-validate'])
            if args.lang in ('rust', 'both'):
                run_rust(path, speed_sizes, 'speedup',
                         extra_args + ['--output-dir', f'results/rust/{name}'])

    print("\n" + "=" * 65)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 65)
    print("\nGenerate annotated plots with:")
    print("  python3.12 generate_ahmed_plots.py")


if __name__ == '__main__':
    main()
