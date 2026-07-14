#!/usr/bin/env python3
"""
Extract gen-only performance metrics from trtllm iterlog files (no ctx.json needed).

Reports: tps/user, output_tps/gen_gpu, output_tput, avg_itertime

Scans <input_dir>/*/concurrency_*/gen_only*.txt. For the tekit disagg benchmark that means
pointing -i at the ISL-OSL directory that contains the per-config run dirs, e.g.:
    <log_dir>/<YYYYMMDD>/8192-1024/
whose children look like  disagg_ctx1_gen1_dep16_batch128_eplb384_mtp3/concurrency_256/gen_only_0.txt

tps/user is corrected by the MTP acceptance rate: tps_per_user = accept_rate / avg_itertime.
The acceptance rate defaults to the DeepSeek-V4-Pro MTP rates {1: 1.7, 3: 2.44}; override with
--accept-rate.

Usage:
    python get_gen_only_perf.py -i <log_dir>
    python get_gen_only_perf.py -i <log_dir> --accept-rate "1:1.7,3:2.44"
"""

import argparse
import glob
import os
import re
import pandas as pd

# Default DeepSeek-V4-Pro MTP acceptance rates (accepted tokens per gen iteration).
# tps/user = accept_rate / avg_itertime. Override with --accept-rate.
DEFAULT_ACCEPT_RATE = {1: 1.7, 3: 2.44}


def process_gen_iterlog(dir_prefix, accept_rate=None):
    """Process gen-only iterlog files and extract performance metrics."""
    if accept_rate is None:
        accept_rate = {}

    # Recursive so -i can point at the base log dir (which holds one subtree per config), a single
    # config dir, or an older nested layout.
    pattern = os.path.join(dir_prefix, "**", "concurrency_*", "gen_only*.txt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files found matching {pattern}")
        return []

    print(f"Found {len(files)} iterlog files")
    summary = []

    for file in sorted(files):
        # Concurrency comes from the concurrency_<N>/ subdir, which is authoritative for both
        # the tekit layout  (.../disagg_ctx1_gen1_dep16_batch128_eplb384_mtp3/concurrency_256/)
        # and the internal layout (.../ctx1_gen1_dep16_concurrency256_eplb384_mtp3/concurrency_256/).
        # tp/dp-mode/eplb/mtp are read individually so we don't depend on the exact token order.
        tp_match = re.search(r'(tep|dep)(\d+)', file)
        conc_match = re.search(r'concurrency_(\d+)', file)
        if not tp_match or not conc_match:
            continue

        dp_tep = tp_match.group(1)
        rank_num = int(tp_match.group(2))
        concurrency = int(conc_match.group(1))
        eplb_match = re.search(r'eplb(\d+)', file)
        eplb_num = int(eplb_match.group(1)) if eplb_match else 0
        mtp_match = re.search(r'_mtp(\d+)', file)
        mtp_num = int(mtp_match.group(1)) if mtp_match else 0

        # Extract ctx/gen instance counts from the config dir name
        config_dir = file.rsplit('/', 2)[0]
        config_name = os.path.basename(config_dir)
        ctx_match = re.search(r'ctx(\d+)_gen(\d+)', config_name)
        ctx_inst = int(ctx_match.group(1)) if ctx_match else 1
        gen_inst = int(ctx_match.group(2)) if ctx_match else 1

        # Parse iterlog
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        log_pattern = (
            r'iter = (\d+), global_rank = (\d+), rank = (\d+), '
            r'currank_total_requests = (\d+)/(\d+), '
            r'host_step_time = ([\d.]+)ms, prev_device_step_time = ([\d.]+)ms, '
            r'timestamp = ([^,]+), num_scheduled_requests: (\d+), '
            r'states = \{\'num_ctx_requests\': (\d+), \'num_ctx_tokens\': (\d+), '
            r'\'num_generation_tokens\': (\d+)\}'
        )
        matches = re.findall(log_pattern, content)
        if not matches:
            continue

        rows = []
        for m in matches:
            rows.append({
                'iter': int(m[0]),
                'global_rank': int(m[1]),
                'elapsed_time': float(m[6]) / 1000,  # prev_device_step_time
                'num_scheduled_requests': int(m[8]),
                'num_ctx_tokens': int(m[10]),
                'num_generation_tokens': int(m[11]),
            })

        df = pd.DataFrame(rows)

        # Filter: gen-only iters (no ctx tokens)
        df = df[df['num_ctx_tokens'] == 0]
        df = df.groupby(['iter', 'global_rank']).last().reset_index()

        # Trim warmup/cooldown
        df = df.iloc[50:-10]

        # Filter to steady-state iters
        if dp_tep == 'tep':
            df = df[df['num_scheduled_requests'] == concurrency]
            df = df[df['num_generation_tokens'] == concurrency * (mtp_num + 1)]
        else:  # dep
            per_rank = concurrency // rank_num
            df = df[df['num_scheduled_requests'] == per_rank]
            df = df[df['num_generation_tokens'] == per_rank * (mtp_num + 1)]

        if df.empty:
            continue

        # Filter outliers (median ±20%)
        median_t = df['elapsed_time'].median()
        df = df[(df['elapsed_time'] >= median_t * 0.8) &
                (df['elapsed_time'] <= median_t * 1.2)]
        if df.empty:
            continue

        # Calculate metrics
        avg_itertime = df['elapsed_time'].mean()
        ar = accept_rate.get(mtp_num, 1.0) if mtp_num > 0 else 1.0
        tps_per_user = ar / avg_itertime if avg_itertime > 0 else 0
        output_tput = tps_per_user * concurrency
        output_tps_per_gen_gpu = output_tput / rank_num

        name = f"ctx{ctx_inst}_gen{gen_inst}_{dp_tep}{rank_num}_c{concurrency}_mtp{mtp_num}"
        summary.append({
            'config': name,
            'concurrency': concurrency,
            'mtp': mtp_num,
            'tp': rank_num,
            'adp': dp_tep == 'dep',
            'eplb': eplb_num,
            'tps_per_user': round(tps_per_user, 2),
            'output_tps_per_gen_gpu': round(output_tps_per_gen_gpu, 2),
            'output_tput': round(output_tput, 2),
            'avg_itertime_ms': round(avg_itertime * 1000, 3),
            'num_iters': len(df),
        })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Extract gen-only performance metrics from iterlog files (no ctx.json needed)')
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Gen-only benchmark directory')
    parser.add_argument('--accept-rate', type=str, default=None,
                        help='MTP accept rate as "mtp:rate,..." (default: "1:1.7,3:2.44")')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output CSV file (default: <input_dir>/gen_only_perf.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} does not exist")
        return

    # Parse accept rate (default to the DeepSeek-V4-Pro MTP rates if not given)
    if args.accept_rate:
        accept_rate = {}
        for pair in args.accept_rate.split(','):
            k, v = pair.split(':')
            accept_rate[int(k)] = float(v)
    else:
        accept_rate = dict(DEFAULT_ACCEPT_RATE)
    print(f"Using MTP accept rates: {accept_rate}")

    summary = process_gen_iterlog(args.input_dir, accept_rate)
    if not summary:
        print("No data extracted")
        return

    df = pd.DataFrame(summary)
    df = df.sort_values('tps_per_user', ascending=False)

    # Save CSV
    output_file = args.output or os.path.join(args.input_dir, 'gen_only_perf.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Print table
    print(f"\n{'Config':<55} {'Conc':>6} {'TPS/User':>10} {'OutTPS/GPU':>12} {'OutTput':>12} {'IterTime':>10} {'Iters':>6}")
    print("-" * 115)
    for _, r in df.iterrows():
        print(f"{r['config']:<55} {r['concurrency']:>6} {r['tps_per_user']:>10.2f} "
              f"{r['output_tps_per_gen_gpu']:>12.2f} {r['output_tput']:>12.2f} "
              f"{r['avg_itertime_ms']:>9.3f}ms {r['num_iters']:>6}")

    print(f"\nTotal: {len(summary)} configs processed")


if __name__ == "__main__":
    main()
