#!/usr/bin/env python3
"""
GPU Power Monitoring Visualization Tool

Generates text-based reports from GPU power CSV data.
Supports both nvidia-smi and amd-smi CSV formats.

Usage:
    python3 plot_gpu_power.py <csv_file> [--output-dir <dir>] [--title <label>]

Supported CSV formats:
    nvidia-smi: timestamp, index, power.draw [W], power.limit [W], temperature.gpu, ...
    amd-smi:    gpu, power, power_cap, temperature_edge, temperature_hotspot, gfx_activity, ...
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


# Column name mappings for different tool outputs
COLUMN_ALIASES = {
    'gpu_id': ['index', 'gpu', 'device'],
    'power': ['power.draw [W]', 'power', 'power_socket_power', 'socket_power', 'average_socket_power'],
    'power_limit': ['power.limit [W]', 'power_cap', 'power_limit', 'slowdown_power_cap'],
    'temp_edge': ['temperature.gpu', 'temperature_edge', 'temp_edge', 'temperature_hotspot_current'],
    'temp_junction': ['temperature.junction', 'temperature_hotspot', 'temp_junction', 'temperature_mem_current'],
    'gpu_util': ['utilization.gpu [%]', 'gfx_activity', 'gpu_util', 'gfx_activity_acc'],
    'mem_used': ['memory.used [MiB]', 'fb_used', 'vram_used', 'mem_used'],
    'mem_total': ['memory.total [MiB]', 'fb_total', 'vram_total', 'mem_total'],
}


def find_column(headers, field):
    """Find a column by trying known aliases."""
    for alias in COLUMN_ALIASES.get(field, []):
        # Try exact match first
        if alias in headers:
            return alias
        # Try case-insensitive
        for h in headers:
            if h.strip().lower() == alias.lower():
                return h
    return None


def parse_float(val):
    """Parse a numeric value, stripping units."""
    if not val or val.strip() in ('N/A', '', 'nan', 'None'):
        return 0.0
    # Strip common units
    cleaned = val.strip().replace(' W', '').replace(' C', '').replace(' %', '').replace(' MiB', '').replace(' MB', '')
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_csv(filepath):
    """Parse GPU power monitoring CSV file (nvidia-smi or amd-smi format)."""
    data = defaultdict(lambda: {'power': [], 'power_limit': [],
                                'temp_edge': [], 'temp_junction': [],
                                'util': [], 'mem_used': [], 'mem_total': []})

    with open(filepath) as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []

        col_gpu = find_column(headers, 'gpu_id')
        col_power = find_column(headers, 'power')
        col_plimit = find_column(headers, 'power_limit')
        col_tedge = find_column(headers, 'temp_edge')
        col_tjunction = find_column(headers, 'temp_junction')
        col_util = find_column(headers, 'gpu_util')
        col_mem_used = find_column(headers, 'mem_used')
        col_mem_total = find_column(headers, 'mem_total')

        for row in reader:
            gpu_id = row.get(col_gpu, '0').strip() if col_gpu else '0'
            d = data[gpu_id]

            if col_power:
                d['power'].append(parse_float(row.get(col_power, '0')))
            if col_plimit:
                d['power_limit'].append(parse_float(row.get(col_plimit, '0')))
            if col_tedge:
                d['temp_edge'].append(parse_float(row.get(col_tedge, '0')))
            if col_tjunction:
                d['temp_junction'].append(parse_float(row.get(col_tjunction, '0')))
            if col_util:
                d['util'].append(parse_float(row.get(col_util, '0')))
            if col_mem_used:
                val = parse_float(row.get(col_mem_used, '0'))
                # amd-smi reports memory in MB, nvidia-smi in MiB
                d['mem_used'].append(val)
            if col_mem_total:
                val = parse_float(row.get(col_mem_total, '0'))
                d['mem_total'].append(val)

    return data


def print_summary(data, title=""):
    """Print summary statistics."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    total_avg_power = 0

    print(f"\n{'GPU':>5} | {'Avg Power':>10} | {'Peak Power':>11} | {'Avg Temp':>9} | {'Peak Temp':>10} | {'Avg Util':>9} | {'Peak Mem':>10}")
    print(f"{'─' * 5}-+-{'─' * 10}-+-{'─' * 11}-+-{'─' * 9}-+-{'─' * 10}-+-{'─' * 9}-+-{'─' * 10}")

    for gpu_id in sorted(data.keys()):
        d = data[gpu_id]
        if not d['power']:
            continue

        avg_p = sum(d['power']) / len(d['power'])
        max_p = max(d['power'])
        temps = d['temp_junction'] if any(t > 0 for t in d['temp_junction']) else d['temp_edge']
        avg_t = sum(temps) / len(temps) if temps else 0
        max_t = max(temps) if temps else 0
        avg_u = sum(d['util']) / len(d['util']) if d['util'] else 0
        max_m = max(d['mem_used']) if d['mem_used'] else 0

        total_avg_power += avg_p

        print(f"GPU {gpu_id:>1} | {avg_p:>8.1f} W | {max_p:>9.1f} W | {avg_t:>6.1f}°C | {max_t:>7.0f}°C | {avg_u:>7.1f}% | {max_m:>7.0f} MiB")

    num_gpus = len(data)
    samples = len(list(data.values())[0]['power']) if data else 0
    print(f"\nTotal: {num_gpus} GPUs, {samples} samples/GPU")
    print(f"Total Avg Power: {total_avg_power:.0f} W")

    limits = []
    for d in data.values():
        if d['power_limit']:
            limits.extend([l for l in d['power_limit'] if l > 0])
    if limits:
        avg_limit = sum(limits) / len(limits)
        peak_total = sum(max(d['power']) for d in data.values() if d['power'])
        peak_tdp_pct = (peak_total / (avg_limit * num_gpus)) * 100
        print(f"Power Limit: {avg_limit:.0f} W/GPU, Peak TDP utilization: {peak_tdp_pct:.0f}%")


def generate_ascii_chart(values, width=60, height=12, y_label=""):
    """Generate a simple ASCII chart."""
    if not values:
        return "No data"

    min_v = min(values)
    max_v = max(values)
    v_range = max_v - min_v if max_v != min_v else 1

    if len(values) > width:
        step = len(values) / width
        sampled = []
        for i in range(width):
            idx = int(i * step)
            chunk_end = int((i + 1) * step)
            chunk = values[idx:chunk_end]
            sampled.append(sum(chunk) / len(chunk) if chunk else 0)
    else:
        sampled = values
        width = len(sampled)

    lines = []
    for row in range(height, -1, -1):
        threshold = min_v + (row / height) * v_range
        line = ""
        for col in range(width):
            line += "█" if sampled[col] >= threshold else " "

        if row == height:
            label = f"{max_v:>8.1f}"
        elif row == 0:
            label = f"{min_v:>8.1f}"
        elif row == height // 2:
            label = f"{(min_v + max_v) / 2:>8.1f}"
        else:
            label = "        "
        lines.append(f"{label} │{line}")

    lines.append(f"{'':>8} └{'─' * width}")
    if y_label:
        lines.append(f"{'':>8}  {y_label}")

    return "\n".join(lines)


def generate_report(data, output_dir, title=""):
    """Generate text-based report with ASCII charts."""
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("No data to plot")
        return

    report_path = os.path.join(output_dir, "power_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"GPU Power Monitoring Report\n")
        if title:
            f.write(f"Run: {title}\n")
        f.write(f"{'=' * 70}\n\n")

        for gpu_id in sorted(data.keys()):
            d = data[gpu_id]
            if not d['power']:
                continue
            f.write(f"\n--- GPU {gpu_id} Power Over Time ---\n")
            f.write(generate_ascii_chart(d['power'], y_label="Watts") + "\n")

        # Total power
        num_samples = min(len(d['power']) for d in data.values())
        total_power = [sum(data[gpu]['power'][i] for gpu in data if i < len(data[gpu]['power']))
                       for i in range(num_samples)]

        f.write(f"\n--- Total Power ({len(data)} GPUs) ---\n")
        f.write(generate_ascii_chart(total_power, height=15, y_label="Watts (total)") + "\n")

        avg_total = sum(total_power) / len(total_power) if total_power else 0
        peak_total = max(total_power) if total_power else 0
        f.write(f"\nAvg: {avg_total:.0f} W | Peak: {peak_total:.0f} W\n")

    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU Power Monitoring Visualization")
    parser.add_argument("csv_file", help="Path to GPU power CSV file")
    parser.add_argument("--output-dir", default="./plots", help="Output directory")
    parser.add_argument("--title", default="", help="Title/label for the run")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: {args.csv_file} not found")
        sys.exit(1)

    data = parse_csv(args.csv_file)
    if not data:
        print("Error: No data found in CSV")
        sys.exit(1)

    print_summary(data, title=args.title)
    generate_report(data, args.output_dir, title=args.title)


if __name__ == "__main__":
    main()
