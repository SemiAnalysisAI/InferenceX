#!/usr/bin/env python3
"""
GPU Power Monitoring Visualization Tool

Generates charts from nvidia-smi or rocm-smi CSV power data:
1. Per-GPU power over time
2. Total power over time (aggregate)
3. Temperature over time
4. GPU utilization over time
5. Power vs utilization scatter (correlation)

Usage:
    python3 plot_gpu_power.py <csv_file> [--output-dir <dir>] [--title <label>]

CSV format (nvidia-smi style):
    timestamp, index, power.draw [W], power.limit [W], temperature.gpu, utilization.gpu [%], ...

CSV format (rocm-smi style):
    timestamp, index, power.draw [W], power.limit [W], temperature.edge [C], temperature.junction [C], utilization.gpu [%], memory.used [MiB], memory.total [MiB]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime


def parse_csv(filepath):
    """Parse GPU power monitoring CSV file."""
    data = defaultdict(lambda: {
        'timestamps': [], 'power': [], 'power_limit': [],
        'temp': [], 'temp_junction': [],
        'util': [], 'mem_used': [], 'mem_total': []
    })

    with open(filepath) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            gpu_id = row.get('index', '0')
            ts_str = row.get('timestamp', '')

            try:
                if '.' in ts_str:
                    ts = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                elif 'T' in ts_str:
                    ts = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
                else:
                    ts = datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S')
            except (ValueError, TypeError):
                continue

            d = data[gpu_id]
            d['timestamps'].append(ts)
            d['power'].append(float(row.get('power.draw [W]', 0)))
            d['power_limit'].append(float(row.get('power.limit [W]', 0)))

            # Handle both nvidia-smi and rocm-smi formats
            if 'temperature.gpu' in headers:
                d['temp'].append(float(row.get('temperature.gpu', 0)))
                d['temp_junction'].append(float(row.get('temperature.gpu', 0)))
            elif 'temperature.edge [C]' in headers:
                d['temp'].append(float(row.get('temperature.edge [C]', 0)))
                d['temp_junction'].append(float(row.get('temperature.junction [C]', 0)))
            else:
                d['temp'].append(0)
                d['temp_junction'].append(0)

            d['util'].append(float(row.get('utilization.gpu [%]', 0)))

            if 'memory.used [MiB]' in headers:
                d['mem_used'].append(float(row.get('memory.used [MiB]', 0)))
                d['mem_total'].append(float(row.get('memory.total [MiB]', 0)))
            elif 'utilization.memory [%]' in headers:
                d['mem_used'].append(float(row.get('utilization.memory [%]', 0)))
                d['mem_total'].append(100.0)

    return data


def compute_elapsed_seconds(timestamps, base_time):
    """Convert timestamps to elapsed seconds from base_time."""
    return [(t - base_time).total_seconds() for t in timestamps]


def generate_ascii_chart(values, width=60, height=15, y_label="", x_label="time (s)"):
    """Generate a simple ASCII chart."""
    if not values:
        return "No data"

    min_v = min(values)
    max_v = max(values)
    v_range = max_v - min_v if max_v != min_v else 1

    # Downsample if needed
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
            if sampled[col] >= threshold:
                line += "█"
            else:
                line += " "

        if row == height:
            label = f"{max_v:>8.1f}"
        elif row == 0:
            label = f"{min_v:>8.1f}"
        elif row == height // 2:
            mid = (min_v + max_v) / 2
            label = f"{mid:>8.1f}"
        else:
            label = "        "

        lines.append(f"{label} │{line}")

    lines.append(f"{'':>8} └{'─' * width}")
    lines.append(f"{'':>8}  {y_label}")

    return "\n".join(lines)


def print_summary(data, title=""):
    """Print summary statistics."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    total_avg_power = 0
    total_peak_power = 0

    print(f"\n{'GPU':>5} | {'Avg Power':>10} | {'Peak Power':>11} | {'Avg Temp':>9} | {'Peak Temp':>10} | {'Avg Util':>9} | {'Peak Mem':>10}")
    print(f"{'─' * 5}-+-{'─' * 10}-+-{'─' * 11}-+-{'─' * 9}-+-{'─' * 10}-+-{'─' * 9}-+-{'─' * 10}")

    for gpu_id in sorted(data.keys()):
        d = data[gpu_id]
        if not d['power']:
            continue

        avg_p = sum(d['power']) / len(d['power'])
        max_p = max(d['power'])
        # Use junction temp if available and non-zero
        temps = d['temp_junction'] if any(t > 0 for t in d['temp_junction']) else d['temp']
        avg_t = sum(temps) / len(temps) if temps else 0
        max_t = max(temps) if temps else 0
        avg_u = sum(d['util']) / len(d['util'])
        max_m = max(d['mem_used']) if d['mem_used'] else 0

        total_avg_power += avg_p
        total_peak_power += max_p

        print(f"GPU {gpu_id:>1} | {avg_p:>8.1f} W | {max_p:>9.1f} W | {avg_t:>6.1f}°C | {max_t:>7.0f}°C | {avg_u:>7.1f}% | {max_m:>7.0f} MiB")

    num_gpus = len(data)
    samples = len(list(data.values())[0]['power']) if data else 0
    print(f"\nTotal: {num_gpus} GPUs, {samples} samples/GPU")
    print(f"Total Avg Power: {total_avg_power:.0f} W")
    print(f"Total Peak Power: ~{total_peak_power:.0f} W")

    # Power limit
    limits = []
    for d in data.values():
        if d['power_limit']:
            limits.extend(d['power_limit'])
    if limits:
        avg_limit = sum(limits) / len(limits)
        if avg_limit > 0:
            peak_tdp_pct = (total_peak_power / (avg_limit * num_gpus)) * 100
            print(f"Power Limit: {avg_limit:.0f} W/GPU, Peak TDP utilization: {peak_tdp_pct:.0f}%")


def generate_plots(data, output_dir, title=""):
    """Generate text-based summary plots."""
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("No data to plot")
        return

    # Find global base time
    all_times = []
    for d in data.values():
        all_times.extend(d['timestamps'])
    if not all_times:
        return
    base_time = min(all_times)

    # 1. Per-GPU power over time
    report_path = os.path.join(output_dir, "power_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"GPU Power Monitoring Report\n")
        if title:
            f.write(f"Run: {title}\n")
        f.write(f"{'=' * 60}\n\n")

        # Per-GPU power charts
        for gpu_id in sorted(data.keys()):
            d = data[gpu_id]
            if not d['power']:
                continue
            f.write(f"\n--- GPU {gpu_id} Power Over Time ---\n")
            chart = generate_ascii_chart(d['power'], width=60, height=12, y_label="Watts")
            f.write(chart + "\n")

        # Total power
        num_samples = min(len(d['power']) for d in data.values())
        total_power = []
        for i in range(num_samples):
            total = sum(data[gpu]['power'][i] for gpu in data if i < len(data[gpu]['power']))
            total_power.append(total)

        f.write(f"\n--- Total Power Over Time ({len(data)} GPUs) ---\n")
        chart = generate_ascii_chart(total_power, width=60, height=15, y_label="Watts (total)")
        f.write(chart + "\n")

        avg_total = sum(total_power) / len(total_power) if total_power else 0
        peak_total = max(total_power) if total_power else 0
        f.write(f"\nAvg Total: {avg_total:.0f} W | Peak Total: {peak_total:.0f} W\n")

        # Temperature
        for gpu_id in sorted(data.keys()):
            d = data[gpu_id]
            temps = d['temp_junction'] if any(t > 0 for t in d['temp_junction']) else d['temp']
            if not temps:
                continue
            f.write(f"\n--- GPU {gpu_id} Temperature Over Time ---\n")
            chart = generate_ascii_chart(temps, width=60, height=10, y_label="°C")
            f.write(chart + "\n")

        # Utilization
        for gpu_id in sorted(data.keys()):
            d = data[gpu_id]
            if not d['util']:
                continue
            f.write(f"\n--- GPU {gpu_id} Utilization Over Time ---\n")
            chart = generate_ascii_chart(d['util'], width=60, height=10, y_label="%")
            f.write(chart + "\n")

        # Summary stats
        f.write(f"\n{'=' * 60}\n")
        f.write("Summary Statistics\n")
        f.write(f"{'=' * 60}\n")
        for gpu_id in sorted(data.keys()):
            d = data[gpu_id]
            if not d['power']:
                continue
            avg_p = sum(d['power']) / len(d['power'])
            max_p = max(d['power'])
            temps = d['temp_junction'] if any(t > 0 for t in d['temp_junction']) else d['temp']
            avg_t = sum(temps) / len(temps) if temps else 0
            max_t = max(temps) if temps else 0
            avg_u = sum(d['util']) / len(d['util'])
            f.write(f"GPU {gpu_id}: Avg Power={avg_p:.1f}W  Peak={max_p:.1f}W  "
                    f"Avg Temp={avg_t:.1f}°C  Peak Temp={max_t:.0f}°C  Avg Util={avg_u:.1f}%\n")

    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU Power Monitoring Visualization")
    parser.add_argument("csv_file", help="Path to GPU power CSV file")
    parser.add_argument("--output-dir", default="./plots", help="Output directory for plots")
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
    generate_plots(data, args.output_dir, title=args.title)


if __name__ == "__main__":
    main()
