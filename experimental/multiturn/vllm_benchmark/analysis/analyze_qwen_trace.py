#!/usr/bin/env python3
"""
Analyze Qwen trace dataset conversation structure.

Reconstructs conversations from JSONL trace data and generates statistics
comparable to analyze_wildchat.py, so we can model our sampled dataset accurately.

Usage:
    python analyze_qwen_trace.py qwen_dataset/qwen_traceA_blksz_16.jsonl
    python analyze_qwen_trace.py qwen_dataset/qwen_traceA_blksz_16.jsonl --output qwen_analysis.png
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_trace(trace_path: Path) -> list[dict]:
    """Load JSONL trace file."""
    records = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} requests from {trace_path}")
    return records


def reconstruct_conversations(records: list[dict]) -> list[list[dict]]:
    """Reconstruct multi-turn conversations from flat request records.

    Conversations are linked by parent_chat_id:
    - parent_chat_id == -1: first turn of a new conversation
    - parent_chat_id == X: continuation of chat X
    """
    by_id = {r["chat_id"]: r for r in records}

    # Build children map
    children = defaultdict(list)
    roots = []
    for r in records:
        if r["parent_chat_id"] == -1:
            roots.append(r["chat_id"])
        else:
            children[r["parent_chat_id"]].append(r["chat_id"])

    # Walk each conversation chain from root
    conversations = []
    for root_id in roots:
        conv = []
        current = root_id
        while current is not None:
            conv.append(by_id[current])
            kids = children.get(current, [])
            # Follow the first child (linear chain)
            current = kids[0] if kids else None
        conversations.append(conv)

    print(f"Reconstructed {len(conversations):,} conversations")
    return conversations


def print_summary(conversations: list[list[dict]]):
    """Print summary statistics comparable to analyze_wildchat.py."""
    turns_per_conv = np.array([len(c) for c in conversations])
    total_input_per_conv = np.array([sum(t["input_length"] for t in c) for c in conversations])
    total_output_per_conv = np.array([sum(t["output_length"] for t in c) for c in conversations])
    n = len(conversations)

    # Per-turn stats
    all_input_lengths = np.array([t["input_length"] for c in conversations for t in c])
    all_output_lengths = np.array([t["output_length"] for c in conversations for t in c])

    # Type distribution
    type_counts = Counter(t["type"] for c in conversations for t in c)

    print("\n" + "=" * 60)
    print("QWEN TRACE CONVERSATION STATISTICS")
    print("=" * 60)
    print(f"\nTotal conversations: {n:,}")
    print(f"Total requests:      {sum(len(c) for c in conversations):,}")

    print(f"\nRequest types:")
    for rtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype}: {count:,} ({100*count/len(all_input_lengths):.1f}%)")

    print(f"\nTurns per conversation:")
    print(f"  Mean:   {turns_per_conv.mean():.1f}")
    print(f"  Median: {np.median(turns_per_conv):.0f}")
    print(f"  Std:    {turns_per_conv.std():.1f}")
    print(f"  Min:    {turns_per_conv.min()}")
    print(f"  Max:    {turns_per_conv.max()}")
    print(f"\nTurns percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(turns_per_conv, p):.0f} turns")
    print(f"\nConversations exceeding turn threshold:")
    for t in [2, 3, 5, 10, 15, 20]:
        count = np.sum(turns_per_conv > t)
        print(f"  > {t:2d} turns: {count:>10,} ({100*count/n:>5.1f}%)")

    print(f"\nInput tokens per request (all turns):")
    print(f"  Mean:   {all_input_lengths.mean():,.0f}")
    print(f"  Median: {np.median(all_input_lengths):,.0f}")
    print(f"  Std:    {all_input_lengths.std():,.0f}")
    print(f"  Min:    {all_input_lengths.min():,}")
    print(f"  Max:    {all_input_lengths.max():,}")
    print(f"\nInput tokens percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(all_input_lengths, p):,.0f} tokens")

    print(f"\nOutput tokens per request (all turns):")
    print(f"  Mean:   {all_output_lengths.mean():,.0f}")
    print(f"  Median: {np.median(all_output_lengths):,.0f}")
    print(f"  Std:    {all_output_lengths.std():,.0f}")
    print(f"  Min:    {all_output_lengths.min():,}")
    print(f"  Max:    {all_output_lengths.max():,}")
    print(f"\nOutput tokens percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(all_output_lengths, p):,.0f} tokens")

    print(f"\nTotal input tokens per conversation:")
    print(f"  Mean:   {total_input_per_conv.mean():,.0f}")
    print(f"  Median: {np.median(total_input_per_conv):,.0f}")
    print(f"  Std:    {total_input_per_conv.std():,.0f}")
    print(f"  Min:    {total_input_per_conv.min():,}")
    print(f"  Max:    {total_input_per_conv.max():,}")

    print(f"\nTotal output tokens per conversation:")
    print(f"  Mean:   {total_output_per_conv.mean():,.0f}")
    print(f"  Median: {np.median(total_output_per_conv):,.0f}")
    print(f"  Std:    {total_output_per_conv.std():,.0f}")
    print(f"  Min:    {total_output_per_conv.min():,}")
    print(f"  Max:    {total_output_per_conv.max():,}")

    # Per-turn breakdown
    max_turn = int(turns_per_conv.max())
    display_turns = min(max_turn, 15)
    print(f"\nMedian input/output tokens by turn number:")
    print(f"  {'Turn':>4s}  {'N':>8s}  {'Input Med':>10s}  {'Input Mean':>10s}  {'Output Med':>10s}  {'Output Mean':>10s}")
    for t in range(1, display_turns + 1):
        inp = [c[t-1]["input_length"] for c in conversations if len(c) >= t]
        out = [c[t-1]["output_length"] for c in conversations if len(c) >= t]
        if inp:
            inp_arr = np.array(inp)
            out_arr = np.array(out)
            print(f"  {t:4d}  {len(inp):8,}  {np.median(inp_arr):10,.0f}  {inp_arr.mean():10,.0f}  {np.median(out_arr):10,.0f}  {out_arr.mean():10,.0f}")

    print("=" * 60)


def compute_sequence_metrics(conversations: list[list[dict]]) -> list[dict]:
    """Compute ISL/OSL/ISL-increment per turn, matching analyze_sequences.py format.

    In the Qwen trace, input_length is the full context sent to the model (ISL),
    and output_length is the generated tokens (OSL).
    """
    all_turns = []
    for conv_idx, conv in enumerate(conversations):
        for turn_idx, turn in enumerate(conv):
            isl = turn["input_length"]
            osl = turn["output_length"]
            isl_increment = isl if turn_idx == 0 else (isl - conv[turn_idx - 1]["input_length"])

            record = {
                "conv_idx": conv_idx,
                "turn_idx": turn_idx,
                "isl": isl,
                "osl": osl,
                "isl_increment": isl_increment,
                "type": turn.get("type", "text"),
            }

            if turn_idx > 0:
                record["prev_osl"] = conv[turn_idx - 1]["output_length"]
                record["isl_increment_minus_prev_osl"] = isl_increment - record["prev_osl"]

            all_turns.append(record)
    return all_turns


def generate_sequence_analysis_plot(turn_data: list[dict], output_path: Path, label: str = ""):
    """Generate 6-panel sequence analysis plot matching analyze_sequences.py."""
    isl = np.array([t["isl"] for t in turn_data])
    osl = np.array([t["osl"] for t in turn_data])
    isl_incr = np.array([t["isl_increment"] for t in turn_data])
    turn_idxs = np.array([t["turn_idx"] for t in turn_data])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title_suffix = f" - {label}" if label else ""
    fig.suptitle(f"Sequence Length Analysis (Qwen Trace{title_suffix})", fontsize=14)

    # (0,0) ISL distribution
    ax = axes[0, 0]
    ax.hist(isl, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(isl.mean(), color="red", linestyle="--", label=f"Mean: {isl.mean():,.0f}")
    ax.axvline(np.median(isl), color="green", linestyle="--", label=f"Median: {np.median(isl):,.0f}")
    ax.set_xlabel("ISL (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Input Sequence Length Distribution")
    ax.legend(fontsize=8)

    # (0,1) OSL distribution
    ax = axes[0, 1]
    ax.hist(osl, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(osl.mean(), color="red", linestyle="--", label=f"Mean: {osl.mean():,.0f}")
    ax.axvline(np.median(osl), color="green", linestyle="--", label=f"Median: {np.median(osl):,.0f}")
    ax.set_xlabel("OSL (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Output Sequence Length Distribution")
    ax.legend(fontsize=8)

    # (0,2) ISL increment distribution
    ax = axes[0, 2]
    ax.hist(isl_incr, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(isl_incr.mean(), color="red", linestyle="--", label=f"Mean: {isl_incr.mean():,.0f}")
    ax.set_xlabel("ISL Increment (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("ISL Increment per Turn")
    ax.legend(fontsize=8)

    # Per-turn aggregations
    max_turn = min(int(turn_idxs.max()), 14)
    turns_range = np.arange(0, max_turn + 1)

    def per_turn_stats(values):
        means, stds = [], []
        for t in turns_range:
            mask = turn_idxs == t
            v = values[mask]
            means.append(v.mean() if len(v) > 0 else 0)
            stds.append(v.std() if len(v) > 0 else 0)
        return np.array(means), np.array(stds)

    # (1,0) ISL vs Turn Number
    ax = axes[1, 0]
    means, stds = per_turn_stats(isl)
    ax.errorbar(turns_range, means, yerr=stds, marker="o", capsize=3, alpha=0.7)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL (tokens)")
    ax.set_title("ISL Growth by Turn")
    ax.grid(True, alpha=0.3)

    # (1,1) OSL vs Turn Number
    ax = axes[1, 1]
    means, stds = per_turn_stats(osl)
    ax.errorbar(turns_range, means, yerr=stds, marker="o", capsize=3, alpha=0.7, color="orange")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean OSL (tokens)")
    ax.set_title("OSL by Turn")
    ax.grid(True, alpha=0.3)

    # (1,2) ISL increment vs Turn Number
    ax = axes[1, 2]
    means, stds = per_turn_stats(isl_incr)
    ax.errorbar(turns_range, means, yerr=stds, marker="o", capsize=3, alpha=0.7, color="green")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL Increment (tokens)")
    ax.set_title("ISL Increment by Turn")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = output_path.with_name(output_path.stem.replace("_analysis", "") + "_sequence_analysis.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sequence analysis plot to {out_file}")


def generate_isl_per_turn_plot(turn_data: list[dict], conversations: list[list[dict]], output_path: Path, label: str = ""):
    """Generate 2-panel ISL per turn chart matching analyze_sequences.py."""
    turn_idxs = np.array([t["turn_idx"] for t in turn_data])
    isl_vals = np.array([t["isl"] for t in turn_data])
    osl_vals = np.array([t["osl"] for t in turn_data])

    max_turn = min(int(turn_idxs.max()), 14)
    turns_range = np.arange(0, max_turn + 1)

    # Filter to turns with enough samples
    turn_counts = {t: np.sum(turn_idxs == t) for t in turns_range}
    turns_range = np.array([t for t in turns_range if turn_counts[t] >= 100])

    isl_means = np.array([isl_vals[turn_idxs == t].mean() for t in turns_range])
    isl_stds = np.array([isl_vals[turn_idxs == t].std() for t in turns_range])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = f" - {label}" if label else ""
    fig.suptitle(f"Input Sequence Length (ISL) Per Turn (Qwen Trace{title_suffix})", fontsize=14)

    # Left: ISL growth line with error bands
    ax = axes[0]
    ax.errorbar(turns_range, isl_means, yerr=isl_stds, marker='o', capsize=4,
                linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(turns_range, isl_means - isl_stds, isl_means + isl_stds,
                    alpha=0.2, color='steelblue')
    for t, m in zip(turns_range, isl_means):
        ax.annotate(f'{m:,.0f}', (t, m), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL (tokens)")
    ax.set_title("Cumulative Input Length Per Turn\n(what the model sees)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns_range)

    # Right: Stacked bar showing ISL composition (input vs cumulative responses)
    ax = axes[1]
    # For each turn, compute mean input_length (new input) and mean prev output contribution
    mean_input_per_turn = []
    mean_output_per_turn = []
    for t in turns_range:
        inp = [c[t]["input_length"] for c in conversations if len(c) > t]
        out = [c[t]["output_length"] for c in conversations if len(c) > t]
        mean_input_per_turn.append(np.mean(inp) if inp else 0)
        mean_output_per_turn.append(np.mean(out) if out else 0)

    # Cumulative composition: ISL at turn t ≈ sum of all user inputs + sum of all prev responses
    # We approximate by computing cumulative means
    cum_user = np.zeros(len(turns_range))
    cum_response = np.zeros(len(turns_range))
    # Use the ISL increment breakdown: new user tokens ≈ isl_increment - prev_osl
    for i, t in enumerate(turns_range):
        mask = turn_idxs == t
        increments = np.array([td["isl_increment"] for td in turn_data if td["turn_idx"] == t])
        prev_osls = np.array([td.get("prev_osl", 0) for td in turn_data if td["turn_idx"] == t])
        new_user = increments - prev_osls if t > 0 else increments
        cum_user[i] = cum_user[i-1] + new_user.mean() if i > 0 else new_user.mean()
        cum_response[i] = cum_response[i-1] + (prev_osls.mean() if t > 0 else 0) if i > 0 else 0

    ax.bar(turns_range, cum_user, label='Cumulative User Input', color='forestgreen', alpha=0.8)
    ax.bar(turns_range, cum_response, bottom=cum_user, label='Cumulative Model Responses', color='coral', alpha=0.8)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Tokens")
    ax.set_title("ISL Composition\n(user input vs model responses)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(turns_range)

    plt.tight_layout()
    out_file = output_path.with_name(output_path.stem.replace("_analysis", "") + "_isl_per_turn.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ISL per turn chart to {out_file}")


def print_sequence_summary(turn_data: list[dict]):
    """Print sequence analysis summary matching analyze_sequences.py output."""
    isl = np.array([t["isl"] for t in turn_data])
    osl = np.array([t["osl"] for t in turn_data])
    isl_incr = np.array([t["isl_increment"] for t in turn_data])
    turn_idxs = np.array([t["turn_idx"] for t in turn_data])

    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH ANALYSIS (matching analyze_sequences.py)")
    print("=" * 70)

    print(f"\nTotal turns analyzed: {len(turn_data):,}")

    print(f"\nInput Sequence Length (ISL) - tokens seen by model:")
    print(f"  Mean:   {isl.mean():,.0f}")
    print(f"  Median: {np.median(isl):,.0f}")
    print(f"  Std:    {isl.std():,.0f}")
    print(f"  Min:    {isl.min():,}")
    print(f"  Max:    {isl.max():,}")

    print(f"\nOutput Sequence Length (OSL) - tokens generated:")
    print(f"  Mean:   {osl.mean():,.0f}")
    print(f"  Median: {np.median(osl):,.0f}")
    print(f"  Std:    {osl.std():,.0f}")
    print(f"  Min:    {osl.min():,}")
    print(f"  Max:    {osl.max():,}")

    print(f"\nISL Increment per turn:")
    print(f"  Mean:   {isl_incr.mean():,.0f}")
    print(f"  Median: {np.median(isl_incr):,.0f}")
    print(f"  Std:    {isl_incr.std():,.0f}")

    print(f"\n--- Statistics by Turn Number ---")
    print(f"  {'Turn':>4s}  {'N':>8s}  {'ISL Mean':>10s}  {'ISL Std':>10s}  {'OSL Mean':>10s}  {'OSL Std':>10s}  {'Incr Mean':>10s}")
    max_turn = min(int(max(t["turn_idx"] for t in turn_data)), 14)
    for t in range(max_turn + 1):
        mask = turn_idxs == t
        if mask.sum() < 10:
            continue
        print(f"  {t:4d}  {mask.sum():8,}  {isl[mask].mean():10,.0f}  {isl[mask].std():10,.0f}  {osl[mask].mean():10,.0f}  {osl[mask].std():10,.0f}  {isl_incr[mask].mean():10,.0f}")

    # ISL increment vs prev OSL breakdown
    later_turns = [t for t in turn_data if t["turn_idx"] > 0]
    if later_turns:
        prev_osls = np.array([t["prev_osl"] for t in later_turns])
        incr = np.array([t["isl_increment"] for t in later_turns])
        new_user = np.array([t["isl_increment_minus_prev_osl"] for t in later_turns])
        print(f"\n--- ISL Increment Breakdown (turns > 0) ---")
        print(f"Formula: ISL_increment = prev_OSL + new_user_tokens")
        print(f"\nISL Increment (total input growth per turn):")
        print(f"  Mean:   {incr.mean():,.0f}")
        print(f"  Median: {np.median(incr):,.0f}")
        print(f"\nPrevious OSL (previous response length):")
        print(f"  Mean:   {prev_osls.mean():,.0f}")
        print(f"  Median: {np.median(prev_osls):,.0f}")
        print(f"\nNew User Tokens (ISL_increment - prev_OSL):")
        print(f"  Mean:   {new_user.mean():,.0f}")
        print(f"  Median: {np.median(new_user):,.0f}")
        ratio = prev_osls / np.where(incr > 0, incr, 1)
        print(f"\nRatio: prev_OSL / ISL_increment:")
        print(f"  Mean:   {ratio.mean():.1%}")
        print(f"  Median: {np.median(ratio):.1%}")

    print("=" * 70)


def generate_plots(conversations: list[list[dict]], output_path: Path, label: str = ""):
    """Generate analysis plots comparable to analyze_wildchat.py."""
    turns_per_conv = np.array([len(c) for c in conversations])
    total_input_per_conv = np.array([sum(t["input_length"] for t in c) for c in conversations])
    all_input_lengths = np.array([t["input_length"] for c in conversations for t in c])
    all_output_lengths = np.array([t["output_length"] for c in conversations for t in c])
    n = len(conversations)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    title_suffix = f" - {label}" if label else ""
    fig.suptitle(f"Qwen Trace: Conversation Statistics (n={n:,}{title_suffix})", fontsize=14)

    # (0,0) Turns histogram
    ax = axes[0, 0]
    max_turns = min(int(turns_per_conv.max()), 20)
    bins = np.arange(0.5, max_turns + 1.5, 1)
    counts, _, bars = ax.hist(turns_per_conv[turns_per_conv <= max_turns], bins=bins, edgecolor='black', alpha=0.7)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{int(count):,}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Number of Conversations")
    ax.set_title("Turns per Conversation")
    ax.set_xticks(range(1, max_turns + 1))
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1) Turns CDF
    ax = axes[0, 1]
    sorted_turns = np.sort(turns_per_conv)
    cdf = np.arange(1, len(sorted_turns) + 1) / len(sorted_turns)
    ax.plot(sorted_turns, cdf, linewidth=2, color='steelblue')
    ax.fill_between(sorted_turns, cdf, alpha=0.3, color='steelblue')
    for p in [50, 75, 90, 95]:
        val = np.percentile(turns_per_conv, p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'P{p}: {val:.0f}', xy=(val, p/100),
                   xytext=(val + 0.5, p/100 - 0.05), fontsize=9)
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Turns CDF")
    ax.set_xlim(0, max(20, np.percentile(turns_per_conv, 99)))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (1,0) Input tokens histogram
    ax = axes[1, 0]
    ax.hist(all_input_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(all_input_lengths), color='red', linestyle='--', label=f'Median: {np.median(all_input_lengths):,.0f}')
    ax.axvline(all_input_lengths.mean(), color='orange', linestyle='--', label=f'Mean: {all_input_lengths.mean():,.0f}')
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Count")
    ax.set_title("Input Tokens per Request")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,1) Output tokens histogram
    ax = axes[1, 1]
    ax.hist(all_output_lengths, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.median(all_output_lengths), color='red', linestyle='--', label=f'Median: {np.median(all_output_lengths):,.0f}')
    ax.axvline(all_output_lengths.mean(), color='orange', linestyle='--', label=f'Mean: {all_output_lengths.mean():,.0f}')
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Count")
    ax.set_title("Output Tokens per Request")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (2,0) Median input/output by turn number
    ax = axes[2, 0]
    max_turn_plot = min(int(turns_per_conv.max()), 15)
    turn_nums = range(1, max_turn_plot + 1)
    med_input = []
    med_output = []
    for t in turn_nums:
        inp = [c[t-1]["input_length"] for c in conversations if len(c) >= t]
        out = [c[t-1]["output_length"] for c in conversations if len(c) >= t]
        med_input.append(np.median(inp) if inp else 0)
        med_output.append(np.median(out) if out else 0)
    ax.bar([t - 0.2 for t in turn_nums], med_input, width=0.4, label='Input', color='steelblue', alpha=0.8)
    ax.bar([t + 0.2 for t in turn_nums], med_output, width=0.4, label='Output', color='coral', alpha=0.8)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Median Tokens")
    ax.set_title("Median Input/Output Tokens by Turn")
    ax.set_xticks(list(turn_nums))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (2,1) Total input tokens per conversation exceedance
    ax = axes[2, 1]
    token_thresholds = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    token_counts_above = [np.sum(total_input_per_conv > t) for t in token_thresholds]
    token_pct_above = [100 * c / n for c in token_counts_above]
    bars = ax.bar(range(len(token_thresholds)), token_pct_above, edgecolor='black', alpha=0.7, color='mediumseagreen')
    for bar, count, pct in zip(bars, token_counts_above, token_pct_above):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=7)
    ax.set_xlabel("Threshold (total input tokens)")
    ax.set_ylabel("% of Conversations")
    ax.set_title("Conversations with > X Total Input Tokens")
    ax.set_xticks(range(len(token_thresholds)))
    ax.set_xticklabels([f'{t:,}' for t in token_thresholds], rotation=45, ha='right')
    if token_pct_above:
        ax.set_ylim(0, max(token_pct_above) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def compute_inter_turn_delays(conversations: list[list[dict]]) -> np.ndarray:
    """Compute inter-turn delays (seconds) from timestamps in multi-turn conversations."""
    delays = []
    for conv in conversations:
        for i in range(1, len(conv)):
            delay = conv[i]["timestamp"] - conv[i - 1]["timestamp"]
            delays.append(delay)
    return np.array(delays) if delays else np.array([])


def print_timing_summary(delays: np.ndarray, label: str = ""):
    """Print inter-turn delay statistics."""
    if len(delays) == 0:
        print("No multi-turn delays to analyze.")
        return

    title_suffix = f" - {label}" if label else ""
    print(f"\n{'=' * 60}")
    print(f"INTER-TURN TIMING ANALYSIS{title_suffix}")
    print(f"{'=' * 60}")
    print(f"\nTotal inter-turn gaps: {len(delays):,}")
    print(f"\nDelay statistics:")
    print(f"  Mean:   {delays.mean():.1f}s")
    print(f"  Median: {np.median(delays):.1f}s")
    print(f"  Std:    {delays.std():.1f}s")
    print(f"  Min:    {delays.min():.1f}s")
    print(f"  Max:    {delays.max():.1f}s")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(delays, p):.1f}s")

    print(f"\nDelay distribution:")
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60),
               (60, 120), (120, 300), (300, 600), (600, 1800)]
    for lo, hi in buckets:
        count = np.sum((delays >= lo) & (delays < hi))
        print(f"  {lo:>5}-{hi:<5}s: {count:>6,} ({100*count/len(delays):>5.1f}%)")
    count = np.sum(delays >= 1800)
    print(f"  1800+    s: {count:>6,} ({100*count/len(delays):>5.1f}%)")

    print(f"\nImplied --request-rate (1/median): {1/np.median(delays):.4f}")
    print(f"Implied --request-rate (1/mean):   {1/delays.mean():.4f}")
    print(f"{'=' * 60}")


def generate_timing_plot(delays: np.ndarray, output_path: Path, label: str = ""):
    """Generate inter-turn timing distribution plots."""
    if len(delays) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" - {label}" if label else ""
    fig.suptitle(f"Inter-Turn Delay Distribution{title_suffix} (n={len(delays):,})", fontsize=14)

    # (0,0) Histogram (linear scale, capped at 600s)
    ax = axes[0, 0]
    capped = delays[delays <= 600]
    ax.hist(capped, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(delays), color='red', linestyle='--',
               label=f'Median: {np.median(delays):.0f}s')
    ax.axvline(delays.mean(), color='orange', linestyle='--',
               label=f'Mean: {delays.mean():.0f}s')
    ax.set_xlabel("Delay (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Histogram (0-600s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1) Histogram (log scale x-axis)
    ax = axes[0, 1]
    log_bins = np.logspace(np.log10(max(delays.min(), 0.1)), np.log10(delays.max()), 50)
    ax.hist(delays, bins=log_bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xscale('log')
    ax.axvline(np.median(delays), color='red', linestyle='--',
               label=f'Median: {np.median(delays):.0f}s')
    ax.axvline(delays.mean(), color='orange', linestyle='--',
               label=f'Mean: {delays.mean():.0f}s')
    ax.set_xlabel("Delay (seconds, log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Histogram (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,0) CDF
    ax = axes[1, 0]
    sorted_delays = np.sort(delays)
    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
    ax.plot(sorted_delays, cdf, linewidth=2, color='steelblue')
    ax.fill_between(sorted_delays, cdf, alpha=0.2, color='steelblue')
    for p in [50, 75, 90, 95]:
        val = np.percentile(delays, p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'P{p}: {val:.0f}s', xy=(val, p/100),
                   xytext=(val * 1.1, p/100 - 0.04), fontsize=9)
    ax.set_xlabel("Delay (seconds)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Delay CDF")
    ax.set_xlim(0, np.percentile(delays, 99))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (1,1) CDF with exponential overlays for comparison
    ax = axes[1, 1]
    ax.plot(sorted_delays, cdf, linewidth=2, color='steelblue', label='Actual')
    # Overlay exponential CDFs for common request rates
    x = np.linspace(0, np.percentile(delays, 99), 500)
    for rate, color, ls in [(0.01, 'red', '--'), (0.05, 'orange', '--'),
                             (0.1, 'green', '--'), (0.2, 'purple', '--')]:
        exp_cdf = 1 - np.exp(-rate * x)
        ax.plot(x, exp_cdf, color=color, linestyle=ls, alpha=0.7,
               label=f'Exponential rate={rate} (mean={1/rate:.0f}s)')
    ax.set_xlabel("Delay (seconds)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Actual vs Exponential (Poisson) CDFs")
    ax.set_xlim(0, np.percentile(delays, 99))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = output_path.with_name(
        output_path.stem.replace("_analysis", "") + "_timing.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved timing plot to {out_file}")


def filter_conversations_by_type(conversations: list[list[dict]], allowed_types: set[str]) -> list[list[dict]]:
    """Filter conversations to only include turns of allowed types.

    Drops turns that don't match, then drops conversations that become empty.
    Multi-turn conversations are truncated at the first non-matching turn
    (to preserve conversation continuity).
    """
    filtered = []
    for conv in conversations:
        filtered_conv = []
        for turn in conv:
            if turn.get("type", "text") in allowed_types:
                filtered_conv.append(turn)
            else:
                # Stop at first non-matching turn to preserve chain integrity
                break
        if filtered_conv:
            filtered.append(filtered_conv)
    return filtered


def run_analysis(conversations: list[list[dict]], output_path: Path, label: str):
    """Run full analysis pipeline on a set of conversations."""
    print(f"\n{'#' * 70}")
    print(f"# {label}")
    print(f"{'#' * 70}")

    print_summary(conversations)
    generate_plots(conversations, output_path, label)

    turn_data = compute_sequence_metrics(conversations)
    print_sequence_summary(turn_data)
    generate_sequence_analysis_plot(turn_data, output_path, label)
    generate_isl_per_turn_plot(turn_data, conversations, output_path, label)

    # Inter-turn timing analysis
    delays = compute_inter_turn_delays(conversations)
    if len(delays) > 0:
        print_timing_summary(delays, label)
        generate_timing_plot(delays, output_path, label)


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen trace dataset conversation structure.")
    parser.add_argument("trace_file", type=str, help="Path to JSONL trace file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output path for the plot (default: <trace_name>_analysis.png)")
    args = parser.parse_args()

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: {trace_path} does not exist")
        return

    if args.output is None:
        output_path = trace_path.with_name(trace_path.stem + "_analysis.png")
    else:
        output_path = Path(args.output)

    records = load_trace(trace_path)
    conversations = reconstruct_conversations(records)

    # Full analysis (all types)
    run_analysis(conversations, output_path, "ALL REQUEST TYPES")

    # Text only analysis
    text_convs = filter_conversations_by_type(conversations, {"text"})
    print(f"\nFiltered to text only: {len(text_convs):,} conversations "
          f"(from {len(conversations):,})")
    text_output = output_path.with_name(output_path.stem.replace("_analysis", "") + "_text_only_analysis.png")
    run_analysis(text_convs, text_output, "TEXT ONLY")


if __name__ == "__main__":
    main()
