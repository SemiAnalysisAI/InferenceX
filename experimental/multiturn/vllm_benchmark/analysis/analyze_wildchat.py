#!/usr/bin/env python3
"""
Analyze WildChat dataset conversation structure.

Loads the full WildChat dataset from HuggingFace and generates:
- Histogram of turns per conversation
- CDF of turns per conversation

Usage:
    python analyze_wildchat.py
    python analyze_wildchat.py --output wildchat_analysis.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no display
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def load_wildchat() -> tuple[list[int], list[int]]:
    """Load inferencemax/WildChat-4.8M-4o-tokcount dataset and get turns and user tokens per conversation."""
    print("Loading inferencemax/WildChat-4.8M-4o-tokcount dataset from HuggingFace...")

    dataset = load_dataset("inferencemax/WildChat-4.8M-4o-tokcount", split="train")
    print(f"Loaded {len(dataset):,} conversations")

    # Columns already exist in dataset
    turns_per_conv = [t for t in dataset["turn"] if t > 0]
    user_tokens_per_conv = [t for t in dataset["user_token_count"] if t > 0]

    return turns_per_conv, user_tokens_per_conv


def generate_plots(turns_per_conv: list[int], user_tokens_per_conv: list[int], output_path: Path):
    """Generate histogram, CDF, and exceedance plots."""
    turns = np.array(turns_per_conv)
    user_tokens = np.array(user_tokens_per_conv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"WildChat: Conversation Statistics (n={len(turns):,})", fontsize=14)

    # Top-left: Histogram
    ax = axes[0, 0]
    max_turns = min(turns.max(), 20)  # Cap at 20 for readability
    bins = np.arange(0.5, max_turns + 1.5, 1)
    counts, _, bars = ax.hist(turns[turns <= max_turns], bins=bins, edgecolor='black', alpha=0.7)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{int(count):,}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Number of Conversations")
    ax.set_title("Histogram")
    ax.set_xticks(range(1, int(max_turns) + 1))
    ax.grid(True, alpha=0.3, axis='y')

    # Top-right: CDF
    ax = axes[0, 1]
    sorted_turns = np.sort(turns)
    cdf = np.arange(1, len(sorted_turns) + 1) / len(sorted_turns)
    ax.plot(sorted_turns, cdf, linewidth=2, color='steelblue')
    ax.fill_between(sorted_turns, cdf, alpha=0.3, color='steelblue')

    for p in [50, 75, 90, 95]:
        val = np.percentile(turns, p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'P{p}: {val:.0f}', xy=(val, p/100),
                   xytext=(val + 0.5, p/100 - 0.05), fontsize=9)

    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF")
    ax.set_xlim(0, max(20, np.percentile(turns, 99)))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Exceedance (conversations > X turns)
    ax = axes[1, 0]
    thresholds = range(1, 21)
    counts_above = [np.sum(turns > t) for t in thresholds]
    pct_above = [100 * c / len(turns) for c in counts_above]

    bars = ax.bar(thresholds, pct_above, edgecolor='black', alpha=0.7, color='coral')

    for bar, count, pct in zip(bars, counts_above, pct_above):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}% ({count:,})', ha='left', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel("Threshold (turns)")
    ax.set_ylabel("% of Conversations")
    ax.set_title("Conversations with > X Turns")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, max(pct_above) * 1.3)  # Extra room for rotated labels
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Exceedance (conversations > X user tokens)
    ax = axes[1, 1]
    token_thresholds = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    token_counts_above = [np.sum(user_tokens > t) for t in token_thresholds]
    token_pct_above = [100 * c / len(user_tokens) for c in token_counts_above]

    bars = ax.bar(range(len(token_thresholds)), token_pct_above, edgecolor='black', alpha=0.7, color='mediumseagreen')

    for bar, count, pct in zip(bars, token_counts_above, token_pct_above):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}% ({count:,})', ha='left', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel("Threshold (user tokens)")
    ax.set_ylabel("% of Conversations")
    ax.set_title("Conversations with > X User Tokens")
    ax.set_xticks(range(len(token_thresholds)))
    ax.set_xticklabels([f'{t:,}' for t in token_thresholds], rotation=45, ha='right')
    ax.set_ylim(0, max(token_pct_above) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def print_summary(turns_per_conv: list[int], user_tokens_per_conv: list[int]):
    """Print summary statistics."""
    turns = np.array(turns_per_conv)
    user_tokens = np.array(user_tokens_per_conv)
    n = len(turns)

    print("\n" + "=" * 50)
    print("WILDCHAT CONVERSATION STATISTICS")
    print("=" * 50)
    print(f"\nTotal conversations: {n:,}")

    print(f"\nTurns per conversation:")
    print(f"  Mean:   {turns.mean():.1f}")
    print(f"  Median: {np.median(turns):.0f}")
    print(f"  Std:    {turns.std():.1f}")
    print(f"  Min:    {turns.min()}")
    print(f"  Max:    {turns.max()}")
    print(f"\nTurns percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(turns, p):.0f} turns")
    print(f"\nConversations exceeding turn threshold:")
    for t in [2, 3, 5, 10, 15, 20]:
        count = np.sum(turns > t)
        print(f"  > {t:2d} turns: {count:>10,} ({100*count/n:>5.1f}%)")

    print(f"\nUser tokens per conversation:")
    print(f"  Mean:   {user_tokens.mean():.1f}")
    print(f"  Median: {np.median(user_tokens):.0f}")
    print(f"  Std:    {user_tokens.std():.1f}")
    print(f"  Min:    {user_tokens.min()}")
    print(f"  Max:    {user_tokens.max()}")
    print(f"\nUser tokens percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(user_tokens, p):,.0f} tokens")
    print(f"\nConversations exceeding token threshold:")
    for t in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
        count = np.sum(user_tokens > t)
        print(f"  > {t:>5,} tokens: {count:>10,} ({100*count/n:>5.1f}%)")
    print("=" * 50)


def main(output_path: Path):
    turns_per_conv, user_tokens_per_conv = load_wildchat()

    if not turns_per_conv:
        print("No conversation data found!")
        return

    print_summary(turns_per_conv, user_tokens_per_conv)
    generate_plots(turns_per_conv, user_tokens_per_conv, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze turns per conversation in WildChat dataset."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="wildchat_turns.png",
        help="Output path for the plot (default: wildchat_turns.png)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    main(output_path)
