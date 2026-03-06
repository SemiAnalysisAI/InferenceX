#!/usr/bin/env python3
"""
Analyze input/output sequence lengths from benchmark responses.

Computes per-turn metrics:
- ISL (Input Sequence Length): cumulative tokens in conversation context
- OSL (Output Sequence Length): tokens generated per turn
- ISL increment: how much the input grows per turn
- Relationship between OSL and next turn's ISL increment

Usage:
    python analyze_sequences.py <results_dir>
    python analyze_sequences.py ~/sweep_results_20260204_062339
    python analyze_sequences.py ~/sweep_results_20260204_062339 --encoding o200k_base
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm


class Tokenizer:
    """Wrapper for tiktoken tokenizer with caching."""

    def __init__(self, encoding: str = "cl100k_base"):
        """
        Initialize tokenizer.

        Args:
            encoding: tiktoken encoding name. Options:
                - "cl100k_base": GPT-4/GPT-3.5-turbo tokenizer (~100k vocab, good approximation for Llama 3)
                - "o200k_base": GPT-4o tokenizer (~200k vocab)
        """
        print(f"Loading tiktoken encoding: {encoding}")
        self.tokenizer = tiktoken.get_encoding(encoding)
        self._cache = {}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not text:
            return 0
        # Use cache to avoid re-tokenizing identical strings
        if text not in self._cache:
            self._cache[text] = len(self.tokenizer.encode(text))
        return self._cache[text]


def analyze_conversation(conversation: dict, tokenizer: Tokenizer) -> list[dict]:
    """Analyze a single conversation's sequence lengths per turn."""
    turns = conversation.get("turns", [])
    if not turns:
        return []

    results = []
    cumulative_isl = 0  # Running total of input context

    for turn_idx, turn in enumerate(turns):
        user_text = turn.get("user", "")
        response_text = turn.get("server_response", "")

        user_tokens = tokenizer.count_tokens(user_text)
        response_tokens = tokenizer.count_tokens(response_text)

        # ISL for this turn = previous cumulative + new user input
        # (The model sees all previous context + new user message)
        turn_isl = cumulative_isl + user_tokens

        # ISL increment from previous turn
        isl_increment = user_tokens if turn_idx == 0 else (turn_isl - results[-1]["isl"])

        results.append({
            "turn_idx": turn_idx,
            "user_tokens": user_tokens,
            "response_tokens": response_tokens,
            "isl": turn_isl,  # Input to model for this turn
            "osl": response_tokens,  # Output from model for this turn
            "isl_increment": isl_increment,
        })

        # Update cumulative for next turn: add user input + model response
        cumulative_isl = turn_isl + response_tokens

    # Compute prev_osl for relationship analysis (OSL of turn N vs ISL increment of turn N+1)
    for i in range(1, len(results)):
        results[i]["prev_osl"] = results[i - 1]["osl"]
        # ISL increment should be ~= prev_osl + new_user_tokens
        results[i]["isl_increment_minus_prev_osl"] = results[i]["isl_increment"] - results[i - 1]["osl"]

    return results


def analyze_results_dir(results_dir: Path, tokenizer: Tokenizer) -> pd.DataFrame:
    """Analyze all responses.json files in a results directory."""
    all_turn_data = []

    # Find all experiment directories
    exp_dirs = [
        d for d in sorted(results_dir.iterdir())
        if d.is_dir() and d.name.startswith("tp") and (d / "responses.json").exists()
    ]

    if not exp_dirs:
        return pd.DataFrame()

    # First pass: count total conversations for progress bar
    total_conversations = 0
    exp_conversations = {}
    for exp_dir in exp_dirs:
        try:
            with open(exp_dir / "responses.json") as f:
                convs = json.load(f)
                exp_conversations[exp_dir] = convs
                total_conversations += len(convs)
        except Exception as e:
            print(f"Error loading {exp_dir / 'responses.json'}: {e}")

    # Second pass: analyze with progress bar
    with tqdm(total=total_conversations, desc="Analyzing conversations") as pbar:
        for exp_dir, conversations in exp_conversations.items():
            # Parse experiment config from name
            parts = exp_dir.name.split("_")
            tp = int(parts[0].replace("tp", ""))
            bs = int(parts[1].replace("bs", ""))
            offload = parts[2].replace("offload", "")

            for conv_idx, conv in enumerate(conversations):
                turn_data = analyze_conversation(conv, tokenizer)
                for turn in turn_data:
                    turn["exp_name"] = exp_dir.name
                    turn["tp"] = tp
                    turn["bs"] = bs
                    turn["offload"] = offload
                    turn["conv_idx"] = conv_idx
                    turn["conv_hash"] = conv.get("conversation_hash", "")
                    all_turn_data.append(turn)
                pbar.update(1)

    return pd.DataFrame(all_turn_data)


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal turns analyzed: {len(df):,}")
    print(f"Total conversations: {df['conv_hash'].nunique():,}")
    print(f"Experiments: {df['exp_name'].nunique()}")

    print("\n--- Per-Turn Statistics (across all experiments) ---")
    print(f"\nInput Sequence Length (ISL) - tokens seen by model:")
    print(f"  Mean:   {df['isl'].mean():,.0f}")
    print(f"  Median: {df['isl'].median():,.0f}")
    print(f"  Std:    {df['isl'].std():,.0f}")
    print(f"  Min:    {df['isl'].min():,.0f}")
    print(f"  Max:    {df['isl'].max():,.0f}")

    print(f"\nOutput Sequence Length (OSL) - tokens generated:")
    print(f"  Mean:   {df['osl'].mean():,.0f}")
    print(f"  Median: {df['osl'].median():,.0f}")
    print(f"  Std:    {df['osl'].std():,.0f}")
    print(f"  Min:    {df['osl'].min():,.0f}")
    print(f"  Max:    {df['osl'].max():,.0f}")

    print(f"\nISL Increment per turn:")
    print(f"  Mean:   {df['isl_increment'].mean():,.0f}")
    print(f"  Median: {df['isl_increment'].median():,.0f}")
    print(f"  Std:    {df['isl_increment'].std():,.0f}")

    # Per-turn analysis
    print("\n--- Statistics by Turn Number ---")
    turn_stats = df.groupby("turn_idx").agg({
        "isl": ["mean", "std", "count"],
        "osl": ["mean", "std"],
        "isl_increment": ["mean", "std"],
        "user_tokens": ["mean"],
    }).round(1)
    turn_stats.columns = ['_'.join(col).strip() for col in turn_stats.columns.values]
    print(turn_stats.head(10).to_string())

    # ISL increment vs previous OSL relationship
    df_with_prev = df[df["turn_idx"] > 0].copy()
    if len(df_with_prev) > 0 and "isl_increment_minus_prev_osl" in df_with_prev.columns:
        print("\n--- ISL Increment Breakdown (turns > 0) ---")
        print("Formula: ISL_increment = prev_OSL + new_user_tokens")
        print("         (input grows by: previous response + new user message)")
        print(f"\nISL Increment (total input growth per turn):")
        print(f"  Mean:   {df_with_prev['isl_increment'].mean():,.0f}")
        print(f"  Median: {df_with_prev['isl_increment'].median():,.0f}")
        print(f"\nPrevious OSL (previous response length):")
        print(f"  Mean:   {df_with_prev['prev_osl'].mean():,.0f}")
        print(f"  Median: {df_with_prev['prev_osl'].median():,.0f}")
        print(f"\nNew User Tokens (ISL_increment - prev_OSL):")
        print(f"  Mean:   {df_with_prev['isl_increment_minus_prev_osl'].mean():,.0f}")
        print(f"  Median: {df_with_prev['isl_increment_minus_prev_osl'].median():,.0f}")
        print(f"  (Actual user_tokens mean: {df_with_prev['user_tokens'].mean():,.0f})")
        print(f"\nRatio: prev_OSL / ISL_increment (how much of increment is from prev response):")
        ratio = df_with_prev['prev_osl'] / df_with_prev['isl_increment']
        print(f"  Mean:   {ratio.mean():.1%}")
        print(f"  Median: {ratio.median():.1%}")

    print("\n" + "=" * 70)


def generate_isl_per_turn_chart(df: pd.DataFrame, output_dir: Path):
    """Generate a clear ISL per turn chart."""
    # Aggregate by turn
    turn_stats = df.groupby("turn_idx").agg({
        "isl": ["mean", "std", "count"],
        "user_tokens": "mean",
        "response_tokens": "mean",
    })
    turn_stats.columns = ['isl_mean', 'isl_std', 'count', 'user_tokens_mean', 'response_tokens_mean']
    turn_stats = turn_stats.reset_index()

    # Only show turns with reasonable sample size
    turn_stats = turn_stats[turn_stats["count"] >= 100]
    max_turns = min(15, len(turn_stats))
    turn_stats = turn_stats.head(max_turns)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Input Sequence Length (ISL) Per Turn", fontsize=14)

    # Left plot: ISL growth as line chart with error bars
    ax = axes[0]
    ax.errorbar(turn_stats["turn_idx"], turn_stats["isl_mean"],
                yerr=turn_stats["isl_std"], marker='o', capsize=4,
                linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(turn_stats["turn_idx"],
                    turn_stats["isl_mean"] - turn_stats["isl_std"],
                    turn_stats["isl_mean"] + turn_stats["isl_std"],
                    alpha=0.2, color='steelblue')

    # Add value labels
    for _, row in turn_stats.iterrows():
        ax.annotate(f'{row["isl_mean"]:.0f}',
                   (row["turn_idx"], row["isl_mean"]),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9)

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL (tokens)")
    ax.set_title("Cumulative Input Length Per Turn\n(what the model sees)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_stats["turn_idx"])

    # Right plot: Stacked bar showing composition
    ax = axes[1]
    turns = turn_stats["turn_idx"].values

    # Calculate cumulative components
    cumulative_user = []
    cumulative_response = []
    running_user = 0
    running_response = 0

    for i, row in turn_stats.iterrows():
        running_user += row["user_tokens_mean"]
        cumulative_user.append(running_user)
        if i > 0:
            # Add previous turn's response
            prev_response = turn_stats.iloc[i-1]["response_tokens_mean"]
            running_response += prev_response
        cumulative_response.append(running_response)

    # Stacked bar chart
    ax.bar(turns, cumulative_user, label='Cumulative User Input', color='forestgreen', alpha=0.8)
    ax.bar(turns, cumulative_response, bottom=cumulative_user, label='Cumulative Model Responses', color='coral', alpha=0.8)

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Tokens")
    ax.set_title("ISL Composition\n(user input vs model responses)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(turns)

    plt.tight_layout()
    output_file = output_dir / "isl_per_turn.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved ISL per turn chart to {output_file}")
    plt.close()


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Sequence Length Analysis", fontsize=14)

    # 1. ISL distribution
    ax = axes[0, 0]
    ax.hist(df["isl"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(df["isl"].mean(), color="red", linestyle="--", label=f"Mean: {df['isl'].mean():.0f}")
    ax.axvline(df["isl"].median(), color="green", linestyle="--", label=f"Median: {df['isl'].median():.0f}")
    ax.set_xlabel("ISL (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Input Sequence Length Distribution")
    ax.legend()

    # 2. OSL distribution
    ax = axes[0, 1]
    ax.hist(df["osl"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(df["osl"].mean(), color="red", linestyle="--", label=f"Mean: {df['osl'].mean():.0f}")
    ax.axvline(df["osl"].median(), color="green", linestyle="--", label=f"Median: {df['osl'].median():.0f}")
    ax.set_xlabel("OSL (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Output Sequence Length Distribution")
    ax.legend()

    # 3. ISL increment distribution
    ax = axes[0, 2]
    ax.hist(df["isl_increment"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(df["isl_increment"].mean(), color="red", linestyle="--", label=f"Mean: {df['isl_increment'].mean():.0f}")
    ax.set_xlabel("ISL Increment (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("ISL Increment per Turn")
    ax.legend()

    # 4. ISL vs Turn Number
    ax = axes[1, 0]
    turn_means = df.groupby("turn_idx")["isl"].agg(["mean", "std"])
    turns = turn_means.index[:15]  # First 15 turns
    ax.errorbar(turns, turn_means.loc[turns, "mean"], yerr=turn_means.loc[turns, "std"],
                marker="o", capsize=3, alpha=0.7)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL (tokens)")
    ax.set_title("ISL Growth by Turn")
    ax.grid(True, alpha=0.3)

    # 5. OSL vs Turn Number
    ax = axes[1, 1]
    turn_osl = df.groupby("turn_idx")["osl"].agg(["mean", "std"])
    ax.errorbar(turns, turn_osl.loc[turns, "mean"], yerr=turn_osl.loc[turns, "std"],
                marker="o", capsize=3, alpha=0.7, color="orange")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean OSL (tokens)")
    ax.set_title("OSL by Turn")
    ax.grid(True, alpha=0.3)

    # 6. ISL increment vs Turn Number
    ax = axes[1, 2]
    turn_incr = df.groupby("turn_idx")["isl_increment"].agg(["mean", "std"])
    ax.errorbar(turns, turn_incr.loc[turns, "mean"], yerr=turn_incr.loc[turns, "std"],
                marker="o", capsize=3, alpha=0.7, color="green")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL Increment (tokens)")
    ax.set_title("ISL Increment by Turn")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "sequence_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved plots to {output_file}")
    plt.close()


def main(results_dir: Path, encoding: str):
    print(f"Analyzing responses in: {results_dir}")

    # Initialize tokenizer
    tokenizer = Tokenizer(encoding)

    df = analyze_results_dir(results_dir, tokenizer)

    if len(df) == 0:
        print("No response data found!")
        return

    # Print summary
    print_summary_stats(df)

    # Generate plots
    generate_plots(df, results_dir)
    generate_isl_per_turn_chart(df, results_dir)

    # Save detailed CSV
    csv_file = results_dir / "sequence_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved detailed data to {csv_file}")

    # Save per-turn summary
    turn_summary = df.groupby("turn_idx").agg({
        "isl": ["mean", "median", "std", "min", "max", "count"],
        "osl": ["mean", "median", "std", "min", "max"],
        "isl_increment": ["mean", "median", "std"],
        "user_tokens": ["mean", "median"],
    }).round(2)
    turn_summary.columns = ['_'.join(col) for col in turn_summary.columns]
    turn_summary_file = results_dir / "sequence_analysis_by_turn.csv"
    turn_summary.to_csv(turn_summary_file)
    print(f"Saved per-turn summary to {turn_summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze input/output sequence lengths from benchmark responses."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results directory containing experiment folders"
    )
    parser.add_argument(
        "--encoding", "-e",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding (default: cl100k_base, alternatives: o200k_base)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    main(results_dir, args.encoding)
