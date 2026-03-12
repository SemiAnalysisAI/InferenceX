#!/usr/bin/env python3
"""
Process WildChat dataset:
1. Filter out gpt-3.5-turbo, gpt-4, and gpt-4-turbo responses (keeps gpt-4o)
2. Add a column counting total user tokens across all turns

Usage:
    python process_wildchat.py --output wildchat_filtered.parquet
    python process_wildchat.py --output wildchat_filtered.parquet --max-items 10000
"""

import argparse

import tiktoken
from datasets import load_dataset


def should_filter_model(model: str) -> bool:
    """Return True if this model should be filtered out."""
    if model.startswith("gpt-3.5-turbo"):
        return True
    if model.startswith("gpt-4-turbo"):
        return True
    # Filter gpt-4* but keep gpt-4o*
    if model.startswith("gpt-4") and not model.startswith("gpt-4o"):
        return True
    return False


def count_user_tokens(conversation: list[dict], encoder: tiktoken.Encoding) -> int:
    """Count total tokens for all user messages in a conversation."""
    total = 0
    for msg in conversation:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                total += len(encoder.encode(content, disallowed_special=()))
    return total


def main():
    parser = argparse.ArgumentParser(description="Process WildChat dataset")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (parquet or json)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to keep after filtering",
    )
    args = parser.parse_args()

    print("Loading WildChat dataset...")
    ds = load_dataset("allenai/WildChat-4.8M", split="train")
    print(f"Loaded {len(ds):,} conversations")

    # Filter out specified models (gpt-3.5-turbo, gpt-4, gpt-4-turbo, but keep gpt-4o)
    print("Filtering out gpt-3.5-turbo, gpt-4, gpt-4-turbo (keeping gpt-4o)...")
    ds = ds.filter(
        lambda x: not should_filter_model(x["model"]),
        num_proc=8,
        desc="Filtering models",
    )
    print(f"After filtering: {len(ds):,} conversations")

    # Limit items if requested
    if args.max_items and len(ds) > args.max_items:
        print(f"Sampling {args.max_items:,} items...")
        ds = ds.shuffle(seed=42).select(range(args.max_items))

    # Add user token count column
    print("Counting user tokens...")
    encoder = tiktoken.get_encoding("cl100k_base")
    ds = ds.map(
        lambda x: {"user_token_count": count_user_tokens(x["conversation"], encoder)},
        num_proc=8,
        desc="Counting tokens",
    )

    # Save output
    print(f"Saving to {args.output}...")
    if args.output.endswith(".parquet"):
        ds.to_parquet(args.output)
    elif args.output.endswith(".json"):
        ds.to_json(args.output)
    else:
        raise ValueError("Output must be .parquet or .json")

    print("Done!")


if __name__ == "__main__":
    main()
