#!/usr/bin/env python3
"""Convert WildChat dataset to OpenAI format for vLLM multi-turn benchmarking."""

import argparse
import json
import random

from datasets import load_dataset
import tqdm


def process_conversation(conv: list[dict], conv_id: str) -> dict | None:
    """Convert raw conversation to OpenAI format with alternating user/assistant."""
    messages = []
    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    if not messages:
        return None

    # Ensure starts with user
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]

    if not messages:
        return None

    # Enforce alternating pattern
    clean = []
    expect_user = True
    for msg in messages:
        if expect_user and msg["role"] == "user":
            clean.append(msg)
            expect_user = False
        elif not expect_user and msg["role"] == "assistant":
            clean.append(msg)
            expect_user = True

    if len(clean) < 2:
        return None

    return {"id": conv_id, "messages": clean}


def load_wildchat(
    model_filter: str | None = None,
    language: str | None = "English",
    non_toxic: bool = True,
    min_turns: int = 4,
    max_items: int = 500,
) -> list[dict]:
    """Load and filter WildChat dataset using streaming."""
    print(f"Loading WildChat (streaming)...")
    print(f"  Filters: model={model_filter or 'any'}, lang={language or 'any'}, min_turns={min_turns}")

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    results = []
    checked = 0

    for row in tqdm.tqdm(ds, desc="Scanning"):
        checked += 1

        if language and row.get("language") != language:
            continue
        if non_toxic and row.get("toxic", False):
            continue
        if model_filter and model_filter.lower() not in row.get("model", "").lower():
            continue

        raw_conv = row.get("conversation", [])
        if len(raw_conv) < min_turns:
            continue

        result = process_conversation(raw_conv, row.get("conversation_hash", str(checked)))
        if result:
            results.append(result)
            if len(results) >= max_items:
                break

    print(f"  Scanned {checked:,} rows, found {len(results)} conversations")
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert WildChat to OpenAI format")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max-items", type=int, default=500, help="Number of conversations")
    parser.add_argument("--min-turns", type=int, default=4, help="Min messages per conversation")
    parser.add_argument("--model-filter", type=str, help="Filter by model name (substring)")
    parser.add_argument("--language", type=str, default="English", help="Language filter")
    parser.add_argument("--include-toxic", action="store_true", help="Include toxic conversations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    conversations = load_wildchat(
        model_filter=args.model_filter,
        language=args.language,
        non_toxic=not args.include_toxic,
        min_turns=args.min_turns,
        max_items=args.max_items,
    )

    if not conversations:
        print("ERROR: No conversations found!")
        return

    print(f"\nWriting {len(conversations)} conversations to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
