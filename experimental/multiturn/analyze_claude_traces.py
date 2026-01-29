#!/usr/bin/env python3
"""
Analyze the Claude Code Traces dataset for multi-turn conversation statistics.

Dataset: https://huggingface.co/datasets/nlile/misc-merged-claude-code-traces-v1

This script downloads the dataset and analyzes:
- Number of turns per conversation
- Distribution of turn counts
- Message role patterns (user/assistant alternation)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import tiktoken


def get_tokenizer():
    """
    Get a tokenizer for counting tokens.

    We use tiktoken's cl100k_base encoding (used by GPT-4, GPT-3.5-turbo).
    This is an APPROXIMATION - different models use different tokenizers:
    - Claude uses its own BPE tokenizer (not public)
    - Llama uses SentencePiece
    - Qwen, Mistral, etc. each have their own

    For benchmarking purposes, cl100k_base gives reasonable estimates.
    Actual token counts may vary ±10-20% depending on the target model.
    """
    return tiktoken.get_encoding("cl100k_base")


def message_to_string(msg: dict) -> str:
    """
    Convert a single message to its string representation.

    This simulates what a chat template does - converting the structured
    message format into text that will be tokenized.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    # Handle content that's a list (contains text, tool_use, tool_result)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "tool_use":
                    # Tool calls get serialized as structured text
                    tool_str = json.dumps({
                        "name": item.get("name"),
                        "input": item.get("input", {})
                    }, indent=2)
                    parts.append(f"<tool_use>\n{tool_str}\n</tool_use>")
                elif item_type == "tool_result":
                    # Tool results include the full output
                    result_content = item.get("content", "")
                    parts.append(f"<tool_result>\n{result_content}\n</tool_result>")
        content = "\n".join(parts)

    return content


def messages_to_prompt(messages: list, up_to_index: int | None = None) -> str:
    """
    Convert a list of messages to a single prompt string.

    This simulates applying a chat template. The exact format varies by model,
    but we use a generic format similar to Llama/ChatML:

    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {user message}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {assistant message}
    <|eot_id|>
    ...

    Args:
        messages: List of message dicts with 'role' and 'content'
        up_to_index: If provided, only include messages up to this index (exclusive)
    """
    if up_to_index is not None:
        messages = messages[:up_to_index]

    parts = ["<|begin_of_text|>"]

    for msg in messages:
        role = msg.get("role", "unknown")
        content = message_to_string(msg)

        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

    return "\n".join(parts)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text, disallowed_special=()))


def analyze_token_growth(messages: list, tokenizer) -> list[dict]:
    """
    Analyze how token count grows with each turn.

    For multi-turn conversations, the KEY insight is:
    - Turn N's INPUT = all messages from turn 0 to N-1 (the context/prefill)
    - Turn N's OUTPUT = the assistant's response at turn N

    Returns a list of dicts with per-turn token stats.
    """
    results = []

    # Track turn number (a turn = user message + assistant response)
    turn_num = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content_str = message_to_string(msg)
        msg_tokens = count_tokens(content_str, tokenizer)

        # Cumulative context = all messages up to and including this one
        cumulative_prompt = messages_to_prompt(messages, up_to_index=i + 1)
        cumulative_tokens = count_tokens(cumulative_prompt, tokenizer)

        result = {
            "message_index": i,
            "role": role,
            "message_tokens": msg_tokens,
            "cumulative_tokens": cumulative_tokens,
        }

        # For assistant messages, this is the OUTPUT of that turn
        # The INPUT was all previous messages
        if role == "assistant":
            input_tokens = count_tokens(messages_to_prompt(messages, up_to_index=i), tokenizer)
            result["turn"] = turn_num
            result["input_tokens"] = input_tokens  # ISL for this turn
            result["output_tokens"] = msg_tokens    # OSL for this turn
            turn_num += 1

        results.append(result)

    return results


def load_dataset(filter_empty_responses: bool = True):
    """Load the Claude Code Traces dataset from HuggingFace.

    Args:
        filter_empty_responses: If True, filter out records where has_empty_response=True
                                (incomplete conversations with no assistant response)
    """
    from datasets import load_dataset

    print("Downloading dataset from HuggingFace...")
    print("(This may take a few minutes for the first download)")

    dataset = load_dataset(
        "nlile/misc-merged-claude-code-traces-v1",
        split="train",
    )
    print(f"Loaded {len(dataset)} total records")

    if filter_empty_responses:
        # Filter out incomplete conversations (no assistant response)
        # See dataset card: ~20% have empty responses
        original_count = len(dataset)
        dataset = dataset.filter(lambda x: not x.get("has_empty_response", False))
        filtered_count = original_count - len(dataset)
        print(f"Filtered out {filtered_count} records with empty responses")
        print(f"Remaining: {len(dataset)} complete conversations")

    return dataset


def count_turns(messages_json: str) -> dict:
    """
    Count turns in a conversation from the messages_json field.

    Returns a dict with:
    - total_messages: total number of messages
    - user_messages: count of user messages
    - assistant_messages: count of assistant messages
    - turns: number of user-assistant turn pairs
    - roles: list of roles in order
    """
    if not messages_json or messages_json == "null":
        return {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "turns": 0,
            "roles": [],
            "has_tools": False,
        }

    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "turns": 0,
            "roles": [],
            "has_tools": False,
            "parse_error": True,
        }

    if not isinstance(messages, list):
        return {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "turns": 0,
            "roles": [],
            "has_tools": False,
        }

    roles = []
    has_tools = False

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            roles.append(role)
            # Check for tool use
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["tool_use", "tool_result"]:
                        has_tools = True

    user_count = roles.count("user")
    assistant_count = roles.count("assistant")

    # A "turn" is typically defined as a user message followed by an assistant response
    # We count the minimum of user/assistant messages as the number of complete turns
    turns = min(user_count, assistant_count)

    return {
        "total_messages": len(messages),
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "turns": turns,
        "roles": roles,
        "has_tools": has_tools,
    }


def analyze_dataset(dataset) -> dict:
    """Analyze the full dataset for turn statistics."""

    turn_counts = []
    message_counts = []
    tool_use_count = 0
    parse_errors = 0
    role_patterns = Counter()

    print("Analyzing conversations...")

    for i, record in enumerate(dataset):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(dataset)} records...")

        messages_json = record.get("messages_json", "")
        stats = count_turns(messages_json)

        if stats.get("parse_error"):
            parse_errors += 1
            continue

        turn_counts.append(stats["turns"])
        message_counts.append(stats["total_messages"])

        if stats["has_tools"]:
            tool_use_count += 1

        # Track role patterns (first 10 roles to keep it manageable)
        pattern = "->".join(stats["roles"][:10])
        if len(stats["roles"]) > 10:
            pattern += "->..."
        role_patterns[pattern] += 1

    turn_counts = np.array(turn_counts)
    message_counts = np.array(message_counts)

    return {
        "total_records": len(dataset),
        "parse_errors": parse_errors,
        "valid_records": len(turn_counts),
        "turn_counts": turn_counts,
        "message_counts": message_counts,
        "tool_use_count": tool_use_count,
        "role_patterns": role_patterns,
    }


def print_statistics(analysis: dict):
    """Print summary statistics."""

    turn_counts = analysis["turn_counts"]
    message_counts = analysis["message_counts"]

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records:        {analysis['total_records']:,}")
    print(f"Valid records:        {analysis['valid_records']:,}")
    print(f"Parse errors:         {analysis['parse_errors']:,}")
    print(f"Records with tools:   {analysis['tool_use_count']:,} ({100*analysis['tool_use_count']/analysis['valid_records']:.1f}%)")

    print("\n" + "-" * 60)
    print("TURN COUNT STATISTICS")
    print("-" * 60)
    print(f"Min turns:            {turn_counts.min()}")
    print(f"Max turns:            {turn_counts.max()}")
    print(f"Mean turns:           {turn_counts.mean():.2f}")
    print(f"Median turns:         {np.median(turn_counts):.1f}")
    print(f"Std dev:              {turn_counts.std():.2f}")

    print("\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile:     {np.percentile(turn_counts, p):.0f} turns")

    print("\n" + "-" * 60)
    print("TURN COUNT DISTRIBUTION")
    print("-" * 60)

    # Bin the turn counts
    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20-49", "50-99", "100+"]

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if high == float('inf'):
            count = np.sum(turn_counts >= low)
        else:
            count = np.sum((turn_counts >= low) & (turn_counts < high))
        pct = 100 * count / len(turn_counts)
        bar = "#" * int(pct / 2)
        print(f"  {bin_labels[i]:>6} turns: {count:>6,} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 60)
    print("MESSAGE COUNT STATISTICS")
    print("-" * 60)
    print(f"Min messages:         {message_counts.min()}")
    print(f"Max messages:         {message_counts.max()}")
    print(f"Mean messages:        {message_counts.mean():.2f}")
    print(f"Median messages:      {np.median(message_counts):.1f}")

    print("\n" + "-" * 60)
    print("TOP 10 ROLE PATTERNS (first 10 messages)")
    print("-" * 60)
    for pattern, count in analysis["role_patterns"].most_common(10):
        pct = 100 * count / analysis["valid_records"]
        print(f"  {count:>6,} ({pct:>5.1f}%): {pattern}")


def plot_distribution(analysis: dict, output_path: Path):
    """Create visualization of turn distribution."""

    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not installed, skipping visualization.")
        return

    turn_counts = analysis["turn_counts"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of turn counts (capped at 100 for visibility)
    ax1 = axes[0]
    capped_turns = np.minimum(turn_counts, 100)
    ax1.hist(capped_turns, bins=100, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Number of Turns (capped at 100)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Turns per Conversation")
    ax1.axvline(turn_counts.mean(), color='red', linestyle='--', label=f'Mean: {turn_counts.mean():.1f}')
    ax1.axvline(np.median(turn_counts), color='green', linestyle='--', label=f'Median: {np.median(turn_counts):.1f}')
    ax1.legend()

    # CDF
    ax2 = axes[1]
    sorted_turns = np.sort(turn_counts)
    cdf = np.arange(1, len(sorted_turns) + 1) / len(sorted_turns)
    ax2.plot(sorted_turns, cdf)
    ax2.set_xlabel("Number of Turns")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("CDF of Turns per Conversation")
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Add percentile markers
    for p in [50, 90, 95, 99]:
        val = np.percentile(turn_counts, p)
        ax2.axhline(p/100, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(f'p{p}={val:.0f}', (val, p/100), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")


def sample_by_turns(dataset, num_turns: int) -> dict | None:
    """Return a randomly sampled record with exactly num_turns turns."""
    import random

    matching_records = []

    for record in dataset:
        messages_json = record.get("messages_json", "")
        stats = count_turns(messages_json)
        if stats["turns"] == num_turns:
            matching_records.append(record)

    if not matching_records:
        return None

    return random.choice(matching_records)


def print_sample(record: dict, output_dir: Path):
    """Pretty print a sampled record and save full JSON with token analysis."""
    # Convert record to a plain dict and parse JSON fields
    record_dict = dict(record)

    # Parse JSON string fields into actual objects for readability
    for field in ["messages_json", "tools_json"]:
        if record_dict.get(field) and record_dict[field] != "null":
            try:
                record_dict[field] = json.loads(record_dict[field])
            except json.JSONDecodeError:
                pass  # Keep as string if parsing fails

    messages = record_dict.get("messages_json", [])

    # Perform token analysis if we have messages
    token_analysis = None
    tokenizer = None
    if isinstance(messages, list) and messages:
        tokenizer = get_tokenizer()
        token_analysis = analyze_token_growth(messages, tokenizer)

        # Add token analysis to the record for the JSON output
        record_dict["_token_analysis"] = token_analysis

    # Save full record to JSON file
    output_path = output_dir / "sampled_record.json"
    with open(output_path, "w") as f:
        json.dump(record_dict, f, indent=2, default=str)
    print(f"Full record saved to: {output_path}")
    print()

    # Print summary to console
    print("=" * 60)
    print("SAMPLED RECORD SUMMARY")
    print("=" * 60)
    print(f"ID: {record.get('id')}")
    print(f"Model: {record.get('model')}")
    print(f"Timestamp: {record.get('timestamp')}")
    print(f"Has tools: {bool(record.get('tools_json'))}")
    print()

    if isinstance(messages, list) and messages:
        print(f"Total messages: {len(messages)}")

        # Print token analysis
        if token_analysis:
            print()
            print("=" * 60)
            print("TOKEN ANALYSIS")
            print("=" * 60)
            print()
            print("How tokenization works:")
            print("  1. Each message is converted to text via a 'chat template'")
            print("  2. The text is tokenized using a BPE tokenizer (we use tiktoken cl100k)")
            print("  3. For each turn, INPUT = all previous context, OUTPUT = assistant response")
            print()
            print("Note: Token counts are approximate. Different models use different tokenizers.")
            print("      Actual counts may vary ±10-20% for Claude, Llama, Qwen, etc.")
            print()
            print("-" * 60)
            print("PER-MESSAGE TOKEN BREAKDOWN:")
            print("-" * 60)
            print(f"{'Msg':>4} {'Role':>10} {'Msg Tokens':>12} {'Cumulative':>12}")
            print("-" * 60)

            for item in token_analysis:
                print(f"{item['message_index']+1:>4} {item['role']:>10} {item['message_tokens']:>12,} {item['cumulative_tokens']:>12,}")

            print()
            print("-" * 60)
            print("PER-TURN ISL/OSL (Input Sequence Length / Output Sequence Length):")
            print("-" * 60)
            print(f"{'Turn':>4} {'ISL (Input)':>14} {'OSL (Output)':>14} {'Total':>14}")
            print("-" * 60)

            for item in token_analysis:
                if "turn" in item:
                    total = item['input_tokens'] + item['output_tokens']
                    print(f"{item['turn']:>4} {item['input_tokens']:>14,} {item['output_tokens']:>14,} {total:>14,}")

            # Summary stats
            turns_data = [item for item in token_analysis if "turn" in item]
            if turns_data:
                print()
                print("-" * 60)
                print("SUMMARY FOR BENCHMARKING:")
                print("-" * 60)
                isls = [t['input_tokens'] for t in turns_data]
                osls = [t['output_tokens'] for t in turns_data]
                print(f"  First turn ISL:  {isls[0]:>10,} tokens")
                print(f"  Last turn ISL:   {isls[-1]:>10,} tokens  (context growth: {isls[-1]/isls[0]:.1f}x)")
                print(f"  Avg ISL:         {sum(isls)/len(isls):>10,.0f} tokens")
                print(f"  Avg OSL:         {sum(osls)/len(osls):>10,.0f} tokens")
                print(f"  Total tokens:    {token_analysis[-1]['cumulative_tokens']:>10,} tokens")

        print()
        print("-" * 60)
        print("CONVERSATION OVERVIEW (truncated):")
        print("-" * 60)
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content that's a list (tool use, etc.)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "unknown")
                        if item_type == "text":
                            text = item.get("text", "")
                            if len(text) > 200:
                                text = text[:200] + "..."
                            text_parts.append(text)
                        elif item_type == "tool_use":
                            tool_name = item.get("name", "unknown")
                            text_parts.append(f"[TOOL_USE: {tool_name}]")
                        elif item_type == "tool_result":
                            text_parts.append(f"[TOOL_RESULT: ...]")
                content = " | ".join(text_parts)
            elif isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."

            # Show token count for this message
            msg_tokens = token_analysis[i]['message_tokens'] if token_analysis else "?"
            print(f"\n[{i+1}] {role.upper()} ({msg_tokens:,} tokens):")
            print(f"  {content[:300]}")
    else:
        print("No messages_json available")
        print(f"\nUser prompt: {record.get('user_prompt', '')[:500]}")
        print(f"\nAssistant response: {record.get('assistant_response', '')[:500]}")

    print()
    print(f"See {output_path} for full record including complete tool inputs/outputs.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Claude Code Traces dataset")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include records with empty assistant responses (default: filter them out)",
    )
    parser.add_argument(
        "--sample-turns",
        type=int,
        metavar="N",
        help="Sample a random record with exactly N turns and print it",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(filter_empty_responses=not args.include_empty)

    # If sampling mode, just sample and exit
    if args.sample_turns is not None:
        print(f"Searching for record with {args.sample_turns} turns...")
        record = sample_by_turns(dataset, args.sample_turns)
        if record:
            output_dir = Path(__file__).parent
            print_sample(record, output_dir)
        else:
            print(f"No records found with exactly {args.sample_turns} turns")
        return

    # Analyze
    analysis = analyze_dataset(dataset)

    # Print statistics
    print_statistics(analysis)

    # Save visualization
    output_dir = Path(__file__).parent
    plot_distribution(analysis, output_dir / "turn_distribution.png")

    # Save raw statistics to JSON for later use
    stats_output = {
        "total_records": analysis["total_records"],
        "valid_records": analysis["valid_records"],
        "parse_errors": analysis["parse_errors"],
        "tool_use_count": analysis["tool_use_count"],
        "turn_stats": {
            "min": int(analysis["turn_counts"].min()),
            "max": int(analysis["turn_counts"].max()),
            "mean": float(analysis["turn_counts"].mean()),
            "median": float(np.median(analysis["turn_counts"])),
            "std": float(analysis["turn_counts"].std()),
            "percentiles": {
                str(p): float(np.percentile(analysis["turn_counts"], p))
                for p in [25, 50, 75, 90, 95, 99]
            },
        },
        "message_stats": {
            "min": int(analysis["message_counts"].min()),
            "max": int(analysis["message_counts"].max()),
            "mean": float(analysis["message_counts"].mean()),
            "median": float(np.median(analysis["message_counts"])),
        },
    }

    stats_path = output_dir / "turn_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


if __name__ == "__main__":
    main()
