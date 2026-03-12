#!/usr/bin/env python3
"""Extract full multi-turn conversations from AIPerf raw export records.

Reconstructs the complete conversation history (user + assistant messages)
from the raw JSONL records exported by AIPerf with --export-level raw.

Usage:
    python extract_conversations.py path/to/profile_export_raw.jsonl
    python extract_conversations.py path/to/profile_export_raw.jsonl -o conversations.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_conversations(raw_jsonl_path: str) -> list[dict]:
    """Extract conversations from raw export JSONL.

    Each line in the JSONL is a record for one request (one turn).
    The payload.messages field contains the full conversation history
    up to and including that turn's user message.
    The responses field contains the server's response for that turn.

    We find the last turn of each conversation (which has the fullest
    history) and append its assistant response to reconstruct the
    complete conversation.
    """
    # Group records by conversation_id, keeping track of turn index
    conversations: dict[str, list[dict]] = defaultdict(list)

    with open(raw_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            metadata = record.get("metadata", {})
            conv_id = metadata.get("conversation_id", "unknown")
            turn_idx = metadata.get("turn_index", 0)

            # Extract the messages sent to the API (full history for this turn)
            payload = record.get("payload", {})
            messages = payload.get("messages", [])

            # Extract assistant response text from SSE responses
            response_text = ""
            responses = record.get("responses", [])
            if responses:
                # Responses are SSE chunks or full text responses
                chunks = []
                for resp in responses:
                    if isinstance(resp, dict):
                        # SSE format: look for content in choices
                        choices = resp.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunks.append(content)
                            # Non-streaming format
                            msg = choice.get("message", {})
                            content = msg.get("content", "")
                            if content:
                                chunks.append(content)
                    elif isinstance(resp, str):
                        chunks.append(resp)
                response_text = "".join(chunks)

            conversations[conv_id].append({
                "turn_index": turn_idx,
                "messages": messages,
                "assistant_response": response_text,
                "max_tokens": payload.get("max_completion_tokens") or payload.get("max_tokens"),
            })

    # Reconstruct full conversations
    result = []
    for conv_id, turns in sorted(conversations.items()):
        turns.sort(key=lambda t: t["turn_index"])

        # Build the full message history
        full_messages = []
        for turn in turns:
            # The last user message in this turn's payload is the new one
            # (prior messages are history)
            if turn["messages"]:
                # For the first turn, take all messages
                # For subsequent turns, only the last user message is new
                if not full_messages:
                    full_messages.extend(turn["messages"])
                else:
                    # Find new messages not in our history
                    # The payload includes full history, so just use the last message
                    last_msg = turn["messages"][-1]
                    full_messages.append(last_msg)

            # Add assistant response
            if turn["assistant_response"]:
                full_messages.append({
                    "role": "assistant",
                    "content": turn["assistant_response"],
                })

        result.append({
            "conversation_id": conv_id,
            "num_turns": len(turns),
            "messages": full_messages,
        })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full conversations from AIPerf raw export"
    )
    parser.add_argument("raw_jsonl", help="Path to profile_export_raw.jsonl")
    parser.add_argument(
        "-o", "--output", default="conversations.json",
        help="Output JSON file (default: conversations.json)",
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Print first N conversations to stdout",
    )
    args = parser.parse_args()

    conversations = extract_conversations(args.raw_jsonl)

    with open(args.output, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"Extracted {len(conversations)} conversations to {args.output}")

    # Summary
    turn_counts = [c["num_turns"] for c in conversations]
    multi = sum(1 for t in turn_counts if t > 1)
    print(f"  Single-turn: {len(turn_counts) - multi}")
    print(f"  Multi-turn:  {multi}")
    print(f"  Max turns:   {max(turn_counts)}")

    if args.preview > 0:
        for conv in conversations[:args.preview]:
            print(f"\n=== {conv['conversation_id']} ({conv['num_turns']} turns) ===")
            for msg in conv["messages"]:
                role = msg["role"]
                content = msg.get("content", "")[:100]
                print(f"  [{role}] {content}...")


if __name__ == "__main__":
    main()
