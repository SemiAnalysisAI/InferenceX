#!/usr/bin/env python3
"""
Convert Claude Code trace batches (runs_*/batch_*.json, one stream-json record
per line) into a STANDARDIZED multi-turn replay dataset for InferenceX-style
pareto benchmarking.

Why this exists
---------------
A Claude Code session is a multi-turn agentic conversation: the model emits an
assistant turn (thinking + text + tool calls), the harness runs the tools and
feeds the results back as the next user turn, and so on. Each turn's prompt is
the *entire growing conversation so far* -> long shared prefixes -> heavy KV
prefix-cache reuse. That is exactly the workload we want to replay against an
inference server.

Replay model: "pre-canned assistant replay"
--------------------------------------------
We reconstruct the full conversation ONCE (flat OpenAI-style `messages`), then
mark each assistant turn. To replay turn k we send `messages[:prefix_len_k]` to
the server, let it generate up to `max_tokens_k` (= the recorded output length
of that turn, so decode load matches the trace) and DISCARD the output -- the
recorded assistant/user turns already sit in `messages`, so the next turn's
prefix is byte-for-byte deterministic across runs and across concurrency. This
keeps the prefix-cache behaviour identical regardless of what the server says.

Output: one JSON object per line (`.replay.jsonl`):
  {
    "session_id":   str,
    "entry_point":  str,
    "source":       "GPUMODE/KernelBook",
    "kernelbook_uuid": int | None,
    "messages":     [ {"role": "system|user|assistant", "content": str}, ... ],
    "turns": [ { "prefix_len":  int,    # request = messages[:prefix_len]
                 "max_tokens":  int,    # recorded output_tokens for this turn
                 "delay_before_s": float,  # capped idle/think gap before the turn
                 "rec_input_tokens":  int,  # Claude's recorded prompt tokens
                 "rec_output_tokens": int,  # Claude's recorded completion tokens
                 "rec_cache_read_tokens": int } ... ],
    "recorded": { "num_turns": int, "correct": bool, "speedup": float|None }
  }

`messages` is stored flat once (O(n)); the replayer slices it per turn, so the
file stays small even for 300-session batches.

Usage:
  python make_replay_dataset.py runs_8t/batch_1.json -o replay/batch_1.replay.jsonl
  python make_replay_dataset.py runs_8t/batch_1.json runs_full/batch_0.jsonl \
         -o replay/all.replay.jsonl --system-prompt system_prompt.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

# Mirrors solve.py:build_prompt closely enough to give a realistic first user
# turn (the exact text Claude received references the reference.py path; the
# file body itself enters the conversation as the first Read tool result, which
# IS preserved in the trace). Token counts are representative, not identical to
# Claude's (different tokenizer) -- expected for cross-model replay.
def reconstruct_task(entry: str, uuid: str, min_evals: int = 8) -> str:
    ref_path = f"{PROJ}/runs/work/{uuid}/reference.py"
    return (
        f"You are an expert GPU kernel engineer specializing in Triton.\n\n"
        f"The file `{ref_path}` contains a PyTorch module named `{entry}` (a subclass of\n"
        f"torch.nn.Module), plus `get_init_inputs()` and `get_inputs()` helpers. Read it first.\n\n"
        f"Write a Triton implementation: a class named `{entry}New` that is a drop-in\n"
        f"replacement for `{entry}` (identical __init__/forward signatures and parameters,\n"
        f"numerically equivalent outputs). Implement the forward computation with Triton\n"
        f"kernels (@triton.jit) wherever sensible; trivial glue may stay in PyTorch.\n\n"
        f"To TEST it, pipe your COMPLETE kernel source on stdin to the judge; it prints\n"
        f"correctness + speedup. Iterate: if not correct, fix and pipe again; if correct,\n"
        f"make it FASTER and pipe again. You must run at least {min_evals} genuinely-different\n"
        f"optimization iterations before stopping."
    )


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _result_text(tool_result) -> str:
    c = tool_result.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return "" if c is None else str(c)


def _render_tool_use(block) -> str:
    """Render an assistant tool call as plain text so the replayed assistant
    turn is model-agnostic (no per-model tool schema needed)."""
    name = block.get("name")
    inp = block.get("input") or {}
    if name == "Bash":
        return f"<tool:Bash>\n{inp.get('command', '')}\n</tool>"
    if name in ("Write", "Edit", "Read"):
        return f"<tool:{name}>\n{json.dumps(inp)[:8000]}\n</tool>"
    return f"<tool:{name}>\n{json.dumps(inp)[:4000]}\n</tool>"


def _prompt_tokens(usage: dict) -> int:
    if not usage:
        return 0
    return (int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("cache_read_input_tokens", 0) or 0)
            + int(usage.get("cache_creation_input_tokens", 0) or 0))


def build_session(rec: dict, system_prompt: str, min_evals: int,
                  idle_cap_s: float) -> dict | None:
    trace = rec.get("trace") or []
    entry = rec.get("entry_point") or "Model"
    uuid = str(rec.get("uuid"))

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": reconstruct_task(entry, uuid, min_evals)})

    turns: list[dict] = []
    prev_ts = None  # wall-clock of the previous tool-result, for think-time gaps

    # Claude Code streams MULTIPLE assistant events per logical turn (one each for
    # thinking / text / every tool_use), each carrying its own small usage. A turn
    # is the run of consecutive assistant events up to the next user (tool_result)
    # event, so we buffer them and flush as a single assistant message: content =
    # concat of all text+tool_use, output_tokens = SUM over the run, prompt usage =
    # the FIRST event's usage (the prompt the turn generated from).
    buf_parts: list[str] = []
    buf_out = 0
    buf_first_usage: dict | None = None

    def flush_turn():
        nonlocal buf_parts, buf_out, buf_first_usage
        content = "\n".join(p for p in buf_parts if p).strip()
        if content:
            usage = buf_first_usage or {}
            turns.append({
                "prefix_len": len(messages),         # request = messages[:prefix_len]
                "max_tokens": 1,                     # set by OSL distribution below
                "delay_before_s": 0.0,               # filled from the next user ts
                "rec_input_tokens": _prompt_tokens(usage),
                "rec_output_tokens": 0,              # set by OSL distribution below
                "rec_cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                "_chars": len(content),              # weight for OSL distribution
            })
            messages.append({"role": "assistant", "content": content})
        buf_parts, buf_out, buf_first_usage = [], 0, None

    for ev in trace:
        t = ev.get("type")
        if t == "assistant":
            msg = ev.get("message", {}) or {}
            usage = msg.get("usage", {}) or {}
            if buf_first_usage is None:
                buf_first_usage = usage
            buf_out += int(usage.get("output_tokens", 0) or 0)
            for c in msg.get("content", []) or []:
                ct = c.get("type")
                if ct == "text":
                    buf_parts.append(c.get("text", ""))
                elif ct == "tool_use":
                    buf_parts.append(_render_tool_use(c))
                # thinking dropped (ephemeral, not resent in history) but its
                # output_tokens stay counted above so OSL reflects real decode load
        elif t == "user":
            flush_turn()  # the tool result closes the assistant turn before it
            msg = ev.get("message", {}) or {}
            ts = _parse_ts(ev.get("timestamp"))
            if turns and prev_ts is not None and ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if gap > 0:
                    turns[-1]["delay_before_s"] = round(min(gap, idle_cap_s), 3)
            if ts is not None:
                prev_ts = ts
            chunks = []
            for c in msg.get("content", []) or []:
                if c.get("type") == "tool_result":
                    chunks.append(_result_text(c))
            text = "\n".join(x for x in chunks if x).strip()
            if text:
                messages.append({"role": "user", "content": text})
    flush_turn()  # trailing assistant turn with no following tool result

    if not turns:
        return None

    # --- OSL distribution -----------------------------------------------------
    # Claude's per-event usage.output_tokens excludes hidden thinking tokens, so
    # it massively undercounts a reasoning model's decode work. The session total
    # in the trailing `result` event IS authoritative. Distribute it across turns
    # weighted by each turn's visible content length, so per-turn OSL is realistic
    # and the per-session sum matches the trace exactly. With ignore_eos the
    # server then generates this many tokens regardless of model -> faithful
    # decode load. Fall back to per-turn visible-text estimate if no total.
    total_out = 0
    for e in reversed(trace):
        if e.get("type") == "result":
            total_out = int((e.get("usage") or {}).get("output_tokens", 0) or 0)
            break
    weights = [t.pop("_chars") for t in turns]
    wsum = sum(weights) or 1
    if total_out > 0:
        # largest-remainder apportionment so the rounded parts sum to total_out
        raw = [total_out * w / wsum for w in weights]
        floor = [int(x) for x in raw]
        rem = total_out - sum(floor)
        order = sorted(range(len(turns)), key=lambda i: raw[i] - floor[i], reverse=True)
        for i in order[:rem]:
            floor[i] += 1
        alloc = [max(v, 1) for v in floor]
    else:
        # no authoritative total: ~4 chars/token visible-text estimate
        alloc = [max(round(w / 4), 1) for w in weights]
    for t, n in zip(turns, alloc):
        t["max_tokens"] = n
        t["rec_output_tokens"] = n

    ev = rec.get("eval") or {}
    return {
        "session_id": rec.get("session_id") or uuid,
        "entry_point": entry,
        "source": "GPUMODE/KernelBook",
        "kernelbook_uuid": int(uuid) if uuid.isdigit() else None,
        "messages": messages,
        "turns": turns,
        "recorded": {
            "num_turns": len(turns),
            "correct": bool(ev.get("correct")),
            "speedup": ev.get("speedup"),
        },
    }


def load_records(paths) -> list[dict]:
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="batch_*.json(l) trace files")
    ap.add_argument("-o", "--out", required=True, help="output .replay.jsonl path")
    ap.add_argument("--system-prompt", default=str(PROJ / "system_prompt.md"),
                    help="system prompt file to prepend to each session "
                         "(set to '' to omit)")
    ap.add_argument("--min-evals", type=int, default=8,
                    help="value used to reconstruct the first user turn text")
    ap.add_argument("--idle-cap-s", type=float, default=60.0,
                    help="cap inter-turn think-time at this many seconds "
                         "(matches InferenceX --trace-idle-gap-cap-seconds 60)")
    args = ap.parse_args()

    sp = Path(args.system_prompt) if args.system_prompt else None
    system_prompt = sp.read_text() if (sp and sp.exists()) else ""

    recs = load_records(args.inputs)
    sessions = []
    for r in recs:
        s = build_session(r, system_prompt, args.min_evals, args.idle_cap_s)
        if s is not None:
            sessions.append(s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for s in sessions:
            fh.write(json.dumps(s) + "\n")

    # provenance / sanity summary
    n_turns = sum(len(s["turns"]) for s in sessions)
    rec_isl = [t["rec_input_tokens"] for s in sessions for t in s["turns"]]
    rec_osl = [t["rec_output_tokens"] for s in sessions for t in s["turns"]]
    import statistics as st
    print(f"wrote {out}")
    print(f"  sessions: {len(sessions)} / {len(recs)} records")
    print(f"  replay turns: {n_turns}  (mean {n_turns/max(len(sessions),1):.1f}/session)")
    print(f"  system prompt: incorporated (~{len(system_prompt)//4} tokens, "
          f"shared across sessions)" if system_prompt else "  system prompt: NONE")
    if rec_isl:
        print(f"  recorded ISL: median {int(st.median(rec_isl)):,}  max {max(rec_isl):,}")
        print(f"  recorded OSL: median {int(st.median(rec_osl)):,}  max {max(rec_osl):,}")


if __name__ == "__main__":
    main()
