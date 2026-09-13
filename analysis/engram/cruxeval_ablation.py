"""CRUXEval output prediction, with and without Engram, and nothing executed.

CRUXEval-O shows the model a small Python function and a concrete input and
asks what the call returns. The usual harness settles that by running the
code; here nothing is executed -- not the reference snippet, not the model's
answer. Grading is a value comparison between two *parsed literals*
(`ast.literal_eval`), which is why the task can be scored while the model is
denied any execution tool: the model has to simulate the program in its head,
and we only have to decide whether two Python values are equal.

That makes it the complement of the likelihood work. NLL says Engram is worth
0.09-2.59 bits/token on corpora the model has plainly memorised; gsm8k says it
is worth nothing on multi-step arithmetic. CRUXEval-O is code -- the domain
every one of the ten strongest 4-grams came from -- but it is code the model
has to *execute mentally* rather than recall. If Engram is a memorisation
cache, removing it should cost little here despite the domain match.

Both arms run back to back in one process against one weight load, with only
the Engram mask changing, so the delta carries no node, image or allocation
difference. Accuracy deltas at this sample size are noisy, so the report also
carries the paired McNemar counts (how many items flipped each way), which is
the statistic that actually applies to a paired binary outcome.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("engram-cruxeval")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engram import gate_probe  # noqa: E402

# Unlike the likelihood run, this task generates, so prefill and decode are
# the real engine phases rather than a positional split: prefill_only leaves
# Engram on while the prompt is read and shuts it for every generated token,
# decode_only does the reverse.
ARMS = {
    "baseline": "all",
    "ablated": "none",
    "prefill_only": "prefill",
    "decode_only": "decode",
}

# The CRUXEval-O direct prompt, two-shot, from the paper's released harness.
INSTRUCTION = (
    "You are given a Python function and an assertion containing an input to "
    "the function. Complete the assertion with a literal (no unsimplified "
    "expressions, no function calls) containing the output when executing the "
    "provided code on the given input, even if the function is incorrect or "
    "incomplete. Do NOT output any extra information. Provide the full "
    "assertion with the correct output in [ANSWER] and [/ANSWER] tags, "
    "following the examples."
)

SHOTS = """
[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]
"""

_ANSWER = re.compile(r"\[ANSWER\](.*?)(?:\[/ANSWER\]|$)", re.S)


def build_prompt(code: str, value: str, open_answer: bool = True) -> str:
    """The direct prompt. `open_answer` leaves the [ANSWER] tag open so a raw
    completion has only the literal left to write; a chat turn gets the closed
    question and writes the whole block itself after its reasoning."""
    text = (
        f"{INSTRUCTION}\n{SHOTS}\n[PYTHON]\n{code}\n"
        f"assert f({value}) == ??\n[/PYTHON]\n"
    )
    return text + "[ANSWER]\n" if open_answer else text


THINK_END = "</think>"


def strip_thinking(text: str) -> tuple[str, bool]:
    """Visible answer of a thinking-mode generation, and whether the model
    actually closed its reasoning block.

    The V4.1 chat encoding ends the prompt with `<think>`, so the completion
    opens inside the reasoning. Everything up to the last `</think>` is
    working, not an answer; grading it would score the model's discarded
    candidates. A generation with no `</think>` ran out of budget while still
    thinking: it is graded wrong and counted separately, because a budget
    failure is not an Engram effect."""
    idx = text.rfind(THINK_END)
    if idx < 0:
        return "", False
    return text[idx + len(THINK_END):], True


def load_v41_encoder(model_path: str | None):
    """`encode_messages` for the DeepSeek-V4.1 prompt format.

    The checkpoint ships no `chat_template`; the format lives in code. vLLM's
    `deepseek_v41` tokenizer mode -- which the served gsm8k arms used -- wraps
    the same reference module, so prefer vLLM's copy and fall back to the one
    in the model directory (`encoding/encoding.py`)."""
    try:
        from vllm.tokenizers.deepseek_v41_encoding import encode_messages
        return encode_messages, "vllm.tokenizers.deepseek_v41_encoding"
    except Exception as exc:  # pragma: no cover - depends on the image
        logger.warning("vllm deepseek_v41 encoder unavailable: %r", exc)
    import importlib.util
    candidates = []
    if model_path and os.path.isdir(model_path):
        candidates.append(os.path.join(model_path, "encoding", "encoding.py"))
    try:
        from huggingface_hub import hf_hub_download
        candidates.append(hf_hub_download(
            "deepseek-ai/DeepSeek-V4.1-Flash", "encoding/encoding.py"))
    except Exception as exc:
        logger.warning("hf_hub_download of encoding.py failed: %r", exc)
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.encode_messages, path
    raise RuntimeError("no DeepSeek-V4.1 encoder found; cannot build chat prompts")


def extract(text: str) -> str | None:
    """The right-hand side of the asserted equality, as raw source text.

    Scanned from the END. This checkpoint is a reasoning model, so with a chat
    template the visible content carries its chain of thought before the
    answer, and that chain routinely contains candidate equalities it then
    rejects. Taking the first `==` would grade the model's discarded working;
    the last one is its conclusion.
    """
    blocks = _ANSWER.findall(text)
    body = (blocks[-1] if blocks else text).strip()
    for line in reversed(body.splitlines()):
        line = line.strip().rstrip(";").rstrip("`").strip()
        if not line or line in ("[/ANSWER]", "[ANSWER]"):
            continue
        if "==" in line:
            return line.split("==", 1)[1].strip()
        if line.startswith("assert "):
            continue
        # A bare literal is only trusted when the prompt's own [ANSWER] tag
        # framed it; loose prose would otherwise be read as an answer.
        if blocks:
            return line
    return None


def equivalent(predicted: str | None, reference: str) -> bool:
    """Compare two Python literals by value, executing neither.

    `ast.literal_eval` builds the value from the parse tree; it never calls a
    function and never runs model-authored code. Anything it refuses to parse
    -- a call, a comprehension, a name -- falls back to a whitespace-normalised
    string comparison, so an unsimplified expression is simply wrong rather
    than evaluated.
    """
    if predicted is None:
        return False
    try:
        return ast.literal_eval(predicted) == ast.literal_eval(reference)
    except (ValueError, SyntaxError, MemoryError, TypeError, RecursionError):
        return " ".join(predicted.split()) == " ".join(reference.split())


def answer_kind(reference: str) -> str:
    """Coarse type of the expected answer, for splitting the delta.

    CRUXEval-O mixes two very different demands. Some items want a value the
    model has to compute (an int, a bool); many want a string or container
    that is largely a rearrangement of characters already present in the
    prompt. An n-gram memory is exactly the mechanism that would help with the
    second and not the first, so the headline delta is not interpretable
    without this split.
    """
    try:
        value = ast.literal_eval(reference)
    except (ValueError, SyntaxError, MemoryError, TypeError, RecursionError):
        return "unparsed"
    if isinstance(value, bool) or value is None:
        return "bool/None"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, (list, tuple, set, dict)):
        return "container"
    return "other"


def _mcnemar(only_baseline: int, only_ablated: int) -> dict:
    """Paired-binary significance for the flips, which is what we sampled."""
    n = only_baseline + only_ablated
    if n == 0:
        return {"discordant": 0, "chi2": None, "note": "no item changed"}
    # Continuity-corrected McNemar; with small n report it as indicative only.
    chi2 = (abs(only_baseline - only_ablated) - 1) ** 2 / n
    return {
        "discordant": n,
        "chi2": round(chi2, 3),
        "approx_sigma": round(math.sqrt(chi2), 2),
        "note": "chi-square approximation is weak below ~25 discordant pairs",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH") or os.environ.get("MODEL"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole 800")
    ap.add_argument("--max-model-len", type=int, default=8192)
    # A reasoning checkpoint needs room to think before the [ANSWER] block;
    # too small a budget truncates the answer rather than the reasoning.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default=(os.environ.get("RESULT_DIR", ".") + "/engram_cruxeval"))
    # `auto` is the historical behaviour: a chat template if the tokenizer has
    # one, raw completion otherwise (this checkpoint has none, so `auto` ran
    # raw). `chat` renders the turn the way vLLM's `deepseek_v41` tokenizer
    # mode does for /v1/chat/completions -- thinking on, reasoning effort
    # `high` -- which is exactly the regime the gsm8k arms were served under.
    ap.add_argument("--prompt-style", choices=["auto", "raw", "chat"], default="auto")
    ap.add_argument("--reasoning-effort", default="high",
                    help="chat only: low/high/xhigh/max or 1-100; vLLM's default is high")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma list from %s" % ",".join(ARMS))
    args = ap.parse_args()
    arms = {a: ARMS[a] for a in args.arms.split(",") if a}
    if "baseline" not in arms or "ablated" not in arms:
        ap.error("--arms must include baseline and ablated")
    os.makedirs(args.out, exist_ok=True)

    meter_dir = "/dev/shm/engram_cruxeval"
    os.makedirs(meter_dir, exist_ok=True)
    os.environ[gate_probe.METER_DIR_ENV] = meter_dir
    analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gate_probe.install_in_workers(analysis_dir)
    gate_probe.install_meter()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    logger.info("cruxeval-O: %d items", len(rows))

    # Prefer the chat template, but do not require it: the local MODEL_PATH
    # copy does not always carry tokenizer_config's chat_template, and the
    # two-shot CRUXEval format is a completion-style prompt to begin with, so
    # raw continuation is a correct fallback rather than a degraded one.
    tokenizer, template_source, encode_messages = None, None, None
    if args.prompt_style == "chat":
        encode_messages, template_source = load_v41_encoder(args.model)
        logger.info("chat prompts via %s: thinking on, reasoning_effort=%s",
                    template_source, args.reasoning_effort)
    elif args.prompt_style == "auto":
        for candidate in (args.model, os.environ.get("MODEL")):
            if not candidate:
                continue
            try:
                tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            except Exception as exc:
                logger.warning("tokenizer %s: %r", candidate, exc)
                continue
            if getattr(tok, "chat_template", None):
                tokenizer, template_source = tok, candidate
                break
            tokenizer = tokenizer or tok
        if template_source:
            logger.info("using the chat template from %s", template_source)
        else:
            logger.warning("no chat_template available; using raw completion prompts")
    else:
        logger.info("raw completion prompts by request")
    thinking = encode_messages is not None
    effort = args.reasoning_effort
    if isinstance(effort, str) and effort.isdigit():
        effort = int(effort)

    prompts = []
    for row in rows:
        text = build_prompt(row["code"], row["input"], open_answer=not thinking)
        if encode_messages is not None:
            text = encode_messages(
                [{"role": "user", "content": text}],
                thinking_mode="thinking", reasoning_effort=effort,
            )
        elif template_source:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)
    logger.info("prompt[0] tail: %r", prompts[0][-160:])

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        # Required, not a precaution. The meter wraps Engram.forward with host
        # work -- a second forward under a mask, then a numpy write -- and
        # doing that inside a CUDA graph capture invalidates the stream
        # (cudaErrorStreamCaptureInvalidated killed the first attempt). The
        # likelihood run is eager for the same reason.
        enforce_eager=True,
        # The two arms score the same prompts; a shared prefix cache would let
        # the ablated arm reuse KV computed while Engram was still on.
        enable_prefix_caching=False,
    )
    # Stop at the closing tag so a completion-style prompt does not run on to
    # invent a third exemplar; the tag is stripped before extraction anyway.
    # A thinking-mode turn must not stop on the tags: the model writes candidate
    # [ANSWER] blocks inside its reasoning and rejects them. It ends its turn
    # with EOS, so let it.
    sampling = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0,
        stop=None if thinking else ["[/ANSWER]", "[PYTHON]"],
    )

    correct: dict[str, list[bool]] = {}
    samples: dict[str, list[dict]] = {}
    records: dict[str, list[dict]] = {}
    coverage: dict[str, dict] = {}
    miss_path = os.path.join(meter_dir, gate_probe.PHASE_MISS)
    phase_miss = {}
    for arm, mode in arms.items():
        if os.path.exists(miss_path):
            os.unlink(miss_path)
        gate_probe.set_mode(meter_dir, mode)
        gate_probe.clear_meter(meter_dir)
        outputs = llm.generate(prompts, sampling)
        flags, shown, recs = [], [], []
        empty, unfinished, gen_tokens = 0, 0, 0
        for row, out in zip(rows, outputs):
            text = out.outputs[0].text if out.outputs else ""
            gen_tokens += len(out.outputs[0].token_ids) if out.outputs else 0
            if not text.strip():
                empty += 1
            if thinking:
                visible, closed = strip_thinking(text)
                if not closed:
                    unfinished += 1
            else:
                visible = text
            predicted = extract(visible)
            ok = equivalent(predicted, row["output"])
            flags.append(ok)
            rec = {"id": row.get("id"), "reference": row["output"],
                   "predicted": predicted, "correct": ok}
            if thinking:
                rec["finished_thinking"] = closed
            recs.append(rec)
            if len(shown) < 5:
                shown.append(rec)
        correct[arm] = flags
        samples[arm] = shown
        records[arm] = recs
        stats = gate_probe.read_meter(meter_dir)
        phase_miss[arm] = os.path.exists(miss_path)
        coverage[arm] = dict(stats, empty_generations=empty,
                             phase_mask_unavailable=phase_miss[arm],
                             mean_generated_tokens=round(gen_tokens / max(len(rows), 1), 1))
        if thinking:
            coverage[arm]["unfinished_thinking"] = unfinished
            if unfinished:
                logger.warning("%s: %d/%d generations hit max_tokens=%d before "
                               "closing </think>; graded wrong, not an Engram effect",
                               arm, unfinished, len(rows), args.max_tokens)
        if phase_miss[arm]:
            logger.error("%s: the engine phase could not be determined on at "
                         "least one forward call; this arm is not a "
                         "measurement", arm)
        logger.info("%s: pass@1=%.4f empty=%d unfinished=%d gen_tok/item=%.0f meter=%s",
                    arm, sum(flags) / max(len(flags), 1), empty, unfinished,
                    gen_tokens / max(len(rows), 1), json.dumps(stats))
        if empty > len(rows) // 10:
            logger.error("%s: %d/%d generations were empty -- the score is not "
                         "a measurement of the model", arm, empty, len(rows))

    # Split the delta by what the answer actually demands.
    kinds = [answer_kind(row["output"]) for row in rows]
    by_kind = {}
    for kind in sorted(set(kinds)):
        sel = [i for i, k in enumerate(kinds) if k == kind]
        b = [correct["baseline"][i] for i in sel]
        by_kind[kind] = {"items": len(sel)}
        for arm, flags in correct.items():
            a = [flags[i] for i in sel]
            by_kind[kind][arm] = {
                "pass@1": round(sum(a) / len(sel), 4),
                "delta": round((sum(a) - sum(b)) / len(sel), 4),
                "only_baseline_correct": sum(1 for x, y in zip(b, a) if x and not y),
                "only_arm_correct": sum(1 for x, y in zip(b, a) if y and not x),
            }
        logger.info("kind %s: %s", kind, json.dumps(by_kind[kind]))

    base, abl = correct["baseline"], correct["ablated"]
    only_base = sum(1 for b, a in zip(base, abl) if b and not a)
    only_abl = sum(1 for b, a in zip(base, abl) if a and not b)
    report = {
        "task": "cruxeval-O (output prediction, no execution)",
        "model": args.model,
        "items": len(rows),
        "prompt_style": (
            "chat-thinking:deepseek_v41 effort=%s via %s" % (effort, template_source)
            if thinking else
            ("chat-template:%s" % template_source) if template_source
            else "raw-completion"),
        "arms": list(arms),
        "max_tokens": args.max_tokens,
        "pass@1": {arm: round(sum(f) / len(f), 4) for arm, f in correct.items()},
        "delta_pass@1": round(sum(abl) / len(abl) - sum(base) / len(base), 4),
        "arms_vs_baseline": {
            arm: {
                "pass@1": round(sum(f) / len(f), 4),
                "delta": round((sum(f) - sum(base)) / len(base), 4),
                "only_baseline_correct": sum(1 for b, a in zip(base, f) if b and not a),
                "only_arm_correct": sum(1 for b, a in zip(base, f) if a and not b),
                "mcnemar": _mcnemar(
                    sum(1 for b, a in zip(base, f) if b and not a),
                    sum(1 for b, a in zip(base, f) if a and not b),
                ),
            }
            for arm, f in correct.items() if arm != "baseline"
        },
        "paired": {
            "both_correct": sum(1 for b, a in zip(base, abl) if b and a),
            "both_wrong": sum(1 for b, a in zip(base, abl) if not b and not a),
            "only_baseline_correct": only_base,
            "only_ablated_correct": only_abl,
            "mcnemar": _mcnemar(only_base, only_abl),
        },
        "by_answer_kind": by_kind,
        "coverage": coverage,
        "samples": samples,
    }
    verdict = gate_probe.ablation_verdict(coverage.get("baseline"), coverage.get("ablated"))
    verdict["phase_mask_unavailable"] = {a: bool(v) for a, v in phase_miss.items()}
    verdict["ok"] = bool(verdict["ok"] and not any(phase_miss.values()))
    report["ablation_verdict"] = verdict

    with open(os.path.join(args.out, "cruxeval_ablation.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    # Per-item flags go to their own file: they are what makes any later
    # slice of this result possible without paying for another run.
    with open(os.path.join(args.out, "cruxeval_items.json"), "w") as handle:
        json.dump(records, handle, indent=2)
    print("===ENGRAM_CRUXEVAL_KINDS_BEGIN===")
    print(json.dumps(by_kind))
    print("===ENGRAM_CRUXEVAL_KINDS_END===")
    print("===ENGRAM_CRUXEVAL_JSON_BEGIN===")
    print(json.dumps(report))
    print("===ENGRAM_CRUXEVAL_JSON_END===")
    logger.info("ABLATION VERDICT %s", json.dumps(verdict))
    if not verdict["ok"]:
        logger.error("ablation not verified; the delta above is not interpretable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
