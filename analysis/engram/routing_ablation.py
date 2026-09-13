"""Does the Engram ablation penalty come from lost features, or from the
routing shift those lost features cause downstream?

Removing Engram changes the residual at layers 1 and 14. Every MoE router
after that sees a different vector and may send the token to different
experts, so the plain ablation measures two things at once: the value that
Engram itself added, and the cost of running the rest of the network through
a different set of experts than it would have chosen. If Engram and the
experts have co-adapted -- experts specialised on the assumption that the
n-gram memory has already resolved certain tokens -- the second term could be
large, and "memory stores facts, experts reason" would be too simple.

The test is teacher-forced on CRUXEval-O: the two-shot prompt followed by the
reference assertion, scored on the answer tokens only. Teacher forcing is
what makes routing pinnable at all -- every arm sees the identical token
sequence, so expert choices recorded at one position in one arm can be
imposed at the same position in another. Six arms, one weight load:

    baseline           Engram on,  routing free            -> record R_base
    ablated            Engram off, routing free            -> record R_abl
    baseline_selfpin   Engram on,  routing forced to R_base   (control: ~= baseline)
    ablated_selfpin    Engram off, routing forced to R_abl    (control: ~= ablated)
    ablated_pinned     Engram off, routing forced to R_base   (the question)
    baseline_routeabl  Engram on,  routing forced to R_abl    (routing shift alone)

If ablated_pinned recovers a large share of the baseline-ablated gap, the
penalty was mostly the routing shift. If baseline_routeabl loses a large
share, the routing shift alone -- with every Engram feature intact -- is
enough to hurt, which is the co-adaptation signature. The self-pin controls
must sit on top of their free counterparts; if they do not, the recorded
top-k does not match what the kernel chose and nothing here is a measurement.

Two metrics on the answer span: bits per token, and whether the greedy
argmax reproduces the reference at every position (a pass@1 under teacher
forcing; stricter than the generation grader, which tolerates quote style).
The recorded routings are also compared directly: per layer, how often the
Engram-on and Engram-off runs agree on the expert set, split by prompt and
answer tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("engram-routing")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engram import gate_probe, route_probe  # noqa: E402
from engram.cruxeval_ablation import build_prompt  # noqa: E402

# arm -> (engram mode, routing mode, routing tag)
ARMS = {
    "baseline":          ("all",  "record", "base"),
    "ablated":           ("none", "record", "abl"),
    "baseline_selfpin":  ("all",  "replay", "base"),
    "ablated_selfpin":   ("none", "replay", "abl"),
    "ablated_pinned":    ("none", "replay", "base"),
    "baseline_routeabl": ("all",  "replay", "abl"),
}

LN2 = math.log(2.0)


def target_text(row: dict) -> str:
    return "assert f(%s) == %s\n[/ANSWER]" % (row["input"], row["output"])


def scored_positions(prompt_ids: list[int], prefix_ids: list[int]) -> int:
    """First position scored: the boundary between prompt and answer.

    Tokenising prefix and prefix+answer separately can disagree at the join,
    so take the longest common prefix rather than trusting len(prefix)."""
    n = 0
    for a, b in zip(prompt_ids, prefix_ids):
        if a != b:
            break
        n += 1
    return n


def score_output(output, cut: int) -> dict:
    """Answer-span NLL (nats), token count, and whether every answer token was
    the greedy argmax (rank 1) from vLLM's prompt logprobs."""
    ids = output.prompt_token_ids
    nll, n, top1, all_top1 = 0.0, 0, 0, True
    for pos in range(cut, len(ids)):
        entry = (output.prompt_logprobs or [None] * len(ids))[pos]
        if entry is None:
            continue
        lp = entry.get(ids[pos])
        if lp is None:
            continue
        nll += -float(getattr(lp, "logprob", lp))
        n += 1
        rank = getattr(lp, "rank", None)
        if rank is not None and int(rank) == 1:
            top1 += 1
        else:
            all_top1 = False
    return {"nll": nll, "tokens": n, "top1": top1, "greedy": bool(all_top1 and n > 0)}


def _paired(base: list[bool], arm: list[bool]) -> dict:
    ob = sum(1 for b, a in zip(base, arm) if b and not a)
    oa = sum(1 for b, a in zip(base, arm) if a and not b)
    n = ob + oa
    chi2 = ((abs(ob - oa) - 1) ** 2 / n) if n else None
    return {"only_baseline": ob, "only_arm": oa, "discordant": n,
            "chi2": round(chi2, 3) if chi2 is not None else None,
            "approx_sigma": round(math.sqrt(chi2), 2) if chi2 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH") or os.environ.get("MODEL"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=(os.environ.get("RESULT_DIR", ".") + "/engram_routing"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            ap.error("unknown arm %r" % a)
    if arms[:2] != ["baseline", "ablated"]:
        ap.error("arms must start with baseline,ablated: they record the routings the others replay")

    meter_dir = "/dev/shm/engram_routing_meter"
    route_dir = "/dev/shm/engram_routing"
    for d in (meter_dir, route_dir):
        os.makedirs(d, exist_ok=True)
    os.environ[gate_probe.METER_DIR_ENV] = meter_dir
    os.environ[route_probe.DIR_ENV] = route_dir
    analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gate_probe.install_in_workers(analysis_dir)
    gate_probe.install_meter()
    route_probe.install()
    route_probe.clear(route_dir)
    route_probe.set_mode(route_dir, "free")
    gate_probe.set_mode(meter_dir, "all")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    logger.info("cruxeval-O teacher-forced: %d items", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prefixes = [build_prompt(r["code"], r["input"]) for r in rows]
    prompts = [p + target_text(r) for p, r in zip(prefixes, rows)]
    prefix_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prefixes]
    longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    logger.info("longest prompt+answer: %d tokens", longest)
    if longest + 8 > args.max_model_len:
        logger.error("max_model_len %d too small for %d tokens", args.max_model_len, longest)
        return 2

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        # One sequence per forward: the routing key is the whole token
        # sequence and the layer index is a call counter within the forward.
        max_num_seqs=1,
        max_num_batched_tokens=max(4096, args.max_model_len),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    flush_sampling = SamplingParams(max_tokens=1, temperature=0.0)

    flushes = [0]

    def flush() -> dict:
        # A distinct prompt each time: the worker de-duplicates flushes by
        # sequence key, so an identical prompt would flush only once.
        flushes[0] += 1
        route_probe.set_mode(route_dir, "flush")
        llm.generate(["flush the routing probe, pass %d." % flushes[0]], flush_sampling)
        route_probe.set_mode(route_dir, "free")
        return route_probe.read_stats(route_dir)

    def arm_is_sound(arm: str, rstats: dict, expected_calls: int | None) -> str | None:
        """Why this arm is not a measurement, or None. Checked as soon as the
        arm finishes so a broken probe costs minutes, not the whole node."""
        engram_mode, routing_mode, _ = ARMS[arm]
        if rstats["ranks"] != args.tp:
            return "only %d of %d ranks reported routing stats" % (rstats["ranks"], args.tp)
        if rstats["miss"]:
            return "a worker flagged a routing miss"
        calls = rstats["record_calls" if routing_mode == "record" else "replay_calls"]
        if not calls or min(calls) <= 0:
            return "%s produced no %s calls" % (arm, routing_mode)
        if len(set(calls)) != 1:
            return "ranks disagree on %s calls: %s" % (routing_mode, calls)
        if expected_calls is not None and calls[0] != expected_calls:
            return "%s calls %d != %d recorded by baseline" % (routing_mode, calls[0], expected_calls)
        # Rows where the router kernel departed from the pin are corrected in
        # place by the probe; they only invalidate the arm if they are common
        # enough that the corrected weights, not the kernel's, carry the result.
        fallback = max(rstats.get("pin_violations", [0]) or [0])
        if routing_mode == "replay" and calls and fallback > 0.01 * calls[0] * 100:
            return "pin fallback on %d rows, too many to trust the kernel path" % fallback
        return None

    per_arm: dict[str, dict] = {}
    keys: list[str] = []
    cuts: list[int] = []
    expected_calls: int | None = None
    aborted: str | None = None
    for arm in arms:
        engram_mode, routing_mode, tag = ARMS[arm]
        gate_probe.set_mode(meter_dir, engram_mode)
        gate_probe.clear_meter(meter_dir)
        route_probe.set_mode(route_dir, routing_mode, tag)
        miss_path = os.path.join(route_dir, route_probe.MISS)
        if os.path.exists(miss_path):
            os.unlink(miss_path)
        outputs = llm.generate(prompts, sampling)
        route_probe.set_mode(route_dir, "free")

        items = []
        for i, (out, pre) in enumerate(zip(outputs, prefix_ids)):
            cut = scored_positions(list(out.prompt_token_ids), pre)
            if arm == arms[0]:
                keys.append(route_probe.seq_key(out.prompt_token_ids))
                cuts.append(cut)
            elif route_probe.seq_key(out.prompt_token_ids) != keys[i]:
                logger.error("%s: item %d tokenised differently from baseline", arm, i)
            s = score_output(out, cut)
            s["id"] = rows[i].get("id")
            items.append(s)
        total_nll = sum(s["nll"] for s in items)
        total_tok = sum(s["tokens"] for s in items)
        total_top1 = sum(s["top1"] for s in items)
        greedy = [s["greedy"] for s in items]
        meter = gate_probe.read_meter(meter_dir)
        rstats = flush()
        per_arm[arm] = {
            "engram": engram_mode, "routing": routing_mode, "routing_tag": tag,
            "bits_per_token": round(total_nll / max(total_tok, 1) / LN2, 5),
            "answer_tokens": total_tok,
            "top1_token_fraction": round(total_top1 / max(total_tok, 1), 5),
            "greedy_reproduces_reference": round(sum(greedy) / len(greedy), 4),
            "greedy_flags": greedy,
            "items": items,
            "engram_meter": meter,
            "routing_stats": rstats,
        }
        logger.info("%s: %.4f bits/tok, top1 %.4f, greedy %.4f, meter mean %.4f max %.4f, routing %s",
                    arm, per_arm[arm]["bits_per_token"], per_arm[arm]["top1_token_fraction"],
                    per_arm[arm]["greedy_reproduces_reference"],
                    meter.get("mean_rel_norm", float("nan")), meter.get("max_rel_norm", float("nan")),
                    json.dumps(rstats))
        problem = arm_is_sound(arm, rstats, expected_calls)
        if problem:
            logger.error("%s is not a measurement: %s -- aborting the remaining arms", arm, problem)
            aborted = "%s: %s" % (arm, problem)
            break
        if arm == "baseline":
            expected_calls = rstats["record_calls"][0]

    # ---- routing agreement between the two recordings
    boundaries = dict(zip(keys, cuts))
    r_base = route_probe.load_routes(route_dir, "base")
    r_abl = route_probe.load_routes(route_dir, "abl")
    agree = route_probe.agreement(r_base, r_abl, boundaries)
    if agree:
        # Summaries a reader can use without the full table.
        ans = [v["answer"]["exact_set"] for v in agree.values()]
        prm = [v["prompt"]["exact_set"] for v in agree.values()]
        logger.info("routing agreement (exact top-%s set), answer span: min %.3f median %.3f max %.3f",
                    "k", min(ans), sorted(ans)[len(ans) // 2], max(ans))
        logger.info("routing agreement, prompt span: min %.3f median %.3f max %.3f",
                    min(prm), sorted(prm)[len(prm) // 2], max(prm))

    # ---- deltas and verdict
    base = per_arm["baseline"]
    abl = per_arm.get("ablated", base)
    gap_bits = abl["bits_per_token"] - base["bits_per_token"]
    gap_greedy = base["greedy_reproduces_reference"] - abl["greedy_reproduces_reference"]
    comparison = {}
    for arm, r in per_arm.items():
        d_bits = r["bits_per_token"] - base["bits_per_token"]
        d_greedy = r["greedy_reproduces_reference"] - base["greedy_reproduces_reference"]
        comparison[arm] = {
            "delta_bits_vs_baseline": round(d_bits, 5),
            "delta_greedy_vs_baseline": round(d_greedy, 4),
            "share_of_ablation_gap_bits": round(d_bits / gap_bits, 4) if gap_bits else None,
            "share_of_ablation_gap_greedy": round(d_greedy / -gap_greedy, 4) if gap_greedy else None,
            "paired_greedy_vs_baseline": _paired(base["greedy_flags"], r["greedy_flags"]),
        }
        if arm.endswith("_selfpin"):
            ref = per_arm["baseline" if arm.startswith("baseline") else "ablated"]
            comparison[arm]["selfpin_delta_bits_vs_free"] = round(r["bits_per_token"] - ref["bits_per_token"], 5)
            comparison[arm]["selfpin_greedy_flips_vs_free"] = sum(
                1 for a, b in zip(r["greedy_flags"], ref["greedy_flags"]) if a != b)

    def ok_selfpin(arm: str) -> bool:
        c = comparison.get(arm)
        if not c:
            return True  # not run
        # Same tokens, same experts, same weights: only kernel-level noise is
        # allowed between a self-pinned arm and its free counterpart.
        return abs(c["selfpin_delta_bits_vs_free"]) < 0.002 and c["selfpin_greedy_flips_vs_free"] <= 2

    replay_ok = all(
        not r["routing_stats"]["miss"] and sum(r["routing_stats"]["replay_calls"]) > 0
        for a, r in per_arm.items() if ARMS[a][1] == "replay"
    )
    record_ok = all(
        sum(r["routing_stats"]["record_calls"]) > 0
        for a, r in per_arm.items() if ARMS[a][1] == "record"
    )
    engram_ok = gate_probe.ablation_verdict(base["engram_meter"], abl["engram_meter"])
    # Every arm must be in the Engram state its name claims, not just the two
    # recording arms: a replay arm with the wrong gate would still score.
    engram_state_ok = all(
        (r["engram_meter"].get("max_rel_norm", 1.0) <= gate_probe.ABLATION_FLOOR)
        if ARMS[a][0] == "none" else (r["engram_meter"].get("mean_rel_norm", 0.0) > 0.0)
        for a, r in per_arm.items()
    )
    verdict = {
        "aborted": aborted,
        "arms_completed": list(per_arm),
        "engram_ablation": engram_ok,
        "engram_state_matches_every_arm": engram_state_ok,
        "routing_recorded": record_ok,
        "routing_replayed_without_miss": replay_ok,
        "selfpin_baseline_matches": ok_selfpin("baseline_selfpin"),
        "selfpin_ablated_matches": ok_selfpin("ablated_selfpin"),
        "routing_path": sorted({p for r in per_arm.values() for p in r["routing_stats"].get("path", [])}),
        "moe_layers_seen": max(r["routing_stats"].get("layers", 0) for r in per_arm.values()),
    }
    verdict["ok"] = bool(aborted is None and set(per_arm) == set(arms)
                         and engram_ok.get("ok") and engram_state_ok and record_ok and replay_ok
                         and verdict["selfpin_baseline_matches"] and verdict["selfpin_ablated_matches"])

    report = {
        "task": "cruxeval-O teacher-forced answer likelihood, routing pinned across Engram arms",
        "model": args.model,
        "items": len(rows),
        "arms": {a: {k: v for k, v in r.items() if k not in ("items", "greedy_flags")} for a, r in per_arm.items()},
        "ablation_gap": {"bits_per_token": round(gap_bits, 5), "greedy": round(gap_greedy, 4)},
        "comparison": comparison,
        "routing_agreement_base_vs_abl": agree,
        "verdict": verdict,
    }
    with open(os.path.join(args.out, "routing_ablation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(args.out, "routing_items.json"), "w") as fh:
        json.dump({a: r["items"] for a, r in per_arm.items()}, fh)
    print("===ENGRAM_ROUTING_JSON_BEGIN===")
    print(json.dumps({k: v for k, v in report.items() if k != "routing_agreement_base_vs_abl"}))
    print("===ENGRAM_ROUTING_JSON_END===")
    print("===ENGRAM_ROUTING_AGREEMENT_BEGIN===")
    print(json.dumps(agree))
    print("===ENGRAM_ROUTING_AGREEMENT_END===")
    logger.info("VERDICT %s", json.dumps(verdict))
    return 0 if verdict["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
