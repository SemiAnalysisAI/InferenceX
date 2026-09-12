"""Measure Engram's contribution as a likelihood delta rather than an accuracy delta.

gsm8k gave 1319 samples and a baseline that moved 0.0045 between identical
runs, which is larger than the effect. Negative log-likelihood gives one
measurement per token instead of one per question, so the same corpus yields
~10^8 samples and the standard error collapses.

The design is paired: for every chunk the same suffix tokens are scored once
per arm -- baseline, fully ablated, Engram restricted to the context prefix,
and Engram restricted to the scored suffix -- back to back in one process,
with only the mask changing. See MODE_NOTE for what the phase split means when
the measurement is a single prompt-scoring forward pass. Prefix caching is
off -- with it on, the second pass would reuse KV computed while Engram was
still contributing, and the measured delta would collapse toward zero for
reasons that have nothing to do with Engram.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import statistics
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("engram-nll")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engram import corpora, gate_probe  # noqa: E402
from engram.scan import _take_chunks  # noqa: E402


# Engram masking mode per arm. See MODE_NOTE for what the phase split means
# when the measurement is a single prompt-scoring forward pass.
ARMS = {
    "baseline": "all",
    "ablated": "none",
    "prefill_only": "prefix",
    "decode_only": "suffix",
}

MODE_NOTE = (
    "Scoring prompt logprobs is one prefill; there is no decode phase to split "
    "on, so 'prefill_only' and 'decode_only' are a POSITIONAL split of the "
    "chunk at --boundary. prefill_only lets Engram act only on the context "
    "prefix (it shapes the KV the suffix attends to, but is shut at the point "
    "of prediction); decode_only shuts it over the prefix and opens it only on "
    "the scored tokens themselves. All four arms are scored on the same suffix "
    "tokens, so the numbers are directly comparable: prefill_only isolates "
    "Engram as a context builder, decode_only as a point-of-prediction lookup."
)


def _token_nll(output) -> list[tuple[int, float]]:
    """(position, negative log-likelihood) from vLLM's prompt logprobs."""
    out = []
    ids = output.prompt_token_ids
    for position, entry in enumerate(output.prompt_logprobs or []):
        if entry is None:  # the first token is unconditioned
            continue
        chosen = entry.get(ids[position])
        if chosen is None:
            continue
        out.append((position, -float(getattr(chosen, "logprob", chosen))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    ap.add_argument("--chunk-tokens", type=int, default=3584)
    ap.add_argument("--chunks-per-domain", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument(
        "--boundary", type=int, default=None,
        help="positional split between the prefix and the scored suffix "
             "(default: half of --chunk-tokens)",
    )
    ap.add_argument(
        "--arms", default=",".join(ARMS),
        help="comma-separated subset of %s" % ",".join(ARMS),
    )
    ap.add_argument("--out", default=os.environ.get("RESULT_DIR", ".") + "/engram_nll")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error("unknown arm(s): %s" % ", ".join(unknown))
    if "baseline" not in arms:
        ap.error("baseline is required -- every delta is measured against it")
    boundary = args.boundary if args.boundary is not None else args.chunk_tokens // 2
    if not 0 < boundary < args.chunk_tokens:
        ap.error("--boundary must be inside (0, --chunk-tokens)")
    logger.info("arms=%s boundary=%d", arms, boundary)
    logger.info("%s", MODE_NOTE)

    os.makedirs(args.out, exist_ok=True)
    meter_dir = "/dev/shm/engram_nll"
    os.makedirs(meter_dir, exist_ok=True)
    os.environ[gate_probe.METER_DIR_ENV] = meter_dir
    analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gate_probe.install_in_workers(analysis_dir)
    gate_probe.install_meter()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        # The positional mask indexes within a single forward call, so a chunk
        # must not be split across chunked-prefill batches.
        max_num_batched_tokens=max(4096, args.chunk_tokens),
        max_num_seqs=1,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
        # Off deliberately: see the module docstring.
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    per_domain = {}
    coverage: dict[str, dict] = {}
    for domain in corpora.all_domains():
        chunks = list(
            _take_chunks(
                corpora.iter_texts(domain), tokenizer, args.chunk_tokens,
                args.chunks_per_domain, 0, 1,
            )
        )
        if not chunks:
            logger.warning("domain %s: no chunks", domain)
            continue

        totals = {x: 0.0 for x in arms}
        counts = {x: 0 for x in arms}
        chunk_means: dict[str, list[float]] = {x: [] for x in arms}
        contribution = {}
        for idx, chunk in enumerate(chunks):
            means = {}
            for arm in arms:
                gate_probe.set_mode(meter_dir, ARMS[arm], boundary)
                # Cleared every chunk, not just the first: the contribution is
                # now checked on every single chunk rather than sampled once.
                gate_probe.clear_meter(meter_dir)
                out = llm.generate({"prompt_token_ids": chunk}, sampling, use_tqdm=False)[0]
                # Score only the suffix, identically in every arm, so the arms
                # differ solely in where Engram was allowed to act.
                nlls = [v for pos, v in _token_nll(out) if pos >= boundary]
                if not nlls:
                    continue
                totals[arm] += sum(nlls)
                counts[arm] += len(nlls)
                means[arm] = sum(nlls) / len(nlls)
                stats = gate_probe.read_meter(meter_dir)
                if idx == 0:
                    contribution[arm] = stats
                # Every chunk is checked, so a single unablated chunk anywhere
                # in the run is caught rather than averaged away.
                if stats:
                    seen = coverage.setdefault(arm, {"chunks": 0, "calls": 0,
                                                     "worst": 0.0, "min": 1e9})
                    seen["chunks"] += 1
                    seen["calls"] += stats["calls"]
                    seen["worst"] = max(seen["worst"], stats["max_rel_norm"])
                    seen["min"] = min(seen["min"], stats["mean_rel_norm"])
            if len(means) == len(arms):
                for arm in arms:
                    chunk_means[arm].append(means[arm])
            if idx % 25 == 0:
                logger.info("domain %s: %d/%d chunks", domain, idx + 1, len(chunks))

        if not all(counts[x] for x in arms) or not chunk_means["baseline"]:
            continue
        base = totals["baseline"] / counts["baseline"]
        entry = {
            "chunks": len(chunk_means["baseline"]),
            "tokens_scored": counts["baseline"],
            "boundary": boundary,
            "nll_baseline": round(base, 6),
            "ppl_baseline": round(math.exp(base), 4),
            "arms": {},
            "contribution_first_chunk": contribution,
        }
        for arm in arms:
            if arm == "baseline":
                continue
            value = totals[arm] / counts[arm]
            # Paired per chunk, then a chunk-level stderr: tokens within a
            # chunk are heavily correlated, so a per-token stderr would
            # overstate the precision.
            deltas = [
                x - y for x, y in zip(chunk_means[arm], chunk_means["baseline"])
            ]
            stderr = (
                statistics.stdev(deltas) / math.sqrt(len(deltas))
                if len(deltas) > 1 else float("nan")
            )
            entry["arms"][arm] = {
                "nll": round(value, 6),
                "delta_nll": round(value - base, 6),
                "delta_bits_per_token": round((value - base) / math.log(2), 6),
                "ppl": round(math.exp(value), 4),
                "ppl_ratio": round(math.exp(value - base), 6),
                "chunk_stderr": round(stderr, 6) if stderr == stderr else None,
                "sigma": (
                    round((value - base) / stderr, 2)
                    if stderr == stderr and stderr > 0 else None
                ),
            }
        per_domain[domain] = entry
        logger.info("%s: %s", domain, json.dumps(per_domain[domain], default=str))

    report = {"per_domain": per_domain, "model": args.model,
              "arms": arms, "boundary": boundary, "mode_note": MODE_NOTE}
    with open(os.path.join(args.out, "nll_ablation.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("===ENGRAM_NLL_JSON_BEGIN===")
    print(json.dumps(report))
    print("===ENGRAM_NLL_JSON_END===")

    if not per_domain:
        logger.error("no domain produced a measurement")
        return 1
    # A zero contribution in the first chunk means the ablation never engaged,
    # and every delta below would be noise.
    first = next(iter(per_domain.values()))["contribution_first_chunk"]
    verdict = gate_probe.ablation_verdict(
        first.get("baseline"), first.get("ablated", first.get("baseline"))
    )
    # Whole-run coverage, not a first-chunk sample.
    base_cov = coverage.get("baseline", {})
    abl_cov = coverage.get("ablated", {})
    verdict["per_arm_coverage"] = {x: coverage.get(x, {}) for x in arms}
    verdict["coverage"] = {"baseline": base_cov, "ablated": abl_cov}
    verdict["every_ablated_chunk_zero"] = (
        bool(abl_cov and abl_cov.get("worst", 1.0) <= gate_probe.ABLATION_FLOOR)
        if "ablated" in arms else None
    )
    verdict["every_baseline_chunk_nonzero"] = bool(
        base_cov and base_cov.get("min", 0.0) > gate_probe.ABLATION_FLOOR
    )
    verdict["ok"] = bool(
        verdict["every_baseline_chunk_nonzero"]
        and (verdict["every_ablated_chunk_zero"] or "ablated" not in arms)
    )
    report["ablation_verdict"] = verdict
    logger.info("ABLATION VERDICT %s", json.dumps(verdict))
    with open(os.path.join(args.out, "nll_ablation.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    if not verdict["ok"]:
        logger.error("ablation not verified; the deltas above are not interpretable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
