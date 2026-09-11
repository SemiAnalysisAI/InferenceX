"""Measure what Engram actually contributes, and confirm the ablation removes it.

`forward call count > 0` proves the patched function ran. It does not prove the
Engram contribution is gone -- that depends on the return convention being read
correctly, and on the contribution not also being applied somewhere else.

This measures it directly. For every Engram call it records

    ||returned - hidden|| / ||hidden||

which *is* the contribution the rest of the model receives, whatever the
convention. Baseline and ablated run in one process against the same prompts,
so the only difference is the toggle: the baseline number says how much Engram
does, and the ablated number must be exactly zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("engram-verify")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engram import gate_probe  # noqa: E402

PROMPTS = [
    "The Simpsons Halloween episodes are called Treehouse of",
    "Phoenix Wright : Ace",
    "import urllib.request\nresponse = ur",
    "Natalia sold clips to 48 friends in April, and half as many in May.",
    "玄奘西游记讲述的是",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default=os.environ.get("RESULT_DIR", ".") + "/engram_verify")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meter_dir = "/dev/shm/engram_verify"
    os.makedirs(meter_dir, exist_ok=True)
    os.environ[gate_probe.METER_DIR_ENV] = meter_dir
    analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gate_probe.install_in_workers(analysis_dir)
    gate_probe.install_meter()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=32, temperature=0.0)

    report = {}
    for phase in ("baseline", "ablated"):
        gate_probe.set_ablate(meter_dir, phase == "ablated")
        gate_probe.clear_meter(meter_dir)
        outputs = [
            llm.generate(p, sampling, use_tqdm=False)[0].outputs[0].text for p in PROMPTS
        ]
        stats = gate_probe.read_meter(meter_dir)
        report[phase] = {"contribution": stats, "completions": outputs}
        logger.info("%s: %s", phase, json.dumps(stats, indent=2))
        for prompt, text in zip(PROMPTS, outputs):
            logger.info("%s | %r -> %r", phase, prompt[-32:], text[:60])

    base = report["baseline"]["contribution"]
    abl = report["ablated"]["contribution"]
    report["verdict"] = {
        "engram_used_in_baseline": bool(base and base.get("mean_rel_norm", 0) > 1e-6),
        "contribution_zero_when_ablated": bool(abl and abl.get("max_rel_norm", 1) == 0.0),
        "completions_changed": sum(
            a != b for a, b in zip(report["baseline"]["completions"], report["ablated"]["completions"])
        ),
    }
    with open(os.path.join(args.out, "verify.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("===ENGRAM_VERIFY_JSON_BEGIN===")
    print(json.dumps(report))
    print("===ENGRAM_VERIFY_JSON_END===")
    v = report["verdict"]
    logger.info("VERDICT %s", json.dumps(v))
    return 0 if v["engram_used_in_baseline"] and v["contribution_zero_when_ablated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
