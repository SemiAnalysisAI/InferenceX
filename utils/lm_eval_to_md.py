#!/usr/bin/env python3
"""
Convert latest lm-evaluation-harness JSON in a results dir into a Markdown table
for GitHub Actions job summary. Prints to stdout.

Usage:
  python3 bench_serving/scripts/lm_eval_to_md.py \
    --results-dir /workspace/eval_out \
    --task gsm8k \
    --framework vLLM \
    --precision fp16 \
    --tp 4 \
    --ep 1 \
    --dp-attention false
"""
import argparse, json, os, re, sys
from collections import Counter
from glob import glob

def find_latest_json(results_dir: str):
    paths = []
    for root, _, _ in os.walk(results_dir):
        paths.extend(glob(os.path.join(root, "*.json")))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def pct(x):
    return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "N/A"

def se(x):
    return f" \u00B1{(x*100):.2f}%" if isinstance(x, (int, float)) else ""

def gpu_cpu_from_pretty_env(pe: str):
    if not isinstance(pe, str) or not pe:
        return "Unknown GPU"
    gpu_lines = [l for l in pe.splitlines() if l.startswith("GPU ")]
    names = [re.sub(r"GPU \d+:\s*", "", l).strip() for l in gpu_lines]
    c = Counter(names)
    gpu_summary = " + ".join([f"{n}\u00D7 {name}" for name, n in c.items()]) if c else "Unknown GPU"
    cpu_line = next((l.split(":", 1)[1].strip() for l in pe.splitlines() if l.startswith("Model name:")), None)
    return gpu_summary + (f" ({cpu_line})" if cpu_line else "")

def extract_metrics(data: dict, task: str):
    # results section can vary across harness versions
    res_all = data.get("results", {}) or {}
    res = res_all.get(task) if isinstance(res_all, dict) else {}
    if not res and isinstance(res_all, dict) and res_all:
        # fallback to first key if requested task missing
        any_key = next(iter(res_all.keys()))
        res = res_all.get(any_key, {})
        task = any_key

    strict = res.get("exact_match,strict-match")
    flex   = res.get("exact_match,flexible-extract")
    strict_se = res.get("exact_match_stderr,strict-match")
    flex_se   = res.get("exact_match_stderr,flexible-extract")

    n_eff = None
    ns = data.get("n-samples") or data.get("n_samples") or {}
    if isinstance(ns, dict):
        tdict = ns.get(task) or ns.get("gsm8k") or {}
        if isinstance(tdict, dict):
            n_eff = tdict.get("effective") or tdict.get("n_eff")

    # model/fewshot/limit are scattered depending on version
    model = data.get("model_name") \
        or data.get("configs", {}).get(task, {}).get("metadata", {}).get("model") \
        or data.get("config", {}).get("model") \
        or ""

    # k-shot
    fewshot = None
    nshot = data.get("n-shot") or data.get("n_shot") or {}
    if isinstance(nshot, dict):
        fewshot = nshot.get(task) or nshot.get("gsm8k")

    # limit
    limit = None
    cfg = data.get("config") or {}
    if isinstance(cfg, dict):
        limit = cfg.get("limit")

    return {
        "task": task,
        "strict": strict,
        "flex": flex,
        "strict_se": strict_se,
        "flex_se": flex_se,
        "n_eff": n_eff,
        "model": model,
        "fewshot": fewshot,
        "limit": limit
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--framework", default=os.environ.get("FRAMEWORK", "vLLM"))
    ap.add_argument("--precision", default=os.environ.get("PRECISION", "fp16"))
    ap.add_argument("--tp", default=os.environ.get("TP", "1"))
    ap.add_argument("--ep", default=os.environ.get("EP_SIZE", "1"))
    ap.add_argument("--dp-attention", default=os.environ.get("DP_ATTENTION", "false"))
    args = ap.parse_args()

    path = find_latest_json(args.results_dir)
    print(f"### {args.task} Evaluation\n")
    if not path or not os.path.exists(path):
        print(f"> No result JSON found in `{args.results_dir}`.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    hardware = gpu_cpu_from_pretty_env(data.get("pretty_env_info", ""))
    m = extract_metrics(data, args.task)

    print("| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |")
    print("|---|---|---:|--:|--:|:--:|--:|--:|--:|")
    print(f"| {hardware} | {args.framework} | {args.precision} | {args.tp} | {args.ep} | {str(args.dp_attention).lower()} | "
          f"{pct(m['strict'])}{se(m['strict_se'])} | {pct(m['flex'])}{se(m['flex_se'])} | {m['n_eff'] or ''} |")

    # metadata line
    lim = m["limit"]
    lim_str = str(int(lim)) if isinstance(lim, (int, float)) else (str(lim) if lim is not None else "")
    fewshot = m["fewshot"] if m["fewshot"] is not None else ""
    print(f"\n_Model_: `{m['model']}` &nbsp;&nbsp; _k-shot_: **{fewshot}** &nbsp;&nbsp; _limit_: **{lim_str}**  \n_Source_: `{os.path.basename(path)}`")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never blow up the CI summary; emit a helpful line instead.
        print(f"> Failed to render evaluation summary: {e}")
        sys.exit(0)