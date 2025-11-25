#!/usr/bin/env python3
"""
Convert latest lm-evaluation-harness and/or lighteval JSONs in a results dir
into Markdown tables for GitHub Actions job summary. Prints to stdout.

Usage (same as before, works even if FRAMEWORK/PRECISION env vars are empty):
  python3 utils/lm_eval_to_md.py \
    --results-dir /workspace/eval_out \
    --task gsm8k \
    --framework vLLM \
    --precision fp16 \
    --tp 4 \
    --ep 1 \
    --dp-attention false
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from glob import glob
from typing import Optional, Tuple, Dict, Any, List


# -----------------------
# Helpers
# -----------------------

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


def detect_framework_kind(data: Dict[str, Any]) -> str:
    """
    Classify JSON as:
      - 'lm-eval'   : lm-evaluation-harness style JSON
      - 'lighteval' : lighteval JSON
      - 'unknown'   : anything else
    """
    # lm-eval has lm_eval_version + results structure like results["gsm8k"]...  [oai_citation:0‡results_2025-11-25T08-30-41.513104.json](sediment://file_000000001658720790705168e4c51783)
    if "lm_eval_version" in data or "pretty_env_info" in data:
        return "lm-eval"
    # lighteval has config_general + config_tasks/results keyed by "<task>|<k>"  [oai_citation:1‡results_2025-11-25T08-40-05.199875.json](sediment://file_000000006f3872078dd9c458c614c1f7)
    if "config_general" in data and "results" in data:
        return "lighteval"
    return "unknown"


def find_all_jsons(results_dir: str) -> List[str]:
    paths = []
    for root, _, _ in os.walk(results_dir):
        for name in glob(os.path.join(root, "*.json")):
            paths.append(name)
    return paths


def find_latest_by_kind(results_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scan all JSONs under results_dir and return:
      (latest_lm_eval_json_path, latest_lighteval_json_path)
    """
    lm_eval_candidates = []
    lighteval_candidates = []

    for path in find_all_jsons(results_dir):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        kind = detect_framework_kind(data)
        mtime = os.path.getmtime(path)
        if kind == "lm-eval":
            lm_eval_candidates.append((mtime, path))
        elif kind == "lighteval":
            lighteval_candidates.append((mtime, path))

    lm_path = max(lm_eval_candidates, default=(None, None))[1]
    le_path = max(lighteval_candidates, default=(None, None))[1]
    return lm_path, le_path


# -----------------------
# lm-eval parsing
# -----------------------

def extract_lm_eval_metrics(data: Dict[str, Any], task: str) -> Dict[str, Any]:
    res_all = data.get("results", {}) or {}
    res = res_all.get(task) if isinstance(res_all, dict) else {}
    if not res and isinstance(res_all, dict) and res_all:
        any_key = next(iter(res_all.keys()))
        res = res_all.get(any_key, {})
        task = any_key

    strict = res.get("exact_match,strict-match")
    flex = res.get("exact_match,flexible-extract")
    strict_se = res.get("exact_match_stderr,strict-match")
    flex_se = res.get("exact_match_stderr,flexible-extract")

    n_eff = None
    ns = data.get("n-samples") or data.get("n_samples") or {}
    if isinstance(ns, dict):
        tdict = ns.get(task) or ns.get("gsm8k") or {}
        if isinstance(tdict, dict):
            n_eff = tdict.get("effective") or tdict.get("n_eff")

    model = data.get("model_name") \
        or data.get("configs", {}).get(task, {}).get("metadata", {}).get("model") \
        or data.get("config", {}).get("model") \
        or ""

    fewshot = None
    nshot = data.get("n-shot") or data.get("n_shot") or {}
    if isinstance(nshot, dict):
        fewshot = nshot.get(task) or nshot.get("gsm8k")

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
        "limit": limit,
    }


def render_lm_eval_section(path: str,
                           args,
                           framework_label: str,
                           precision_label: str) -> Tuple[str, Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    hardware = gpu_cpu_from_pretty_env(data.get("pretty_env_info", ""))
    m = extract_lm_eval_metrics(data, args.task)

    print(f"### {args.task} Evaluation (lm-eval-harness)\n")
    print("| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |")
    print("|---|---|---:|--:|--:|:--:|--:|--:|--:|")
    print(
        f"| {hardware} | {framework_label} | {precision_label} | {args.tp} | {args.ep} | "
        f"{str(args.dp_attention).lower()} | "
        f"{pct(m['strict'])}{se(m['strict_se'])} | "
        f"{pct(m['flex'])}{se(m['flex_se'])} | {m['n_eff'] or ''} |"
    )

    lim = m["limit"]
    lim_str = str(int(lim)) if isinstance(lim, (int, float)) else (str(lim) if lim is not None else "")
    fewshot = m["fewshot"] if m["fewshot"] is not None else ""
    print(
        f"\n_Model_: `{m['model']}` &nbsp;&nbsp; "
        f"_k-shot_: **{fewshot}** &nbsp;&nbsp; "
        f"_limit_: **{lim_str}**  \n"
        f"_Source_: `{os.path.basename(path)}`"
    )
    return hardware, m


# -----------------------
# lighteval parsing
# -----------------------

def extract_lighteval_metrics(data: Dict[str, Any], task_base: str) -> Dict[str, Any]:
    res_all = data.get("results", {}) or {}

    # Prefer task-specific key like "gsm8k|5" over "all"  [oai_citation:2‡results_2025-11-25T08-40-05.199875.json](sediment://file_000000006f3872078dd9c458c614c1f7)
    task_key = None
    for k in res_all.keys():
        if k.startswith(task_base):
            task_key = k
            break
    if task_key is None and "all" in res_all:
        task_key = "all"

    r = res_all.get(task_key, {})
    em = r.get("extractive_match")
    em_se = r.get("extractive_match_stderr")

    # Fewshot & other metadata from config_tasks if available
    fewshot = None
    cfg_tasks = data.get("config_tasks", {})
    if isinstance(cfg_tasks, dict) and task_key in cfg_tasks:
        fewshot = cfg_tasks[task_key].get("num_fewshots")

    # Model name from config_general
    cg = data.get("config_general", {}) or {}
    model = cg.get("model_name") or cg.get("model_config", {}).get("model_name", "")

    return {
        "task": task_key or task_base,
        "em": em,
        "em_se": em_se,
        "fewshot": fewshot,
        "model": model,
        # lighteval JSON you showed doesn’t expose an obvious effective N; leave blank
        "n_eff": None,
    }


def render_lighteval_section(path: str,
                             args,
                             framework_label: str,
                             precision_label: str,
                             hardware_fallback: Optional[str]) -> None:
    with open(path, "r") as f:
        data = json.load(f)

    m = extract_lighteval_metrics(data, args.task)
    hardware = hardware_fallback or "Unknown GPU"

    print(f"### {args.task} Evaluation (lighteval)\n")
    print("| Hardware | Framework | Precision | TP | EP | DP Attention | Extractive Match | N (eff) |")
    print("|---|---|---:|--:|--:|:--:|--:|--:|")
    print(
        f"| {hardware} | {framework_label} | {precision_label} | {args.tp} | {args.ep} | "
        f"{str(args.dp_attention).lower()} | "
        f"{pct(m['em'])}{se(m['em_se'])} | {m['n_eff'] or ''} |"
    )

    fewshot = m["fewshot"] if m["fewshot"] is not None else ""
    print(
        f"\n_Model_: `{m['model']}` &nbsp;&nbsp; "
        f"_k-shot_: **{fewshot}**  \n"
        f"_Source_: `{os.path.basename(path)}`"
    )


# -----------------------
# main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--framework", default=os.environ.get("FRAMEWORK", ""))
    ap.add_argument("--precision", default=os.environ.get("PRECISION", ""))
    ap.add_argument("--tp", default=os.environ.get("TP", "1"))
    ap.add_argument("--ep", default=os.environ.get("EP_SIZE", "1"))
    ap.add_argument("--dp-attention", default=os.environ.get("DP_ATTENTION", "false"))
    args = ap.parse_args()

    # Robust defaults if env vars / CLI args are empty
    framework_label = args.framework or os.environ.get("FRAMEWORK") or "unknown"
    precision_label = args.precision or os.environ.get("PRECISION") or "unknown"

    lm_path, le_path = find_latest_by_kind(args.results_dir)

    if not lm_path and not le_path:
        print(f"### {args.task} Evaluation\n")
        print(f"> No result JSON found in `{args.results_dir}`.")
        return

    hardware_from_lm = None

    # 1) lm-eval section (if present)
    if lm_path:
        hardware_from_lm, _ = render_lm_eval_section(
            lm_path, args, framework_label, precision_label
        )

    # Spacer between sections if both exist
    if lm_path and le_path:
        print("\n")

    # 2) lighteval section (if present)
    if le_path:
        render_lighteval_section(
            le_path, args, framework_label, precision_label, hardware_from_lm
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never blow up the CI summary; emit a helpful line instead.
        print(f"> Failed to render evaluation summary: {e}")
        sys.exit(0)