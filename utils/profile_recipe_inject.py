#!/usr/bin/env python3
"""Inject torch-profiler settings into an srt-slurm recipe (in place).

Used by the multinode launchers when PROFILE=1. The mutation happens on the
runtime copy of the recipe inside the srt-slurm clone, never on the checked-in
recipe. Three things are injected:

1. A first-class srtctl ``profiling:`` section (type: torch, per-phase
   start/stop steps). srtctl then exports the worker leader endpoints
   (PROFILE_{PREFILL,DECODE,AGG}_ENDPOINTS / _IPS), WORKER_PORT, and
   PROFILE_OUTPUT_DIR=/logs/profiles into the benchmark stage, and sets
   SGLANG_TORCH_PROFILER_DIR on the workers. The agentic benchmark stage's
   launch_agentic_profile_trigger (benchmarks/benchmark_lib.sh) consumes those
   endpoints to trigger bounded captures.
2. Eager execution: kernels replayed inside a captured CUDA graph lose their
   launching-operator attribution, so profiled servers run eager unless
   PROFILE_DISABLE_CUDA_GRAPH=0.
3. For vllm workers only: a ``profiler-config`` entry in vllm_config —
   srtctl's torch profiling env only covers sglang, and modern vllm configures
   the torch profiler exclusively through --profiler-config
   (torch_profiler_record_shapes is off by default and required here).
"""

import argparse
import json
import sys

import yaml

PROFILES_DIR = "/logs/profiles"
PHASES = ("prefill", "decode", "aggregated")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="Path to the srt-slurm recipe yaml (mutated in place)")
    parser.add_argument("--num-steps", type=int, default=2,
                        help="Forward steps per phase to capture")
    parser.add_argument("--keep-cuda-graphs", action="store_true",
                        help="Do not force eager execution")
    args = parser.parse_args()

    with open(args.recipe) as fh:
        recipe = yaml.safe_load(fh)

    backend = recipe.get("backend")
    if not isinstance(backend, dict):
        print(f"[profile-inject] no backend section in {args.recipe}", file=sys.stderr)
        return 1

    engine_cfg_key = next(
        (k for k in ("sglang_config", "vllm_config", "trtllm_config") if k in backend), None
    )
    if engine_cfg_key is None:
        print(f"[profile-inject] no engine config section in {args.recipe}", file=sys.stderr)
        return 1
    engine_cfg = backend[engine_cfg_key] or {}

    changed = []

    # 1. srtctl profiling section: per-phase step windows. The step counters
    # only matter for srt-slurm's own sa-bench trigger; the agentic custom
    # benchmark triggers captures itself. Declaring the section is what makes
    # srtctl export worker endpoints + profiler dirs.
    profiling = {"type": "torch"}
    for phase in PHASES:
        if phase in engine_cfg:
            profiling[phase] = {"start_step": 0, "stop_step": args.num_steps}
    recipe["profiling"] = profiling
    changed.append(f"profiling={profiling}")

    for phase in PHASES:
        cfg = engine_cfg.get(phase)
        if not isinstance(cfg, dict):
            continue
        if not args.keep_cuda_graphs:
            if engine_cfg_key == "sglang_config":
                cfg["disable-cuda-graph"] = True
                changed.append(f"{engine_cfg_key}.{phase}.disable-cuda-graph=true")
            elif engine_cfg_key == "vllm_config":
                cfg["enforce-eager"] = True
                changed.append(f"{engine_cfg_key}.{phase}.enforce-eager=true")
        if engine_cfg_key == "vllm_config":
            mode = "agg" if phase == "aggregated" else phase
            cfg["profiler-config"] = json.dumps({
                "profiler": "torch",
                "torch_profiler_dir": f"{PROFILES_DIR}/{mode}",
                "torch_profiler_record_shapes": True,
                # vllm has no client-side num_steps auto-stop; bound the
                # session engine-side or eager whole-run traces reach tens
                # of GB (and one unbounded session OOM-killed a slurm step).
                "ignore_frontend": True,
                "max_iterations": max(args.num_steps * 4, 8),
            })
            changed.append(f"{engine_cfg_key}.{phase}.profiler-config (torch)")

    with open(args.recipe, "w") as fh:
        yaml.safe_dump(recipe, fh, sort_keys=False, default_flow_style=False)

    print(f"[profile-inject] {args.recipe}:")
    for c in changed:
        print(f"[profile-inject]   {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
