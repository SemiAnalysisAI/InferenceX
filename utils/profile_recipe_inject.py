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
4. ExecutionTrace capture (sglang + vllm, unless --no-execution-trace, which
   the launchers pass for spec-decode recipes -- libtorch's observer segfaults
   under EAGLE-MTP -- or when PROFILE_EXECUTION_TRACE=0): the
   sitecustomize shim benchmarks/patches/execution_trace_shim/ wraps
   torch.profiler.profile.start/stop to record the operator DAG (dataflow
   edges, joinable to the kineto cpu_ops on rf_id <-> "Record function id")
   for every capture session. srtctl mounts the InferenceX checkout at
   /infmax-workspace and the shared log dir at /logs in every worker
   container, so delivery is two env vars per worker environment block:
   PYTHONPATH pointing at the shim dir, and PROFILE_EXECUTION_TRACE_DIR
   pointing under /logs/profiles/<mode> next to the kineto traces (which the
   benchmark stage's stage_profile_outputs already stages).
"""

import argparse
import json
import sys

import yaml

PROFILES_DIR = "/logs/profiles"
PHASES = ("prefill", "decode", "aggregated")
# In-container paths: srtctl mounts INFMAX_WORKSPACE (the InferenceX checkout,
# staged to a compute-visible FS by the launchers) at /infmax-workspace in
# every worker container. The shim dir contains only sitecustomize.py.
ET_SHIM_DIR = "/infmax-workspace/benchmarks/patches/execution_trace_shim"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="Path to the srt-slurm recipe yaml (mutated in place)")
    parser.add_argument("--num-steps", type=int, default=2,
                        help="Forward steps per phase to capture")
    parser.add_argument("--keep-cuda-graphs", action="store_true",
                        help="Do not force eager execution")
    parser.add_argument("--no-execution-trace", action="store_true",
                        help="Do not arm the ExecutionTraceObserver shim on workers")
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

    # ExecutionTrace shim: torch-level, so it works for both sglang schedulers
    # and vllm workers; trtllm workers don't run their capture through
    # torch.profiler, so skip them (no point importing the shim there).
    inject_et = (
        not args.no_execution_trace
        and engine_cfg_key in ("sglang_config", "vllm_config")
    )

    for phase in PHASES:
        cfg = engine_cfg.get(phase)
        if not isinstance(cfg, dict):
            continue
        if inject_et:
            mode = "agg" if phase == "aggregated" else phase
            env_key = f"{phase}_environment"
            env = backend.setdefault(env_key, {})
            # ET files land beside the phase's kineto traces; srtctl points
            # SGLANG_TORCH_PROFILER_DIR / the vllm profiler-config at
            # {PROFILES_DIR}/{mode}, and stage_profile_outputs' find covers
            # both levels.
            env["PROFILE_EXECUTION_TRACE_DIR"] = f"{PROFILES_DIR}/{mode}"
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{ET_SHIM_DIR}:{existing_pythonpath}"
                if existing_pythonpath
                else ET_SHIM_DIR
            )
            changed.append(f"backend.{env_key}: execution-trace shim armed")
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
