"""Attribute the Kimi-K3 gfx942 A16W4 SiTUv2 MoE illegal access.

Drives the aiter fused MoE at the exact K3 shape on one GPU, with no vLLM in
the picture, and checks it against the independent FP32 oracle in k3_oracle.py.

One case per process, selected by $PROBE_CASE. A HIP fault poisons the context
for every later launch in the same process, so cases that must stay independent
cannot share one:

    m2      M=2 alone.       The failing canary's PP0 ranks died right after
                             their M=2 dispatch.
    m4096   M=4096 alone.    PP1 ran only M=4096 and survived, but it was never
                             synchronized before PP0 brought the job down, so
                             that observation needs confirming here.
    seq     M=4096 then M=2, deliberately in ONE process -- this is the state
                             reuse hypothesis, and splitting it would destroy
                             the thing under test.
    graph   capture + replay at M=2, for a kernel that is correct in eager and
                             wrong under capture.

Writes $PROBE_ARTIFACT_DIR/k3_probe_$PROBE_CASE.json. Exit status is 0 whenever
the probe itself ran to completion -- a faulting kernel is a result, not a
probe failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import k3_oracle as oracle


FAULT_MARKERS = (
    "illegal memory access",
    "hipErrorIllegalAddress",
    "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
    "HSA_STATUS_ERROR_EXCEPTION",
    "Memory access fault",
)


class KernelNameCapture(logging.Handler):
    """aiter logs the heuristic FlyDSL fallback (kn1=..., kn2=...) at warning."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        if "kn1=" in message or "FlyDSL" in message:
            self.lines.append(message)

    def take(self) -> list[str]:
        taken, self.lines = self.lines, []
        return taken


def parse_kernel_names(lines: list[str]) -> tuple[str, str]:
    kn1 = kn2 = ""
    for line in lines:
        for token in line.replace("(", " ").replace(")", " ").split():
            if token.startswith("kn1="):
                kn1 = token[4:].strip("',\"")
            elif token.startswith("kn2="):
                kn2 = token[4:].strip("',\"")
    return kn1, kn2


def is_fault(text: str) -> bool:
    return any(marker in text for marker in FAULT_MARKERS)


def sha256_file(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def provenance() -> dict:
    """Identify the aiter actually imported, and fail loudly on the wrong one.

    The overlay is installed into the image's site-packages at run time, so a
    result is only meaningful alongside the digest of the files that produced
    it -- a silently un-applied overlay otherwise looks like a clean baseline.
    """
    import aiter
    from aiter.jit.utils.chip_info import get_gfx

    root = os.path.dirname(os.path.abspath(aiter.__file__))
    expected = os.environ.get("PROBE_EXPECT_AITER_ROOT", "")
    if expected and os.path.realpath(root) != os.path.realpath(expected):
        raise SystemExit(
            f"PROBE_ABORT wrong aiter: imported {root}, expected {expected}"
        )

    info = dict(
        aiter_file=os.path.abspath(aiter.__file__),
        aiter_root=root,
        aiter_version=getattr(aiter, "__version__", "unknown"),
        overlay_ref=os.environ.get("KIMIK3_AITER_OVERLAY_REF", "unset"),
        sha256_fused_moe=sha256_file(os.path.join(root, "fused_moe.py")),
        sha256_mixed_moe_gemm_2stage=sha256_file(
            os.path.join(root, "ops", "flydsl", "kernels", "mixed_moe_gemm_2stage.py")
        ),
        gfx=get_gfx(),
        torch=torch.__version__,
        hip=torch.version.hip,
        device_count=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0),
        serialize_kernel=os.environ.get("AMD_SERIALIZE_KERNEL", "unset"),
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES", "unset"),
        rocr_visible_devices=os.environ.get("ROCR_VISIBLE_DEVICES", "unset"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS", "unset"),
        aiter_jit_dir=os.environ.get("AITER_JIT_DIR", "unset"),
        triton_cache_dir=os.environ.get("TRITON_CACHE_DIR", "unset"),
    )
    for key, value in info.items():
        print(f"PROBE_ENV {key}={value}", flush=True)
    return info


def build_aiter_inputs(case):
    """Shuffle the oracle's canonical tensors into the layout aiter expects.

    The gate_up flag is not cosmetic: A16W4 stage1 reads w1 gate/up interleaved
    and w2 plain, so w1 and its scale take True while w2 and its scale take
    False. op_tests/test_moe_2stage.py's a16w4 branch is the reference. Feeding
    the wrong flag produces out-of-range reads that look exactly like the bug
    under investigation.
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    # shuffle_scale_a16w4 operates on the flattened [E*N, K/32] view used by
    # AITER's own a16w4 tests; the oracle intentionally keeps [E, N, K/32].
    w1_scale = case.w1_scale.view(-1, case.w1_scale.shape[-1])
    w2_scale = case.w2_scale.view(-1, case.w2_scale.shape[-1])

    return dict(
        w1=shuffle_weight_a16w4(case.w1_packed, 16, True),
        w1_scale=shuffle_scale_a16w4(w1_scale, oracle.EXPERTS, True),
        w2=shuffle_weight_a16w4(case.w2_packed, 16, False),
        w2_scale=shuffle_scale_a16w4(w2_scale, oracle.EXPERTS, False),
    )


def launch_aiter(case, shuffled):
    """Enqueue the fused MoE. Never synchronizes: the caller owns that.

    swiglu_limit is deliberately not passed. The kernel clamps only on its
    swiglu path, and A16W4 does not even forward the argument
    (pass_swiglu_limit=not _is_a16w4), so the oracle leaves SiTUv2 unclamped to
    match.
    """
    import aiter
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe(
        case.inp,
        shuffled["w1"],
        shuffled["w2"],
        case.topk_w,
        case.topk_ids,
        w1_scale=shuffled["w1_scale"],
        w2_scale=shuffled["w2_scale"],
        quant_type=aiter.QuantType.per_1x32,
        activation=aiter.ActivationType.Situv2,
        doweight_stage1=False,
        beta=oracle.SITU_BETA,
        linear_beta=oracle.SITU_LINEAR_BETA,
        gate_mode=GateMode.SEPARATED.value,
    )


def blank_numeric() -> dict:
    return dict(
        ran=False,
        # Only the fused output is observable through this API. Reporting the
        # e2e verdict as a per-stage one would be an invented attribution.
        stage1_pass=None,
        stage2_pass=None,
        stage_attribution="unmeasured: fused_moe exposes only the final output",
        e2e_pass=False,
        cosine=0.0, max_abs_error=0.0, relative_L2=0.0, percent_close=0.0,
        nan_count=0, inf_count=0,
    )


def check_numeric(case, got, label: str) -> dict:
    reference = oracle.ref_stage2(case, oracle.ref_stage1(case))
    metrics = oracle.compare(got.float(), reference, label=label)
    numeric = blank_numeric()
    numeric.update(metrics)
    numeric["ran"] = True
    numeric["e2e_pass"] = bool(oracle.verdict(metrics))
    print(oracle.report(metrics), flush=True)
    return numeric


def measure(label: str, m: int, capture: KernelNameCapture, body) -> dict:
    step = dict(label=label, M=m, kn1="", kn2="", crashed=False, fault=False,
                crash_site="", numeric=blank_numeric())
    print(f"--- {label} (M={m}) ---", flush=True)
    capture.take()
    try:
        step["numeric"] = body()
    except Exception:
        text = traceback.format_exc()
        step["crashed"] = True
        step["fault"] = is_fault(text)
        step["crash_site"] = text.strip().splitlines()[-1][:400]
        step["traceback"] = text[-4000:]
        print(text, flush=True)
    step["kn1"], step["kn2"] = parse_kernel_names(capture.take())
    print(f"PROBE_STEP {label} kn1={step['kn1']} kn2={step['kn2']} "
          f"crashed={step['crashed']} fault={step['fault']}", flush=True)
    return step


def eager_step(m: int, capture: KernelNameCapture, verify: bool = True) -> dict:
    def body():
        case = oracle.make_case(m, device="cuda")
        shuffled = build_aiter_inputs(case)
        got = launch_aiter(case, shuffled)
        torch.cuda.synchronize()
        if not verify:
            return blank_numeric()
        return check_numeric(case, got, f"M={m}/eager/e2e")

    return measure(f"eager_m{m}", m, capture, body)


def graph_step(m: int, capture: KernelNameCapture) -> dict:
    def body():
        case = oracle.make_case(m, device="cuda")
        shuffled = build_aiter_inputs(case)

        # Warm up on a side stream so JIT and autotune finish their allocations
        # before capture, then synchronize OUTSIDE the capture region. A
        # synchronize inside it is illegal and would abort the capture rather
        # than test the kernel.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                launch_aiter(case, shuffled)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = launch_aiter(case, shuffled)
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()
        return check_numeric(case, got, f"M={m}/cudagraph/e2e")

    return measure(f"cudagraph_m{m}", m, capture, body)


def case_m2(capture):
    return [eager_step(2, capture)]


def case_m4096(capture):
    return [eager_step(4096, capture)]


def case_seq(capture):
    """M=4096 then M=2 in ONE process: the state reuse hypothesis.

    The second step runs even though the first was verified, because the point
    is the transition, not either shape alone.
    """
    steps = [eager_step(4096, capture)]
    if steps[0]["fault"]:
        print("PROBE_ABORT_SEQ first step faulted; the context is poisoned and "
              "the M=2 step would report garbage", flush=True)
        return steps
    steps.append(eager_step(2, capture))
    return steps


def case_graph(capture):
    return [graph_step(2, capture)]


CASES = {
    "m2": case_m2,
    "m4096": case_m4096,
    "seq": case_seq,
    "graph": case_graph,
}


def main() -> int:
    name = os.environ.get("PROBE_CASE", "m2")
    if name not in CASES:
        raise SystemExit(f"PROBE_ABORT unknown PROBE_CASE={name}; "
                         f"expected one of {sorted(CASES)}")
    artifact_dir = os.environ.get("PROBE_ARTIFACT_DIR", ".")
    os.makedirs(artifact_dir, exist_ok=True)

    capture = KernelNameCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger("aiter").addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    summary = dict(
        case=name,
        env=provenance(),
        shape=dict(model_dim=oracle.MODEL_DIM, inter_dim=oracle.INTER_DIM,
                   experts=oracle.EXPERTS, topk=oracle.TOPK),
        situ=dict(beta=oracle.SITU_BETA, linear_beta=oracle.SITU_LINEAR_BETA,
                  limit=oracle.SITU_LIMIT),
        steps=CASES[name](capture),
    )
    summary["faulted"] = any(step["fault"] for step in summary["steps"])
    summary["crashed"] = any(step["crashed"] for step in summary["steps"])

    out_path = os.path.join(artifact_dir, f"k3_probe_{name}.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"PROBE_RESULTS={out_path}", flush=True)
    print(f"PROBE_DONE case={name} faulted={summary['faulted']}", flush=True)
    sys.stdout.flush()

    if summary["faulted"]:
        # A poisoned HIP context usually aborts the runtime during interpreter
        # shutdown, which would bury this completed result under a SIGABRT.
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
