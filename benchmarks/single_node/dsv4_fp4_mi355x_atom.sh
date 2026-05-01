#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE"

# ROCm/ATOM#650 is still a PR1 DSv4 skeleton. The local overlay below gives
# DSv4 persistent per-request cache slots so CONC>1 no longer corrupts the
# recurrent KV/compressor/indexer state. It keeps sparse attention per sequence,
# but batches attention projections, mHC, and MoE/FFN work layer-by-layer.
if [ "$EP_SIZE" -ne 1 ]; then
    echo "FATAL: ROCm/ATOM#650 PR1 has not validated expert parallel serving; EP_SIZE must be 1, got $EP_SIZE" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1

# DSv4-specific ATOM env vars. Prefer the native AITER MXFP4 MoE path after
# overlaying the AITER perf stack below. Set AITER_DSV4_FP4_MOE_BACKEND=triton
# to return to ROCm/ATOM#650's original triton_kernels matmul_ogs path.
if [ "${AITER_DSV4_PERF_STACK:-1}" = "1" ]; then
    DEFAULT_AITER_DSV4_FP4_MOE_BACKEND=native
else
    DEFAULT_AITER_DSV4_FP4_MOE_BACKEND=triton
fi
AITER_DSV4_FP4_MOE_BACKEND=${AITER_DSV4_FP4_MOE_BACKEND:-$DEFAULT_AITER_DSV4_FP4_MOE_BACKEND}
if [ "$AITER_DSV4_FP4_MOE_BACKEND" = "triton" ]; then
    export ATOM_USE_TRITON_MOE=1
else
    unset ATOM_USE_TRITON_MOE
    unset ATOM_USE_TRITON_GEMM
fi
export AITER_LOG_LEVEL=WARNING

# Pull in the AITER pieces that matter for DSv4 FP4 on MI355X:
#   * origin/main@bb4ea92e includes ROCm/aiter#2770 a16w4 MoE support,
#     ROCm/aiter#2916 mhc_pre device-allocation fix, and ROCm/aiter#2924
#     FlyDSL GDR decode tuned configs.
#   * ROCm/aiter#2822 speeds up batched MXFP4 GEMM on gfx950.
#   * ROCm/aiter#2900 fixes MXFP4 scale padding for non-256 K.
#   * ROCm/aiter#2642 enables/fixes TP=4/8 MXFP4 MoE dispatch.
#   * sunway513/aiter@e450e4d adds DSv4 FP4 MoE tuned rows that route
#     eligible token counts to FlyDSL FP4 MoE kernels instead of default CK
#     heuristics when the image has the optional flydsl package.
#   * Oseltamivir/aiter@023eb3b adds DSv4 sparse MQA sink and Indexer
#     scorer/top-k Triton ops so ATOM can avoid the PR650 Torch fallback, plus
#     a 4x128 sparse-attn tile that reduces repeated QK score work for D=512.
#
# The open performance PRs cherry-pick cleanly over the pinned main SHA as
# of 2026-04-29.
# Keep this as a runtime overlay until AMD publishes an ATOM image with these
# AITER changes baked in; then remove this block and pin that image instead.
if [ "${AITER_DSV4_PERF_STACK:-1}" = "1" ]; then
    AITER_PERF_REPO=${AITER_PERF_REPO:-https://github.com/ROCm/aiter.git}
    AITER_PERF_DIR=${AITER_PERF_DIR:-/tmp/aiter-dsv4-fp4-perf}
    AITER_PERF_BASE_SHA=${AITER_PERF_BASE_SHA:-bb4ea92eaf7a8420ab6bcc460095d310d02dd628}
    AITER_PERF_PATCH_REFS=(
        "${AITER_PERF_BATCHED_FP4_REF:-pull/2822/head}"
        "${AITER_PERF_MXFP4_SCALE_REF:-pull/2900/head}"
        "${AITER_PERF_MOE_REF:-pull/2642/head}"
    )
    AITER_DSV4_TUNED_FMOE=${AITER_DSV4_TUNED_FMOE:-1}
    AITER_DSV4_TUNED_FMOE_REPO=${AITER_DSV4_TUNED_FMOE_REPO:-https://github.com/sunway513/aiter.git}
    AITER_DSV4_TUNED_FMOE_SHA=${AITER_DSV4_TUNED_FMOE_SHA:-e450e4deb992c5ecd9db5ef5ef79f1d40208bc9c}
    AITER_DSV4_TUNED_FMOE_PATH=${AITER_DSV4_TUNED_FMOE_PATH:-aiter/configs/model_configs/dsv4_fp4_tuned_fmoe.csv}
    AITER_DSV4_SPARSE_INDEXER=${AITER_DSV4_SPARSE_INDEXER:-1}
    AITER_DSV4_SPARSE_INDEXER_REPO=${AITER_DSV4_SPARSE_INDEXER_REPO:-https://github.com/Oseltamivir/aiter.git}
    AITER_DSV4_SPARSE_INDEXER_REF=${AITER_DSV4_SPARSE_INDEXER_REF:-dsv4-sparse-indexer}
    AITER_DSV4_SPARSE_INDEXER_SHA=${AITER_DSV4_SPARSE_INDEXER_SHA:-023eb3bc190cd58646517a6c96a3b6b799bc1f40}

    rm -rf "$AITER_PERF_DIR"
    git clone --filter=blob:none "$AITER_PERF_REPO" "$AITER_PERF_DIR"
    (
        cd "$AITER_PERF_DIR"
        git fetch --depth=1 origin "$AITER_PERF_BASE_SHA"
        git checkout --force "$AITER_PERF_BASE_SHA"
        test "$(git rev-parse HEAD)" = "$AITER_PERF_BASE_SHA"

        for ref in "${AITER_PERF_PATCH_REFS[@]}"; do
            # Do not use --depth=1 here. A shallow PR-head fetch hides the
            # parent commit and makes git treat the cherry-pick as add/add
            # conflicts across unrelated files.
            git fetch origin "$ref"
            git cherry-pick --no-commit FETCH_HEAD
        done

        if [ "$AITER_DSV4_TUNED_FMOE" = "1" ]; then
            mkdir -p "$(dirname "$AITER_DSV4_TUNED_FMOE_PATH")"
            git fetch --depth=1 "$AITER_DSV4_TUNED_FMOE_REPO" "$AITER_DSV4_TUNED_FMOE_SHA"
            test "$(git rev-parse FETCH_HEAD)" = "$AITER_DSV4_TUNED_FMOE_SHA"
            git show "FETCH_HEAD:$AITER_DSV4_TUNED_FMOE_PATH" > "$AITER_DSV4_TUNED_FMOE_PATH"
            grep -q '7168,512,385,6,ActivationType.Silu' "$AITER_DSV4_TUNED_FMOE_PATH" \
                || { echo "FATAL: DSv4 FP4 tuned fMoE rows not found in $AITER_DSV4_TUNED_FMOE_PATH"; exit 1; }
        fi

        if [ "$AITER_DSV4_SPARSE_INDEXER" = "1" ]; then
            git fetch --depth=1 "$AITER_DSV4_SPARSE_INDEXER_REPO" "$AITER_DSV4_SPARSE_INDEXER_REF"
            test "$(git rev-parse FETCH_HEAD)" = "$AITER_DSV4_SPARSE_INDEXER_SHA"
            for file_path in \
                aiter/ops/triton/_triton_kernels/attention/dsv4_indexer.py \
                aiter/ops/triton/_triton_kernels/attention/sparse_mqa_sink.py \
                aiter/ops/triton/attention/dsv4_indexer.py \
                aiter/ops/triton/attention/sparse_mqa_sink.py \
                op_tests/test_dsv4_indexer.py \
                op_tests/test_sparse_mqa_sink.py
            do
                mkdir -p "$(dirname "$file_path")"
                git show "FETCH_HEAD:$file_path" > "$file_path"
            done
            python3 - <<'PYEOF'
from pathlib import Path

path = Path("aiter/ops/triton/__init__.py")
source = path.read_text()
needle = '    "prefill_attention": "attention.prefill_attention",\n'
insert = '    "dsv4_indexer": "attention.dsv4_indexer",\n    "sparse_mqa_sink": "attention.sparse_mqa_sink",\n'
if '"dsv4_indexer": "attention.dsv4_indexer"' not in source:
    if needle not in source:
        raise SystemExit("FATAL: aiter triton __init__.py missing attention map anchor")
    source = source.replace(needle, needle + insert, 1)
elif '"sparse_mqa_sink": "attention.sparse_mqa_sink"' not in source:
    source = source.replace(
        '    "dsv4_indexer": "attention.dsv4_indexer",\n',
        insert,
        1,
    )
path.write_text(source)
PYEOF
        fi

        if [ ! -d 3rdparty/composable_kernel/include ]; then
            git submodule update --init --recursive --depth=1 3rdparty/composable_kernel \
                || git submodule update --init --recursive 3rdparty/composable_kernel
        fi

        PREBUILD_KERNELS=${AITER_PREBUILD_KERNELS:-0} \
        python3 -m pip install --no-deps --no-build-isolation --force-reinstall -e .
    )

    if [ "$AITER_DSV4_TUNED_FMOE" = "1" ]; then
        export AITER_DSV4_TUNED_FMOE_FILE="$AITER_PERF_DIR/$AITER_DSV4_TUNED_FMOE_PATH"
    fi
    if [ "$AITER_DSV4_TUNED_FMOE" = "1" ] && [ -z "${AITER_CONFIG_FMOE:-}" ]; then
        export AITER_CONFIG_FMOE="$AITER_PERF_DIR/aiter/configs/tuned_fmoe.csv:$AITER_DSV4_TUNED_FMOE_FILE"
    fi

    python3 - <<'PYEOF'
import csv
import os
from pathlib import Path
import aiter

root = Path(aiter.__file__).resolve().parent
moe = (root / "fused_moe.py").read_text()
mhc = (root / "ops" / "mhc.py").read_text()
fp4_utils = (root / "utility" / "fp4_utils.py").read_text()
dsv4_tuned_fmoe = Path(os.environ["AITER_DSV4_TUNED_FMOE_FILE"]) if os.environ.get("AITER_DSV4_TUNED_FMOE_FILE") else None
required = {
    "native MXFP4 MoE skip_inter_quant": "skip_inter_quant" in moe,
    "mhc_pre device allocation fix": (
        "device = residual.device" in mhc
        and "dtype=dtypes.bf16, device=device" in mhc
    ),
    "FlyDSL GDR decode tuned configs": (
        root / "ops" / "flydsl" / "gdr_decode_tuned.jsonl"
    ).exists(),
    "MXFP4 scaleN_pad fix": "scaleN_pad" in fp4_utils,
    "DSv4 FP4 tuned fMoE config": dsv4_tuned_fmoe is None or dsv4_tuned_fmoe.exists(),
    "DSv4 sparse MQA sink Triton op": (
        root / "ops" / "triton" / "attention" / "sparse_mqa_sink.py"
    ).exists(),
    "DSv4 Indexer Triton op": (
        root / "ops" / "triton" / "attention" / "dsv4_indexer.py"
    ).exists(),
}
missing = [name for name, ok in required.items() if not ok]
if missing:
    raise SystemExit(f"FATAL: AITER DSv4 perf stack verification failed: {missing}")

if dsv4_tuned_fmoe is not None and dsv4_tuned_fmoe.exists():
    config_paths = os.environ.get("AITER_CONFIG_FMOE", "").split(":")
    if str(dsv4_tuned_fmoe) not in config_paths:
        print(
            "WARN: AITER_CONFIG_FMOE was user-supplied and does not include "
            f"{dsv4_tuned_fmoe}; DSv4 tuned fMoE rows may not be active."
        )
    try:
        from aiter.ops.flydsl import is_flydsl_available
    except Exception as exc:
        print(f"aiter DSv4 tuned fMoE installed; FlyDSL availability check failed: {exc!r}")
    else:
        flydsl_available = is_flydsl_available()
        print(f"aiter FlyDSL available: {flydsl_available}")
        if flydsl_available:
            from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

            missing_kernels = set()
            with dsv4_tuned_fmoe.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    for name in (row.get("kernelName1", ""), row.get("kernelName2", "")):
                        if name.startswith("flydsl_") and get_flydsl_kernel_params(name) is None:
                            missing_kernels.add(name)
            if missing_kernels:
                raise SystemExit(
                    "FATAL: DSv4 FP4 tuned fMoE references missing FlyDSL kernels: "
                    f"{sorted(missing_kernels)[:5]}"
                )
print(f"aiter DSv4 perf stack imported from: {root}")
PYEOF
else
    echo "WARN: AITER_DSV4_PERF_STACK=0; using image-provided aiter"
fi

# Apply an ATOM DSv4 overlay over the image's wheel-installed atom. The default
# fork commit is ROCm/ATOM#650 head plus AITER sparse_attn/Indexer dispatch.
# Keep ATOM_PR_* overridable so this can be pointed back at ROCm/ATOM#650 while
# debugging upstream movement.
ATOM_PR_REPO=${ATOM_PR_REPO:-https://github.com/Oseltamivir/ATOM.git}
ATOM_PR_REF=${ATOM_PR_REF:-dsv4-pr650-head-aiter-sparse}
ATOM_PR_SHA=${ATOM_PR_SHA:-486d35fdeeb50c471329c2cd08681df9c3ad53ce}
export ATOM_PR_DIR="/tmp/atom-pr650"

if [ ! -d "$ATOM_PR_DIR/.git" ]; then
    git clone --filter=blob:none "$ATOM_PR_REPO" "$ATOM_PR_DIR"
fi
(
    cd "$ATOM_PR_DIR"
    git remote set-url origin "$ATOM_PR_REPO"
    # Try a targeted fetch first (fast); fall back to fetching the PR ref if
    # the server doesn't allow fetching the SHA directly.
    git fetch --depth=1 origin "$ATOM_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin "$ATOM_PR_REF" 2>/dev/null \
        || git fetch --depth=1 origin pull/650/head
    git checkout --force "$ATOM_PR_SHA"
    test "$(git rev-parse HEAD)" = "$ATOM_PR_SHA"

    # ROCm/aiter#2916 keeps ATOM's mhc_pre fast path usable. Fail if the
    # pinned ATOM checkout no longer exposes that aiter hook; silently
    # disabling it would hide the regression this benchmark is meant to catch.
    grep -q 'mhc_pre = getattr(_aiter, "mhc_pre", None)' atom/models/deepseek_v4.py \
        || { echo "FATAL: ATOM DSv4 mhc_pre aiter hook not found"; exit 1; }

    # ROCm/ATOM#650 sparse_attn_v4.py is a correctness-first torch fallback.
    # Route DSv4 sparse MQA through the forked AITER Triton kernel first. Keep
    # the old chunk/decode mitigations only as an explicit fallback path.
    python3 - <<'PYEOF'
from pathlib import Path

path = Path("atom/model_ops/sparse_attn_v4.py")
source = path.read_text()
marker = "ATOM_DSV4_AITER_SPARSE_ATTN"
if marker not in source:
    source = source.replace(
        "from typing import Tuple\n\nimport torch\n",
        "from typing import Tuple\n\nimport os\n\nimport torch\n",
        1,
    )
    old = """    out_dtype = q.dtype
    device = q.device

    # ----- Gather KV per query position -----
"""
    new = """    out_dtype = q.dtype
    device = q.device

    if os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN", "1") == "1" and q.is_cuda:
        try:
            from aiter.ops.triton.attention.sparse_mqa_sink import sparse_mqa_sink

            block_size = int(
                os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_SIZE", "128")
                or "128"
            )
            tile_k = int(
                os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_TILE_K", "64") or "64"
            )
            block_h = int(
                os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_H", "4") or "4"
            )
            block_d = int(
                os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_D", "128") or "128"
            )
            score_d = int(
                os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_SCORE_D", "64") or "64"
            )
            q_flat = q.reshape(B * M, H, D).contiguous()
            topk_flat = topk_idxs.reshape(B * M, K).contiguous().int()
            num_blocks = (N + block_size - 1) // block_size
            padded_n = num_blocks * block_size
            if padded_n != N:
                kv_padded = kv.new_zeros((B, padded_n, D))
                kv_padded[:, :N] = kv
            else:
                kv_padded = kv.contiguous()
            kv_blocks = (
                kv_padded.view(B, num_blocks, block_size, D)
                .reshape(B * num_blocks, block_size, D)
                .contiguous()
            )
            block_table = torch.arange(
                B * num_blocks, device=device, dtype=torch.int32
            ).view(B, num_blocks)
            cu_seqlens_q = torch.arange(
                0, (B + 1) * M, M, device=device, dtype=torch.int32
            )
            seqused_k = torch.full((B,), N, device=device, dtype=torch.int32)
            out = torch.empty_like(q_flat)
            sparse_mqa_sink(
                q_flat,
                kv_blocks,
                out,
                cu_seqlens_q,
                seqused_k,
                float(softmax_scale),
                topk_flat,
                block_table,
                attn_sink.float().contiguous(),
                tile_k=tile_k,
                block_h=block_h,
                block_d=block_d,
                score_d=score_d,
            )
            return out.view(B, M, H, D).to(out_dtype)
        except Exception as exc:
            if os.environ.get("ATOM_DSV4_AITER_SPARSE_ATTN_STRICT", "1") == "1":
                raise
            print(f"WARN: AITER DSv4 sparse_attn failed, falling back to Torch: {exc!r}")

    chunk_tokens = int(os.environ.get("ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS", "0") or "0")
    if B == 1 and chunk_tokens > 0 and M > chunk_tokens:
        return torch.cat(
            [
                sparse_attn(
                    q[:, start : start + chunk_tokens],
                    kv,
                    attn_sink,
                    topk_idxs[:, start : start + chunk_tokens],
                    softmax_scale,
                )
                for start in range(0, M, chunk_tokens)
            ],
            dim=1,
        )

    if B == 1 and M == 1:
        valid_1d = topk_idxs[0, 0] != -1
        if not bool(valid_1d.any()):
            return torch.zeros_like(q)
        idx_1d = topk_idxs[0, 0]
        if bool(valid_1d.all()):
            kv_f32 = kv[0].index_select(0, idx_1d.long()).float()
        else:
            kv_f32 = kv[0].index_select(0, idx_1d[valid_1d].long()).float()
        q_f32 = q[0, 0].float()
        scores = torch.matmul(q_f32, kv_f32.transpose(0, 1)) * float(softmax_scale)
        sink = attn_sink.float().view(H, 1)
        cmax = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        exp_scores = (scores - cmax).exp()
        denom = exp_scores.sum(dim=-1, keepdim=True) + (sink - cmax).exp()
        out = (exp_scores / denom.clamp(min=1e-30)).matmul(kv_f32)
        return out.view(1, 1, H, D).to(out_dtype)

    # ----- Gather KV per query position -----
"""
    if old not in source:
        raise SystemExit("FATAL: sparse_attn_v4.py did not match expected PR650 source")
    source = source.replace(old, new, 1)
    path.write_text(source)
    print(f"applied DSv4 sparse_attn_v4 AITER/decode/chunk patch: {path}")
else:
    print(f"DSv4 sparse_attn_v4 AITER/decode/chunk patch already present: {path}")
PYEOF

    # Local multi-request overlay for ROCm/ATOM#650. ATOM's scheduler passes
    # DSv4 a token-flat batch, but PR650 treats every request as cache slot 0
    # (`kv_cache[:1]` and matching compressor/indexer state). Reuse ATOM's
    # mamba-state slot allocator for DSv4, then split only sparse attention and
    # cache mutation per sequence while batching attention projections, mHC, and
    # MoE/FFN layer-by-layer. This fixes correctness for CONC>1 and avoids the
    # worst all-layers-per-request loop until upstream vectorizes the DSv4
    # sparse-attention/cache path.
    if ! grep -q 'def forward_batched' atom/models/deepseek_v4.py \
        && ! grep -q '_v4_get_seq_metadata' atom/models/deepseek_v4.py; then
    sed 's/^$/ /' <<'PATCH' | git apply --recount
diff --git a/atom/model_engine/llm_engine.py b/atom/model_engine/llm_engine.py
index 8de9532..ddde446 100644
--- a/atom/model_engine/llm_engine.py
+++ b/atom/model_engine/llm_engine.py
@@ -171,7 +171,16 @@ class InputOutputProcessor:
             self.num_speculative_tokens = (
                 self.config.speculative_config.num_speculative_tokens
             )
-        mamba_model_types = {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}
+        mamba_model_types = {
+            "qwen3_next",
+            "qwen3_5_text",
+            "qwen3_5_moe_text",
+            "deepseek_v4",
+            "deepseek_v4_pro",
+        }
-        if self.config.hf_config.model_type in mamba_model_types:
+        architectures = getattr(self.config.hf_config, "architectures", []) or []
+        if self.config.hf_config.model_type in mamba_model_types or any(
+            "DeepseekV4" in arch for arch in architectures
+        ):
             self.mamba_enabled = True

diff --git a/atom/model_engine/model_runner.py b/atom/model_engine/model_runner.py
index 72e9d84..598f2a5 100644
--- a/atom/model_engine/model_runner.py
+++ b/atom/model_engine/model_runner.py
@@ -659,6 +659,14 @@ class ModelRunner:
             )
         return False

+    def is_deepseek_v4(self) -> bool:
+        model_type = getattr(self.hf_text_config, "model_type", None)
+        architectures = getattr(self.hf_text_config, "architectures", []) or []
+        return model_type in (
+            "deepseek_v4",
+            "deepseek_v4_pro",
+        ) or any("DeepseekV4" in arch for arch in architectures)
+
     def is_qwen_next(self) -> bool:
         if not hasattr(self.hf_text_config, "model_type"):
             return False
@@ -1250,9 +1256,10 @@ class ModelRunner:

         # GDN recurrent state: deduct mamba tensor memory from pool budget
         mamba_per_slot = self._compute_mamba_per_slot_bytes()
+        needs_recurrent_slots = mamba_per_slot > 0 or self.is_deepseek_v4()
         slots_per_req = 1 + self.num_spec_tokens
         max_mamba_slots = (
-            config.max_num_seqs * slots_per_req if mamba_per_slot > 0 else 0
+            config.max_num_seqs * slots_per_req if needs_recurrent_slots else 0
         )
         mamba_tensor_bytes = max_mamba_slots * mamba_per_slot
         available_for_pool = available_for_kv - mamba_tensor_bytes
@@ -1270,7 +1277,7 @@ class ModelRunner:
         # Store for BlockManager and allocate_kv_cache
         config.mamba_equiv_per_req = mamba_equiv
         config.max_mamba_slots = max_mamba_slots
-        config.num_mamba_groups = config.max_num_seqs if mamba_per_slot > 0 else 0
+        config.num_mamba_groups = config.max_num_seqs if needs_recurrent_slots else 0
         self.max_mamba_slots = max_mamba_slots

         num_kvcache_blocks = available_for_pool // block_bytes
@@ -1309,7 +1316,7 @@ class ModelRunner:
         return {
             "num_kvcache_blocks": num_kvcache_blocks,
             "mamba_equiv_per_req": mamba_equiv,
-            "num_mamba_groups": config.max_num_seqs if mamba_per_slot > 0 else 0,
+            "num_mamba_groups": config.max_num_seqs if needs_recurrent_slots else 0,
         }

     def allocate_kv_cache(self, num_kvcache_blocks):
@@ -1782,6 +1789,13 @@ class ModelRunner:
             )
         attn_metadata, positions = self.attn_metadata_builder.build(batch=batch, bs=bs)
         context_bs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs
+        if self.is_deepseek_v4():
+            cache_slots = list(batch.mamba_state_slots)
+            if len(cache_slots) < context_bs:
+                cache_slots = list(range(context_bs))
+            attn_metadata.dsv4_cache_slots = torch.tensor(
+                cache_slots[:context_bs], dtype=torch.int64, device=self.device
+            )

         # graph_bs should be batch size (number of sequences), not token count
         graph_bs = num_input_tokens if is_prefill else bs
diff --git a/atom/models/deepseek_v4.py b/atom/models/deepseek_v4.py
index 46cf1b0..0d84c78 100644
--- a/atom/models/deepseek_v4.py
+++ b/atom/models/deepseek_v4.py
@@ -506,7 +506,9 @@ class Compressor(nn.Module):
         new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
         return new_tensor

-    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
+    def forward(
+        self, x: torch.Tensor, start_pos: int, cache_slot: int = 0
+    ) -> Optional[torch.Tensor]:
         """Compress KV for the input tokens. Writes into self.kv_cache when a
         compression block boundary is hit; otherwise just buffers state and returns None.

@@ -524,6 +526,7 @@ class Compressor(nn.Module):
         if x.dim() == 2:
             x = x.unsqueeze(0)  # [num_tokens, dim] → [1, num_tokens, dim]
         bsz, seqlen, _ = x.size()
+        slot = slice(cache_slot, cache_slot + bsz)
         ratio = self.compress_ratio
         overlap = self.overlap
         d = self.head_dim
@@ -545,16 +548,16 @@ class Compressor(nn.Module):
             # Save the last `ratio` overlap-slice tokens into kv_state for use
             # by the next decode call's overlap window.
             if overlap and cutoff >= ratio:
-                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
-                self.score_state[:bsz, :ratio] = (
+                self.kv_state[slot, :ratio] = kv[:, cutoff - ratio : cutoff]
+                self.score_state[slot, :ratio] = (
                     score[:, cutoff - ratio : cutoff] + self.ape
                 )
             # Save the trailing partial block (remainder tokens) into kv_state.
             if remainder > 0:
-                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
+                kv, self.kv_state[slot, offset : offset + remainder] = kv.split(
                     [cutoff, remainder], dim=1
                 )
-                self.score_state[:bsz, offset : offset + remainder] = (
+                self.score_state[slot, offset : offset + remainder] = (
                     score[:, cutoff:] + self.ape[:remainder]
                 )
                 score = score[:, :cutoff]
@@ -570,20 +573,20 @@ class Compressor(nn.Module):
             should_compress = (start_pos + 1) % self.compress_ratio == 0
             score = score + self.ape[start_pos % ratio]
             if overlap:
-                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
-                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
+                self.kv_state[slot, ratio + start_pos % ratio] = kv.squeeze(1)
+                self.score_state[slot, ratio + start_pos % ratio] = score.squeeze(1)
                 if should_compress:
                     kv_state = torch.cat(
                         [
-                            self.kv_state[:bsz, :ratio, :d],
-                            self.kv_state[:bsz, ratio:, d:],
+                            self.kv_state[slot, :ratio, :d],
+                            self.kv_state[slot, ratio:, d:],
                         ],
                         dim=1,
                     )
                     score_state = torch.cat(
                         [
-                            self.score_state[:bsz, :ratio, :d],
-                            self.score_state[:bsz, ratio:, d:],
+                            self.score_state[slot, :ratio, :d],
+                            self.score_state[slot, ratio:, d:],
                         ],
                         dim=1,
                     )
@@ -591,14 +594,14 @@ class Compressor(nn.Module):
                         dim=1, keepdim=True
                     )
                     # Roll: the just-completed window becomes the next overlap window.
-                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
-                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
+                    self.kv_state[slot, :ratio] = self.kv_state[slot, ratio:]
+                    self.score_state[slot, :ratio] = self.score_state[slot, ratio:]
             else:
-                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
-                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
+                self.kv_state[slot, start_pos % ratio] = kv.squeeze(1)
+                self.score_state[slot, start_pos % ratio] = score.squeeze(1)
                 if should_compress:
                     kv = (
-                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
+                        self.kv_state[slot] * self.score_state[slot].softmax(dim=1)
                     ).sum(dim=1, keepdim=True)

         if not should_compress:
@@ -622,9 +625,9 @@ class Compressor(nn.Module):
             act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

         if start_pos == 0:
-            self.kv_cache[:bsz, : seqlen // ratio] = kv
+            self.kv_cache[slot, : seqlen // ratio] = kv
         else:
-            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
+            self.kv_cache[slot, start_pos // ratio] = kv.squeeze(1)
         return kv


@@ -696,6 +699,7 @@ class Indexer(nn.Module):
         qr: torch.Tensor,
         start_pos: int,
         offset: int,
+        cache_slot: int = 0,
     ) -> torch.Tensor:
         """Compute sparse top-k indices over the indexer's compressed KV cache.

@@ -715,6 +719,7 @@ class Indexer(nn.Module):
         ratio = self.compress_ratio
         rd = self.rope_head_dim
         end_pos = start_pos + seqlen
+        slot = slice(cache_slot, cache_slot + 1)

         # Lazy plumb the indexer's kv_cache + freqs_cis into its compressor.
         if self.compressor.kv_cache is None:
@@ -729,7 +734,7 @@ class Indexer(nn.Module):
         fp4_act_quant_inplace(q, _FP4_BLOCK_SIZE)

         # ----- Indexer KV (Compressor takes 2D, mutates kv_cache) -----
-        self.compressor(x, start_pos)
+        self.compressor(x, start_pos, cache_slot)
         # weights_proj is ATOM Linear → 2D input; restore B=1 dim for einsum.
         weights = (
             self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
@@ -737,7 +742,7 @@ class Indexer(nn.Module):

         # ----- Index score -----
         index_score = torch.einsum(
-            "bshd,btd->bsht", q, self.kv_cache[:1, : end_pos // ratio]
+            "bshd,btd->bsht", q, self.kv_cache[slot, : end_pos // ratio]
         )
         index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

@@ -959,7 +964,9 @@ class DeepseekV4Attention(nn.Module):

         self.wo_a.quant_type = _QT.No

-    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
+    def forward(
+        self, x: torch.Tensor, start_pos: int, cache_slot: int = 0
+    ) -> torch.Tensor:
         """Compute attention for `x` at absolute position `start_pos`.

         Args:
@@ -978,6 +985,7 @@ class DeepseekV4Attention(nn.Module):
         win = self.window_size
         ratio = self.compress_ratio
         rd = self.rope_head_dim
+        slot = slice(cache_slot, cache_slot + 1)

         # First-call plumbing: hand the (compressed-half) KV cache + freqs_cis
         # to the compressor / indexer.
@@ -992,14 +1000,14 @@ class DeepseekV4Attention(nn.Module):
         # with garbage. Real prefill only overwrites a few slots, leaving
         # stale warmup data that poisons decode attention.
         if start_pos == 0:
-            self.kv_cache.zero_()
+            self.kv_cache[slot].zero_()
             if self.compress_ratio:
-                self.compressor.kv_state.zero_()
-                self.compressor.score_state.fill_(float("-inf"))
+                self.compressor.kv_state[slot].zero_()
+                self.compressor.score_state[slot].fill_(float("-inf"))
                 if self.indexer is not None:
-                    self.indexer.kv_cache.zero_()
-                    self.indexer.compressor.kv_state.zero_()
-                    self.indexer.compressor.score_state.fill_(float("-inf"))
+                    self.indexer.kv_cache[slot].zero_()
+                    self.indexer.compressor.kv_state[slot].zero_()
+                    self.indexer.compressor.score_state[slot].fill_(float("-inf"))

         # ----- Q: low-rank projection + per-head RMSNorm + partial RoPE -----
         # ATOM TP linears require 2D inputs; subsequent ops (RoPE, sparse_attn)
@@ -1023,7 +1031,7 @@ class DeepseekV4Attention(nn.Module):
         if self.compress_ratio:
             offset = kv.size(1) if start_pos == 0 else win
             if self.indexer is not None:
-                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
+                compress_topk_idxs = self.indexer(x, qr, start_pos, offset, cache_slot)
             else:
                 compress_topk_idxs = _get_compress_topk_idxs(
                     ratio, 1, seqlen, start_pos, offset, device=x.device
@@ -1037,42 +1045,136 @@ class DeepseekV4Attention(nn.Module):
         # implicit B=1.) -----
         if start_pos == 0:
             if seqlen <= win:
-                self.kv_cache[:1, :seqlen] = kv
+                self.kv_cache[slot, :seqlen] = kv
             else:
                 cutoff = seqlen % win
                 (
-                    self.kv_cache[:1, cutoff:win],
-                    self.kv_cache[:1, :cutoff],
+                    self.kv_cache[slot, cutoff:win],
+                    self.kv_cache[slot, :cutoff],
                 ) = kv[
                     :, -win:
                 ].split([win - cutoff, cutoff], dim=1)
             if self.compress_ratio:
-                if (kv_compress := self.compressor(x, start_pos)) is not None:
+                if (kv_compress := self.compressor(x, start_pos, cache_slot)) is not None:
                     kv = torch.cat([kv, kv_compress], dim=1)
             o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
         else:
-            self.kv_cache[:1, start_pos % win] = kv.squeeze(1)
+            self.kv_cache[slot, start_pos % win] = kv.squeeze(1)
             if self.compress_ratio:
-                self.compressor(x, start_pos)
+                self.compressor(x, start_pos, cache_slot)
             o = sparse_attn(
                 q,
-                self.kv_cache[:1],
+                self.kv_cache[slot],
                 self.attn_sink,
                 topk_idxs,
                 self.softmax_scale,
             )

         # Inverse RoPE on output's rope dims to remove absolute-position contribution
         # carried in by the value-side RoPE of the KV entries.
         _apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

         # ----- Grouped output LoRA -----
         # o: [1, S, H, D] → drop B; reshape into groups for the einsum.
         o = o.squeeze(0).view(seqlen, self.n_local_groups, -1)  # [S, g, H/g * D]
         wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
         o = torch.einsum("sgd,grd->sgr", o, wo_a)  # [S, g, o_lora_rank]
         x = self.wo_b(o.flatten(1))  # 2D [S, dim]
         return x

+    def forward_batched(
+        self, x: torch.Tensor, seq_meta: list[tuple[int, int, int, int]]
+    ) -> torch.Tensor:
+        assert (
+            x.dim() == 2
+        ), f"DeepseekV4Attention expects 2D [num_tokens, dim], got {x.shape}"
+        total_tokens = x.size(0)
+        win = self.window_size
+        ratio = self.compress_ratio
+        rd = self.rope_head_dim
+
+        if self.compress_ratio and self.compressor.kv_cache is None:
+            self.compressor.kv_cache = self.kv_cache[:, win:]
+            self.compressor.freqs_cis = self.freqs_cis
+            if self.indexer is not None:
+                self.indexer.freqs_cis = self.freqs_cis
+
+        qr_all = self.q_norm(self.wq_a(x))
+        q_all = self.wq_b(qr_all).view(total_tokens, self.n_local_heads, self.head_dim)
+        q_all = q_all * torch.rsqrt(q_all.square().mean(-1, keepdim=True) + self.eps)
+        kv_all = self.kv_norm(self.wkv(x)).view(total_tokens, self.head_dim)
+
+        outputs = []
+        for start, end, start_pos, cache_slot in seq_meta:
+            seqlen = end - start
+            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
+            slot = slice(cache_slot, cache_slot + 1)
+
+            if start_pos == 0:
+                self.kv_cache[slot].zero_()
+                if self.compress_ratio:
+                    self.compressor.kv_state[slot].zero_()
+                    self.compressor.score_state[slot].fill_(float("-inf"))
+                    if self.indexer is not None:
+                        self.indexer.kv_cache[slot].zero_()
+                        self.indexer.compressor.kv_state[slot].zero_()
+                        self.indexer.compressor.score_state[slot].fill_(float("-inf"))
+
+            q = q_all[start:end].unsqueeze(0)
+            _apply_rotary_emb(q[..., -rd:], freqs_cis)
+            kv = kv_all[start:end].unsqueeze(0)
+            _apply_rotary_emb(kv[..., -rd:], freqs_cis)
+            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)
+
+            topk_idxs = _get_window_topk_idxs(
+                win, 1, seqlen, start_pos, device=x.device
+            )
+            if self.compress_ratio:
+                offset = kv.size(1) if start_pos == 0 else win
+                if self.indexer is not None:
+                    compress_topk_idxs = self.indexer(
+                        x[start:end], qr_all[start:end], start_pos, offset, cache_slot
+                    )
+                else:
+                    compress_topk_idxs = _get_compress_topk_idxs(
+                        ratio, 1, seqlen, start_pos, offset, device=x.device
+                    )
+                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
+            topk_idxs = topk_idxs.int()
+
+            if start_pos == 0:
+                if seqlen <= win:
+                    self.kv_cache[slot, :seqlen] = kv
+                else:
+                    cutoff = seqlen % win
+                    (
+                        self.kv_cache[slot, cutoff:win],
+                        self.kv_cache[slot, :cutoff],
+                    ) = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
+                if self.compress_ratio:
+                    kv_compress = self.compressor(x[start:end], start_pos, cache_slot)
+                    if kv_compress is not None:
+                        kv = torch.cat([kv, kv_compress], dim=1)
+                o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
+            else:
+                self.kv_cache[slot, start_pos % win] = kv.squeeze(1)
+                if self.compress_ratio:
+                    self.compressor(x[start:end], start_pos, cache_slot)
+                o = sparse_attn(
+                    q,
+                    self.kv_cache[slot],
+                    self.attn_sink,
+                    topk_idxs,
+                    self.softmax_scale,
+                )
+
+            _apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
+            outputs.append(o.squeeze(0))
+
+        o = torch.cat(outputs, dim=0).view(total_tokens, self.n_local_groups, -1)
+        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
+        o = torch.einsum("sgd,grd->sgr", o, wo_a)
+        return self.wo_b(o.flatten(1))
+

@@ -1599,6 +1701,7 @@ class Block(nn.Module):
         x: torch.Tensor,
         start_pos: int,
         input_ids: Optional[torch.Tensor],
+        cache_slot: int = 0,
     ) -> torch.Tensor:
         # ----- Attention sub-layer with mHC mixing -----
         residual = x
@@ -1606,7 +1709,7 @@ class Block(nn.Module):
             x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
         )
         x = self.attn_norm(x)
-        x = self.attn(x, start_pos)
+        x = self.attn(x, start_pos, cache_slot)
         x = self.hc_post(x, residual, post, comb)

         # ----- FFN sub-layer with mHC mixing -----
@@ -1821,11 +1924,81 @@ class DeepseekV4Model(nn.Module):
         self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
         self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

+    def _forward_one(
+        self,
+        input_ids: torch.Tensor,
+        start_pos: int,
+        cache_slot: int,
+    ) -> torch.Tensor:
+        h = self.embed(input_ids)  # [num_tokens, dim]
+        # Expand to hc_mult copies for Hyper-Connections: [num_tokens, hc, dim]
+        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
+
+        for layer in self.layers:
+            h = layer(h, start_pos, input_ids, cache_slot)
+
+        logits = self.head(
+            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
+        )
+        return logits
+
+    def _head_tokens(self, h: torch.Tensor) -> torch.Tensor:
+        x = self.head.hc_head(
+            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
+        )
+        return F.linear(self.norm(x).float(), self.head.weight)
+
+    def _forward_layerwise_batched(
+        self,
+        input_ids: torch.Tensor,
+        positions: torch.Tensor,
+        cu_seqlens_q: torch.Tensor,
+        cache_slots: torch.Tensor,
+        num_seqs: int,
+    ) -> torch.Tensor:
+        seq_meta: list[tuple[int, int, int, int]] = []
+        last_indices: list[int] = []
+        for seq_idx in range(num_seqs):
+            start = int(cu_seqlens_q[seq_idx].item())
+            end = int(cu_seqlens_q[seq_idx + 1].item())
+            if end <= start:
+                continue
+            seq_start = int(positions[start].item())
+            cache_slot = int(cache_slots[seq_idx].item())
+            seq_meta.append((start, end, seq_start, cache_slot))
+            last_indices.append(end - 1)
+        if not seq_meta:
+            return self._forward_one(input_ids[:1], 0, 0)
+
+        h = self.embed(input_ids)
+        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
+
+        for layer in self.layers:
+            residual = h
+            x, post, comb = layer.hc_pre(
+                h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
+            )
+            x = layer.attn_norm(x)
+            x = layer.attn.forward_batched(x, seq_meta)
+            h = layer.hc_post(x, residual, post, comb)
+
+            residual = h
+            x, post, comb = layer.hc_pre(
+                h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
+            )
+            x = layer.ffn_norm(x)
+            x = layer.ffn(x, input_ids)
+            h = layer.hc_post(x, residual, post, comb)
+
+        last_indices_t = torch.tensor(last_indices, device=h.device, dtype=torch.long)
+        return self._head_tokens(h.index_select(0, last_indices_t))
+
     @torch.inference_mode()
     def forward(
         self,
         input_ids: torch.Tensor,
         start_pos: int = 0,
+        positions: Optional[torch.Tensor] = None,
         **model_kwargs: dict,
     ) -> torch.Tensor:
         """Forward.
@@ -1844,17 +2017,42 @@ class DeepseekV4Model(nn.Module):
                 input_ids.size(0) == 1
             ), "B>1 batched input_ids needs attn_metadata; not supported yet"
             input_ids = input_ids.flatten()
-        h = self.embed(input_ids)  # [num_tokens, dim]
-        # Expand to hc_mult copies for Hyper-Connections: [num_tokens, hc, dim]
-        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
+        if positions is None:
+            positions = torch.arange(
+                start_pos,
+                start_pos + input_ids.numel(),
+                device=input_ids.device,
+                dtype=torch.int64,
+            )
+        else:
+            positions = positions.flatten()

-        for layer in self.layers:
-            h = layer(h, start_pos, input_ids)
+        attn_metadata = None
+        context = None
+        try:
+            from atom.utils.forward_context import get_forward_context

-        logits = self.head(
-            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
-        )
-        return logits
+            forward_context = get_forward_context()
+            attn_metadata = forward_context.attn_metadata
+            context = forward_context.context
+        except Exception:
+            pass
+
+        cu_seqlens_q = getattr(attn_metadata, "cu_seqlens_q", None)
+        if cu_seqlens_q is None or context is None or context.batch_size <= 1:
+            seq_start = int(positions[0].item()) if positions.numel() else int(start_pos)
+            cache_slots = getattr(attn_metadata, "dsv4_cache_slots", None)
+            cache_slot = int(cache_slots[0].item()) if cache_slots is not None else 0
+            return self._forward_one(input_ids, seq_start, cache_slot)
+
+        num_seqs = int(context.batch_size)
+        cache_slots = getattr(attn_metadata, "dsv4_cache_slots", None)
+        if cache_slots is None or cache_slots.numel() < num_seqs:
+            cache_slots = torch.arange(num_seqs, device=input_ids.device, dtype=torch.int64)
+
+        return self._forward_layerwise_batched(
+            input_ids, positions, cu_seqlens_q, cache_slots, num_seqs
+        )


 class DeepseekV4ForCausalLM(nn.Module):
@@ -1918,6 +2116,9 @@ class DeepseekV4ForCausalLM(nn.Module):
         # config lacks `quantization_config` (e.g. dummy / toy validation),
         # this still works — base spec is QuantType.No.
         self.args.quant_config = make_v4_quant_config(self.hf_config)
+        self.args.max_batch_size = max(
+            self.args.max_batch_size, int(getattr(config, "max_num_seqs", 1))
+        )
         self.model = DeepseekV4Model(args=self.args)

     def forward(
@@ -1929,7 +2130,12 @@ class DeepseekV4ForCausalLM(nn.Module):
         **model_kwargs: dict,
     ) -> torch.Tensor:
         start_pos = int(positions[0].item()) if positions is not None else 0
-        return self.model(input_ids=input_ids, start_pos=start_pos, **model_kwargs)
+        return self.model(
+            input_ids=input_ids,
+            start_pos=start_pos,
+            positions=positions,
+            **model_kwargs,
+        )

     def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
         # In V4, the LM head is fused into DeepseekV4Model.forward (it consumes
PATCH
    else
        echo "ATOM DSv4 multi-request overlay already present: atom/models/deepseek_v4.py"
    fi

    # Replace the Indexer's [S,H,T] Torch scorer/topk with the forked AITER
    # scorer. For 1k1k, the op takes its dense causal fast path and skips
    # scoring entirely because every committed compressed entry is selected.
    python3 - <<'PYEOF'
from pathlib import Path

path = Path("atom/models/deepseek_v4.py")
source = path.read_text()
marker = "ATOM_DSV4_AITER_INDEXER"
if marker not in source:
    old = """        # ----- Index score -----
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[slot, : end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        # ----- Top-k selection over compressed positions -----
        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = (
                topk_idxs
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs
"""
    new = """        # ----- Index score / top-k selection over compressed positions -----
        n_committed = end_pos // ratio
        if n_committed <= 0:
            return torch.empty(1, seqlen, 0, dtype=torch.int32, device=x.device)

        if os.environ.get("ATOM_DSV4_AITER_INDEXER", "1") == "1" and q.is_cuda:
            try:
                from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer_topk

                positions = torch.arange(
                    start_pos, end_pos, device=x.device, dtype=torch.int64
                )
                topk_idxs = dsv4_indexer_topk(
                    q.squeeze(0),
                    self.kv_cache[slot, :n_committed].squeeze(0),
                    weights.squeeze(0),
                    positions,
                    self.index_topk,
                    offset,
                    ratio=ratio,
                )
                return topk_idxs.unsqueeze(0)
            except Exception as exc:
                if os.environ.get("ATOM_DSV4_AITER_INDEXER_STRICT", "1") == "1":
                    raise
                print(f"WARN: AITER DSv4 Indexer failed, falling back to Torch: {exc!r}")

        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[slot, :n_committed]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, n_committed), dim=-1)[1]
        if start_pos == 0:
            mask = (
                topk_idxs
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs
"""
    if old not in source:
        raise SystemExit("FATAL: deepseek_v4.py did not match expected Indexer fallback")
    source = source.replace(old, new, 1)
    path.write_text(source)
    print(f"applied DSv4 AITER Indexer patch: {path}")
else:
    print(f"DSv4 AITER Indexer patch already present: {path}")
PYEOF

    # --no-deps: don't churn the image's pinned ROCm/torch/triton/aiter.
    # --force-reinstall: replace the wheel-installed atom with the editable copy.
    pip install --no-deps --force-reinstall -e .
)

# Install triton_kernels. The release atom0.1.2.post image cleans up
# /triton-test/ from the build stage, so it's typically absent. Fall back
# to ROCm/triton's RI3.5.x branch — NOT triton-lang/triton upstream:
#
#   * Upstream triton-lang/triton refactored the matmul_ogs module into
#     matmul.py (and removed routing.py). PR #650's fused_moe_triton.py
#     imports `from triton_kernels.matmul_ogs import matmul_ogs,
#     PrecisionConfig` and `from triton_kernels.routing import routing`,
#     which only resolve against the ROCm fork's release-internal branch.
#   * ROCm/triton RI3.5.x at e491726 has matmul_ogs.py (with PrecisionConfig
#     and matmul_ogs), routing.py, CDNA4MXScaleLayout in layout.py (the
#     class PR #650 imports), and target_info.py that imports only is_hip /
#     is_hip_cdna3 / is_hip_cdna4 — no is_hip_gfx1250, which the image's
#     bundled triton would reject.
#
# triton_kernels is a self-contained subpackage (pyproject deps: numpy,
# pytest); installing it does not perturb the image's triton itself.
# Bump only after AMD ships a newer ATOM image whose bundled triton
# exports is_hip_gfx1250, at which point we can move to a newer RI branch.
TRITON_KERNELS_SHA="e49172654d55f460c6fc24d77a3ea8a286bcaee8"
# --force-reinstall mirrors the atom install above: triton_kernels also ships
# as a wheel in the image, and without --force-reinstall pip can short-circuit
# the editable switch when name/version match, leaving the wheel build active.
if [ -d /triton-test/python/triton_kernels/ ]; then
    pip install --no-deps --force-reinstall -e /triton-test/python/triton_kernels/
else
    TRITON_DIR="/tmp/rocm-triton"
    if [ ! -d "$TRITON_DIR/.git" ]; then
        git clone --filter=blob:none https://github.com/ROCm/triton.git "$TRITON_DIR"
    fi
    (
        cd "$TRITON_DIR"
        git fetch --depth=1 origin "$TRITON_KERNELS_SHA" 2>/dev/null \
            || git fetch --depth=1 origin RI3.5.x
        git checkout --force "$TRITON_KERNELS_SHA"
        pip install --no-deps --force-reinstall -e python/triton_kernels/
    )
fi

# Preflight version checks. The chosen base image
# (atom0.1.2.post, rebuilt 2026-04-23) was tagged after ATOM pinned
# transformers==5.2.0 (commit 67d6cb61, 2026-03-13), so transformers compat
# is expected; we still assert it explicitly to fail fast with a clear
# message rather than timing out wait_for_server_ready on a confusing
# import error inside the server log. The two non-trivial deps the PR
# introduces are transformers' deepseek_v3 config class (mapped from
# deepseek_v4 in atom/config.py) and triton_kernels.CDNA4MXScaleLayout
# (renamed from GFX950MXScaleLayout in fused_moe_triton.py).
python3 - <<'PYEOF'
import importlib, os, sys
import atom

# Verify the editable install actually took effect — Python could still be
# importing the wheel-installed atom if pip's --force-reinstall silently no-op'd
# (e.g., the wheel and the editable copy share a setup.py path mismatch).
atom_path = os.path.abspath(atom.__file__)
expected = os.path.abspath(os.environ["ATOM_PR_DIR"])
print(f"atom imported from: {atom_path}")
if expected not in atom_path:
    sys.exit(f"FATAL: atom is importing from {atom_path}, not from PR checkout {expected}. "
             f"The pip --force-reinstall -e . did not take effect.")

import transformers
print(f"transformers version: {transformers.__version__}")

# Use CONFIG_MAPPING directly: AutoConfig.for_model() returns an instance
# (transformers 5.2.0 source: `return config_class(*args, **kwargs)`), not a
# class, so `.__name__` would AttributeError. CONFIG_MAPPING maps model_type
# to the config class directly and is unambiguous.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "deepseek_v3" not in CONFIG_MAPPING:
    sys.exit(f"FATAL: transformers in this image cannot resolve deepseek_v3 model_type. "
             f"ATOM PR #650 maps deepseek_v4 -> deepseek_v3 in _CONFIG_REGISTRY and needs "
             f"transformers to know the v3 schema. Available types: "
             f"{sorted(k for k in CONFIG_MAPPING if 'deepseek' in k)}")
print(f"deepseek_v3 config class: {CONFIG_MAPPING['deepseek_v3'].__name__}")

try:
    layout_mod = importlib.import_module("triton_kernels.tensor_details.layout")
    if not hasattr(layout_mod, "CDNA4MXScaleLayout"):
        avail = [n for n in dir(layout_mod) if "Layout" in n]
        sys.exit(f"FATAL: triton_kernels.tensor_details.layout has no CDNA4MXScaleLayout. "
                 f"PR #650's fused_moe_triton.py change renamed GFX950MXScaleLayout -> "
                 f"CDNA4MXScaleLayout, but this image's triton_kernels still uses the old "
                 f"name. Available Layout classes: {avail}")
    print("triton_kernels.CDNA4MXScaleLayout: present")
except ModuleNotFoundError as e:
    sys.exit(f"FATAL: triton_kernels not importable. PR #650's MoE path needs it. Error: {e}")
PYEOF

# DSv4-Pro's native max_position_embeddings is 1,048,576 (1M tokens), so we
# can't leave --max-model-len blank for 1k1k the way the dsr1-atom scripts
# do — ATOM would allocate KV cache for 1M context and OOM during warmup
# (~240 GiB consumed before the dummy forward, then sparse_attn's
# torch.where wants another ~36 GiB and there isn't 36 GiB free). DSR1's
# native context is only 128k, which is why the same blank pattern works
# there. Set 1k1k explicitly; 8k1k retains the existing 10240 cap that's
# already running successfully.
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    MAX_MODEL_LEN_VALUE=2304
else
    MAX_MODEL_LEN_VALUE=10240
fi
CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN_VALUE="$EVAL_MAX_MODEL_LEN"
    CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

BLOCK_SIZE=${BLOCK_SIZE:-16}
export ATOM_DSV4_AITER_SPARSE_ATTN=${ATOM_DSV4_AITER_SPARSE_ATTN:-1}
export ATOM_DSV4_AITER_INDEXER=${ATOM_DSV4_AITER_INDEXER:-1}
export ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS=${ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS:-256}
export ATOM_DSV4_AITER_SPARSE_ATTN_TILE_K=${ATOM_DSV4_AITER_SPARSE_ATTN_TILE_K:-64}
export ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_H=${ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_H:-4}
export ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_D=${ATOM_DSV4_AITER_SPARSE_ATTN_BLOCK_D:-128}
export ATOM_DSV4_AITER_SPARSE_ATTN_SCORE_D=${ATOM_DSV4_AITER_SPARSE_ATTN_SCORE_D:-64}
# --enforce-eager is required: ROCm/ATOM#650 (PR1 skeleton) has no CUDAGraph
# support yet (deferred to a follow-up PR). max-num-seqs is sized to the
# client concurrency with a floor at 4 — the ATOM default (512) makes the
# KV/GDN-mamba allocator overshoot the GPU budget ("GDN mamba tensor
# exceeds available KV budget"), and using 1 hangs warmup at 0% GPU. 4
# is the minimum we've seen complete warmup successfully (also the PR's
# offline repro value). The local PR650 overlay above maps each request to a
# persistent DSv4 cache slot; without it, deepseek_v4.py's `kv_cache[:1]`
# writes corrupt non-slot-0 lanes at CONC>1.
MAX_NUM_SEQS=$(( CONC < 4 ? 4 : CONC ))
# Allow prefill batching again. The layer-wise DSv4 overlay splits attention by
# sequence before sparse_attn, so two 8k-ish prompts no longer become one giant
# sparse-attention problem, while MoE/FFN still sees the larger token batch.
DEFAULT_MAX_NUM_BATCHED_TOKENS=$(( MAX_MODEL_LEN_VALUE > 16384 ? MAX_MODEL_LEN_VALUE : 16384 ))
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}
setup_atom_profile_args
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    "${ATOM_PROFILE_ARGS[@]}" \
    --trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
