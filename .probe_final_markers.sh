#!/usr/bin/env bash
# Final marker sweep: the precise identifiers each vendor hunk introduces,
# checked against the target tree. Anything ABSENT in target is a real addition
# we must carry; anything PRESENT means upstream already landed it in the 465
# commits between ref and target and the hunk should be dropped, not forced.
set -u
D=/home/jiacao/3way-20260812-2214
R="$D/ref/usr/local/lib/python3.12/dist-packages/vllm"
V="$D/vendor/src/vllm/vllm"
T="$D/target/usr/local/lib/python3.12/dist-packages/vllm"

chk() { # $1=marker $2=feature-label
    local m="$1" lbl="$2"
    local r v t
    r=$(grep -rl -- "$m" "$R" --include='*.py' 2>/dev/null | wc -l)
    v=$(grep -rl -- "$m" "$V" --include='*.py' 2>/dev/null | wc -l)
    t=$(grep -rl -- "$m" "$T" --include='*.py' 2>/dev/null | wc -l)
    local verdict
    if   [ "$v" -eq 0 ];                  then verdict="n/a"
    elif [ "$t" -gt 0 ] && [ "$r" -gt 0 ]; then verdict="pre-existing"
    elif [ "$t" -gt 0 ];                   then verdict="UPSTREAM-LANDED"
    else                                        verdict="MUST-ADD"
    fi
    printf "  r=%-3s v=%-3s t=%-3s  %-16s %-46s %s\n" "$r" "$v" "$t" "$verdict" "$m" "$lbl"
}

echo "=== MegaMoE ==="
chk "flydsl_mega_moe"                  "kernel-backend enum value"
chk "use_mega_moe"                     "layer flag"
chk "mega_moe_experts"                 "new module"
chk "mega_moe_runtime"                 "new module"
chk "finalize_mega_moe_layers"         "post-load hook"
chk "make_deepseek_v4_mega_expert_params_mapping" "weight mapping"

echo "=== Gluon sparse attention ==="
chk "VLLM_ROCM_DSV4_SPARSE_GLUON"      "env knob"
chk "pa_decode_sparse"                 "aiter entry point"

echo "=== FSE / heterogeneous shared expert ==="
chk "shared_expert_id"                 "fused_moe kwarg"
chk "fused_moe_supports_heterogeneous_shared_expert" "capability probe"

echo "=== tuned GEMM (DSv4 attention projections) ==="
chk "from aiter.tuned_gemm import tgemm" "tgemm import"

echo "=== misc plumbing the above depend on ==="
chk "cudagraph_warmup_context"         "platform hook"
chk "gemm_a8w8_blockscale_bpreshuffle" "bpreshuffle GEMM op"
chk "aiter_per1x128_quant"             "per-1x128 quant"
chk "transpose_scale"                  "quant kwarg"
chk "process_weights_after_loading()"  "model-level post-load hook"
chk "_fused_wqa_wkv_gemm"              "attention refactor hook"
chk "swiglu_limit"                     "swiglu limit passthrough"
