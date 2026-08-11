#!/usr/bin/env bash
# Apply the Kimi-K3 runtime optimisation patches inside the container.
#
# WHY THESE ARE RUNTIME PATCHES AND NOT COMMITS
# Our vLLM fork branch is rebased onto cb8104839c, the same nightly the prepatched
# aigmkt/kimi-k3-agentx image is built from -- so we share that image's BASE. But the optimisations
# are not in that base: they are ports of four upstream PRs onto it, applied at run time by the
# single-node recipe. Confirmed by looking: `git log --grep` finds none of #51011, #51040, #51088 or
# #51171 in cb8104839c. Rebasing gets us the same starting point, not the optimisations, so they come
# along separately.
#
# They are safe to stack on our fork. Their target files and ours do not intersect at all:
#   theirs  rocm_aiter_mla.py, triton_mla.py, attn_utils.py, model_runner.py,
#           models/kimi_k3/nvidia/mla.py, aiter mla_gluon.py
#   ours    moriio_*, mooncake/store/worker.py, sched/scheduler.py, kv_cache_utils.py,
#           kv_cache_interface.py, rejection_sampler_utils.py, gdn/kda
# So the two sets are orthogonal and neither can silently reinterpret the other's hunks.
#
# ORDER MATTERS. The reference recipe applies #51011 only after #51088 and vllm_dspark_ps_enable.py,
# because the later patches expect the persistent-scheduling metadata the earlier ones introduce.
#
# Every step is opt-in via K3_OPT_PATCHES=1 and individually skippable, because the point of carrying
# these separately is to be able to A/B them one at a time against the numbers we already have on the
# un-patched base (input 3,872 tok/s at conc 16, TTFT p50 15.3 s, acceptance 4.4-5.0 on GSM8K).
set +e

k3_opt_patches_apply() {
    if [[ "${K3_OPT_PATCHES:-0}" != "1" ]]; then
        echo "[k3-opt] K3_OPT_PATCHES != 1, skipping the optimisation patch set"
        return 0
    fi
    local here vllm_root aiter_root
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"
    vllm_root="$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)"
    if [[ -z "$vllm_root" || ! -d "$vllm_root/vllm" ]]; then
        echo "[k3-opt] ERROR: cannot locate the installed vllm package root" >&2
        return 1
    fi
    aiter_root="$(python3 -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))' 2>/dev/null)"
    echo "[k3-opt] vllm_root=$vllm_root aiter_root=${aiter_root:-<none>}"

    local rc=0

    # 1. The K3 base edits (aiter gluon MLA, K3 model MLA, aiter MLA backend, attn_utils, model_runner).
    #    Skipped on an image that already carries them -- that is what "prepatched" means.
    if [[ "${K3_BASE_PATCHES_PREAPPLIED:-0}" == "1" ]]; then
        echo "[k3-opt] K3_BASE_PATCHES_PREAPPLIED=1: apply_k3_container_patches.sh skipped"
    elif [[ "${K3_OPT_SKIP_BASE:-0}" == "1" ]]; then
        echo "[k3-opt] K3_OPT_SKIP_BASE=1: base patches skipped"
    else
        k3_opt_step "base K3 container patches" bash "$here/apply_k3_container_patches.sh" || rc=1
    fi

    # 2. #51088 force persistent scheduling, on the aiter MLA backend.
    k3_opt_diff "#51088 force-ps" "$here/vllm_pr51088_force_ps.diff" "$vllm_root" || rc=1

    # 3. DSpark persistent-scheduling enablement. Must follow #51088.
    k3_opt_step "dspark ps enable" python3 "$here/vllm_dspark_ps_enable.py" || rc=1

    # 4. #51011 reorder qlen. The reference applies this after #51088 + ps-enable.
    k3_opt_step "#51011 reorder-qlen" python3 "$here/vllm_pr51011_reorder_qlen.py" || rc=1

    # 5. #51040 fp8 prefill pad16.
    k3_opt_step "#51040 fp8-prefill-pad16" python3 "$here/vllm_pr51040_fp8_prefill_pad16.py" || rc=1

    # 6. #51171 triton MLA cudagraph. Only the triton_mla.py hunk is in this diff.
    k3_opt_diff "#51171 triton-mla-cg" "$here/vllm_pr51171_triton_mla_cg.diff" "$vllm_root" || rc=1

    # 7. qlen padding, so the ASM decode kernel accepts K3's 12 heads/rank at TP8.
    k3_opt_step "qlen pad" python3 "$here/vllm_qlen_pad.py" || rc=1

    if [[ "$rc" != "0" ]]; then
        # Do not continue silently. A half-applied optimisation set is a configuration nobody has
        # measured, and attributing a later number to it would be wrong.
        echo "[k3-opt] ERROR: at least one patch failed; refusing to run a half-patched engine" >&2
        return 1
    fi
    echo "[k3-opt] all optimisation patches applied"
}

# A named step, with the skip knob derived from the label so each one can be dropped for an A/B.
k3_opt_step() {
    local label="$1"; shift
    local var="K3_OPT_SKIP_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_' | tr 'a-z' 'A-Z')"
    if [[ "${!var:-0}" == "1" ]]; then
        echo "[k3-opt] $label: skipped ($var=1)"
        return 0
    fi
    if [[ ! -e "$1" ]] && [[ ! -e "$2" ]]; then
        echo "[k3-opt] $label: patch file missing" >&2
        return 1
    fi
    echo "[k3-opt] $label: applying"
    if "$@"; then
        echo "[k3-opt] $label: ok"
        return 0
    fi
    echo "[k3-opt] $label: FAILED" >&2
    return 1
}

# `patch --forward --batch` treats an already-applied hunk as a non-fatal skip, which is what makes
# this idempotent across restarts within one allocation.
k3_opt_diff() {
    local label="$1" diff="$2" root="$3"
    local var="K3_OPT_SKIP_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_' | tr 'a-z' 'A-Z')"
    if [[ "${!var:-0}" == "1" ]]; then
        echo "[k3-opt] $label: skipped ($var=1)"
        return 0
    fi
    if [[ ! -f "$diff" ]]; then
        echo "[k3-opt] $label: missing $diff" >&2
        return 1
    fi
    echo "[k3-opt] $label: applying $(basename "$diff")"
    local out
    out=$(cd "$root" && patch -p1 --forward --batch < "$diff" 2>&1)
    local prc=$?
    printf '%s\n' "$out" | sed 's/^/    /'
    if [[ "$prc" == "0" ]] || printf '%s' "$out" | grep -q 'previously applied'; then
        echo "[k3-opt] $label: ok"
        return 0
    fi
    echo "[k3-opt] $label: FAILED (patch rc=$prc)" >&2
    return 1
}

# Report what actually landed, because "the patch ran" and "the engine is using it" are different
# claims and only the second one matters. Grep the engine's own startup log for these afterwards.
k3_opt_patches_expected_markers() {
    cat <<'EOF'
[k3-opt] after bring-up, confirm in the engine log:
  Setting attention block size to N tokens   the N the LMCache geometry is derived from
  mla_gluon / persistent                     #51088 + ps-enable reached the ASM decode path
  qlen                                       qlen pad/reorder took effect
A patch that applied but did not change the engine's behaviour looks identical to one that did.
EOF
}
