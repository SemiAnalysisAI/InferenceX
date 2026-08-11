#!/usr/bin/env bash
# LMCache MP as a prefill-side KV offload tier for Kimi-K3 in 1P1D disaggregation.
#
# Sourced by server_vllm.sh when KV_OFFLOAD_BACKEND=lmcache-k3. Nothing here runs at source time;
# the caller drives lmcache_mp_derive_geometry -> lmcache_mp_install -> lmcache_mp_start and then
# asks for lmcache_mp_connector_json / lmcache_mp_serve_args.
#
# Ported from the validated single-node arm
# (benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh, SemiAnalysisAI/InferenceX
# 14ab91eda12a3c2938b241218788379f25bda2cf). The numeric couplings below are copied rather than
# re-derived: each one has a run that died on it, and the failures are all engine-init or
# 20-minutes-in, i.e. expensive to rediscover.
#
# WHY PREFILL ONLY BY DEFAULT
# Decode's KV arrives over MoRIIO already computed, so a decode-side tier has almost nothing to load
# and spends bandwidth writing KV nobody re-reads. That is not a guess here: on the Mooncake tier the
# decode side measured load_get=0 with saves only, and the "100% external hit rate" it reported came
# from the MoRIIO transfer rather than the tier. LMCACHE_ON_DECODE=true turns it on for A/B.
#
# WHAT IS STILL UNVERIFIED ON THIS PATH (multi-node adds it; the single-node reference cannot answer)
#   1. MultiConnector asserts every child supports HMA unless the hybrid KV manager is disabled. K3 is
#      hybrid, so LMCacheMPConnector must declare SupportsHMA. vLLM's in-tree connector still carries
#      a hard refusal (reformat_block_ids raises when len(block_ids) > 1); LMCache's own module may
#      not. LMCACHE_CONNECTOR_MODULE_PATH selects which implementation resolves, and
#      lmcache_mp_assert_hybrid_ok checks it before a 1.7 TB weight load rather than after.
#   2. `align` interacts with MoRIIO's mamba transfer. MoRIIO derives per-group recurrent-state offsets
#      from the align window, and the prefill/decode windows differ under spec decode. The single-node
#      arm has no MoRIIO, so this pairing is new.
#   3. GPU headroom. LMCacheMPConnector holds GPU staging buffers and roughly halved the allocation
#      threshold single-node (wall at 184,320 computed tokens vs 360,960-373,248 without). MoRIIO
#      registers buffers too, so this is a second GPU-side consumer the reference never had.

# --------------------------------------------------------------------------------------------------
# Geometry. Sets LMCACHE_UNIFIED_BLOCK, LMCACHE_CHUNK_SIZE, LMCACHE_MNBT and validates them.
# --------------------------------------------------------------------------------------------------
lmcache_mp_derive_geometry() {
    local tp="${1:-8}"
    local dtype="${KV_CACHE_DTYPE:-}"

    # N is vLLM's UNIFIED attention block size and is NOT a constant: it moves with TP and with the
    # KV dtype. Upstream's K3 recipe quotes 768 for TP8, but that is the bf16 figure; fp8 halves the
    # element size so twice as many tokens fit one page and the engine logs 1536. Hardcoding 768
    # would violate both LMCache constraints on every fp8 run.
    if [[ "$dtype" == "fp8" || "$dtype" == fp8_* ]]; then
        LMCACHE_UNIFIED_BLOCK="${LMCACHE_UNIFIED_BLOCK:-1536}"
    else
        LMCACHE_UNIFIED_BLOCK="${LMCACHE_UNIFIED_BLOCK:-768}"
    fi

    # Chunk size is NOT the attention block. K3 is hybrid and the KDA cache group carries its own,
    # larger tokens_per_block; LMCache requires chunk_size to be a multiple of THAT. A run died at
    # engine init with "LMCache chunk size 1536 must be a multiple of engine group 14
    # tokens_per_block 3072" while N and the token budget were both correct. 2N satisfies LMCache's
    # chunk % 3072, the launcher's own chunk % N, and leaves the budget derivation untouched. Under
    # bf16 the two happened to agree, which is why this only surfaced with fp8.
    if [[ "$dtype" == "fp8" || "$dtype" == fp8_* ]]; then
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-$(( LMCACHE_UNIFIED_BLOCK * 2 ))}"
    else
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-$LMCACHE_UNIFIED_BLOCK}"
    fi

    # `align` snapshots the KDA state only on a block boundary at the end of a scheduler step, so the
    # per-step budget must cover at least one full block but stay under two. Upstream validated 1500
    # against N=768; 2N-72 keeps the same ratio at any N.
    LMCACHE_MNBT="${LMCACHE_MAX_NUM_BATCHED_TOKENS:-$(( LMCACHE_UNIFIED_BLOCK * 2 - 72 ))}"

    # Guard the derivation rather than trust it: a wrong N is an engine-init failure that arrives
    # after the weight load, so it costs a whole bring-up to discover.
    if (( LMCACHE_CHUNK_SIZE % LMCACHE_UNIFIED_BLOCK != 0 )); then
        echo "ERROR: LMCache chunk-size $LMCACHE_CHUNK_SIZE is not a multiple of N=$LMCACHE_UNIFIED_BLOCK" >&2
        return 1
    fi
    if (( LMCACHE_MNBT < LMCACHE_UNIFIED_BLOCK )) || (( LMCACHE_MNBT >= LMCACHE_UNIFIED_BLOCK * 2 )); then
        echo "ERROR: --max-num-batched-tokens $LMCACHE_MNBT outside [N, 2N) for N=$LMCACHE_UNIFIED_BLOCK" >&2
        return 1
    fi

    export LMCACHE_UNIFIED_BLOCK LMCACHE_CHUNK_SIZE LMCACHE_MNBT
    echo "[lmcache] N=$LMCACHE_UNIFIED_BLOCK (kv-cache-dtype=${dtype:-auto}, tp=$tp)" \
         "chunk-size=$LMCACHE_CHUNK_SIZE max-num-batched-tokens=$LMCACHE_MNBT"
    echo "[lmcache] NOTE: re-read the engine's own 'Setting attention block size to N tokens' line;" \
         "a derivation that disagrees with the engine is the failure this cannot catch itself"
}

# --------------------------------------------------------------------------------------------------
# L1 sizing. /dev/shm is the hard cap, because that is what backs the pool.
# --------------------------------------------------------------------------------------------------
lmcache_mp_size_l1() {
    local shm_free_gb cap
    shm_free_gb=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
    shm_free_gb="${shm_free_gb:-0}"

    if [[ -n "${LMCACHE_L1_SIZE_GB:-}" ]]; then
        :
    else
        # Upstream's K3 command uses a flat 100 GB rather than the reference 512 GB, and that is the
        # right default here too: memory pinning is unavailable in the ROCm container, so LMCache
        # disables its lazy allocator and allocates the WHOLE pool before serving. A smaller pool is
        # a much shorter and more predictable bring-up. Raise it once the arm is green.
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_DEFAULT_GB:-100}"
        local frac_gb
        frac_gb=$(awk -v s="$shm_free_gb" -v f="${LMCACHE_L1_SHM_FRACTION:-0.60}" 'BEGIN{printf "%d", s*f}')
        if (( frac_gb > 0 && frac_gb < LMCACHE_L1_SIZE_GB )); then
            echo "[lmcache] L1 sized from /dev/shm: ${frac_gb}GB (${LMCACHE_L1_SHM_FRACTION:-0.60} of ${shm_free_gb}GB free)"
            LMCACHE_L1_SIZE_GB="$frac_gb"
        fi
    fi

    cap=$(awk -v s="$shm_free_gb" 'BEGIN{printf "%d", s*0.90}')
    if (( cap > 0 && LMCACHE_L1_SIZE_GB > cap )); then
        echo "[lmcache] capping L1 ${LMCACHE_L1_SIZE_GB}GB -> ${cap}GB (/dev/shm has ${shm_free_gb}GB free)" >&2
        LMCACHE_L1_SIZE_GB="$cap"
    fi
    if (( LMCACHE_L1_SIZE_GB <= 0 )); then
        echo "ERROR: L1 resolved to ${LMCACHE_L1_SIZE_GB}GB; is DOCKER_SHM_SIZE set on the container?" >&2
        return 1
    fi
    export LMCACHE_L1_SIZE_GB
    echo "[lmcache] L1=${LMCACHE_L1_SIZE_GB}GB backing the run (/dev/shm free ${shm_free_gb}GB)"
}

# --------------------------------------------------------------------------------------------------
# Install. Source build, because upstream tags ROCm separately and PyPI is not reliably gfx950.
# --------------------------------------------------------------------------------------------------
lmcache_mp_install() {
    # v0.5.3 is the first tagged release containing 1dfa0777 "[Hotfix] KV cache shape edit for Kimi
    # K3 MLA layers (#4278)" -- a direct ancestor 50 commits back with nothing divergent. Pinning the
    # release rather than the hotfix commit is a straight upgrade, not a change of branch.
    local ref="${LMCACHE_GIT_REF:-140819c9d57a975dbc5678a6459a218e544cb58b}"
    local src="${LMCACHE_SRC:-/opt/lmcache-src}"

    if python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null 2>&1; then
        echo "[lmcache] already importable: $(python3 -c 'import lmcache; print(getattr(lmcache, "__version__", "unknown"))' 2>/dev/null)"
        return 0
    fi

    echo "[lmcache] building from source at $ref (ROCm/gfx950)"
    rm -rf "$src"
    git clone --filter=blob:none https://github.com/LMCache/LMCache.git "$src" || return 1
    git -C "$src" checkout --detach "$ref" || return 1
    git -C "$src" --no-pager log -1 --oneline
    (
        cd "$src" || exit 1
        export PYTORCH_ROCM_ARCH="gfx942,gfx950"
        export BUILD_WITH_HIP=1
        if command -v uv >/dev/null 2>&1; then
            uv pip install --system -e . --no-build-isolation
        else
            pip install -e . --no-build-isolation
        fi
    ) || return 1
    python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" || return 1
    echo "[lmcache] source build OK"
}

# --------------------------------------------------------------------------------------------------
# Hybrid guard. MultiConnector asserts every child supports HMA and K3 is hybrid, so find out here
# rather than in an assert deep in engine init after the weight load.
# --------------------------------------------------------------------------------------------------
lmcache_mp_assert_hybrid_ok() {
    local mod="${LMCACHE_CONNECTOR_MODULE_PATH:-}"
    python3 - "$mod" <<'PY'
import importlib
import sys

mod_path = sys.argv[1] or None
candidates = []
if mod_path:
    candidates.append(mod_path)
candidates += [
    "lmcache.integration.vllm.lmcache_mp_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
]

for name in candidates:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        print(f"[lmcache-hma] {name}: import failed ({type(e).__name__})")
        continue

    cls = getattr(mod, "LMCacheMPConnector", None)
    supports = None
    if cls is not None:
        # vLLM marks HMA support with a SupportsHMA mixin / class attribute depending on version, so
        # accept either shape rather than asserting one.
        supports = any(
            b.__name__ == "SupportsHMA" for b in getattr(cls, "__mro__", [])
        ) or bool(getattr(cls, "supports_hma", False))

    # The refusal that matters: a helper that raises on a multi-group block-id tuple.
    refuses = False
    fn = getattr(mod, "reformat_block_ids", None)
    if fn is not None:
        try:
            fn(([1, 2], [3, 4]))
        except Exception as e:
            refuses = "hybrid" in str(e).lower() or isinstance(e, RuntimeError)
    print(f"[lmcache-hma] {name}: SupportsHMA={supports} refuses_multi_group={refuses}")
PY
    echo "[lmcache] read the lines above before trusting this arm: K3 has 4 KV-cache groups, so an" \
         "implementation that refuses a multi-group block-id tuple cannot serve it"
}

# --------------------------------------------------------------------------------------------------
# Server lifecycle. One node-local server per node that runs the tier.
# --------------------------------------------------------------------------------------------------
lmcache_mp_start() {
    local tp="${1:-8}"
    local logdir="${2:-/run_logs}"
    local host_name="${3:-$(hostname)}"

    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-6555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-6556}"
    LMCACHE_MQ_TIMEOUT="${LMCACHE_MQ_TIMEOUT:-300}"
    LMCACHE_EVICTION_WATERMARK="${LMCACHE_EVICTION_WATERMARK:-0.85}"
    LMCACHE_EVICTION_RATIO="${LMCACHE_EVICTION_RATIO:-0.10}"
    LMCACHE_LOG="${logdir}/lmcache_${host_name}.log"

    # Upstream's validated K3 command, plus worker and eviction tuning for a TP8 agentic workload.
    # Deliberately minimal: the `reference` profile's extra flags (--max-gpu-workers, --l1-align-bytes,
    # --supported-transfer-mode) are not part of what upstream validated on K3, and an unrecognised
    # flag is a silent server-start failure.
    local -a cmd=(
        lmcache server
        --host "$LMCACHE_HOST"
        --port "$LMCACHE_PORT"
        --http-host "$LMCACHE_HOST"
        --http-port "$LMCACHE_HTTP_PORT"
        --chunk-size "$LMCACHE_CHUNK_SIZE"
        --max-workers "${LMCACHE_MAX_WORKERS:-$tp}"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --eviction-trigger-watermark "$LMCACHE_EVICTION_WATERMARK"
        --eviction-ratio "$LMCACHE_EVICTION_RATIO"
        --eviction-policy LRU
    )
    printf '%q ' "${cmd[@]}" > "${logdir}/lmcache_${host_name}_command.txt"
    printf '\n' >> "${logdir}/lmcache_${host_name}_command.txt"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY RUN: ${cmd[*]}"
        return 0
    fi

    "${cmd[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    export LMCACHE_PID LMCACHE_PORT LMCACHE_HTTP_PORT LMCACHE_MQ_TIMEOUT LMCACHE_LOG
    echo "[lmcache] server pid=$LMCACHE_PID log=$LMCACHE_LOG"
    lmcache_mp_wait_ready
}

lmcache_mp_wait_ready() {
    # Not instant, and not a hang: memory pinning is unavailable on ROCm, so LMCache disables its lazy
    # allocator and allocates the whole L1 pool up front. Expect minutes, scaling with pool size.
    # The heartbeat is what makes a slow allocation visibly different from a wedged server.
    local attempts="${LMCACHE_READY_ATTEMPTS:-1800}"
    local -a paths=(/health /healthz /v1/health /)
    local i p
    for (( i = 1; i <= attempts; i++ )); do
        for p in "${paths[@]}"; do
            if curl -sf -m 2 -o /dev/null "http://127.0.0.1:${LMCACHE_HTTP_PORT}${p}" 2>/dev/null; then
                echo "[lmcache] ready after ${i}s (endpoint ${p})"
                return 0
            fi
        done
        if [[ -n "${LMCACHE_PID:-}" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "ERROR: LMCache server exited during startup. Log follows:" >&2
            cat "$LMCACHE_LOG" >&2 || true
            return 1
        fi
        if (( i % 60 == 0 )); then
            echo "  ... still waiting for LMCache (${i}s / ${attempts}s), L1=${LMCACHE_L1_SIZE_GB:-?}GB"
        fi
        sleep 1
    done
    echo "ERROR: LMCache not ready after ${attempts}s. Tried ${paths[*]} on ${LMCACHE_HTTP_PORT}." >&2
    cat "$LMCACHE_LOG" >&2 || true
    return 1
}

lmcache_mp_stop() {
    [[ -n "${LMCACHE_PID:-}" ]] || return 0
    kill "$LMCACHE_PID" 2>/dev/null
    for _ in $(seq 1 30); do
        kill -0 "$LMCACHE_PID" 2>/dev/null || { echo "[lmcache] server stopped"; return 0; }
        sleep 1
    done
    kill -9 "$LMCACHE_PID" 2>/dev/null
    echo "[lmcache] server force-killed"
}

# --------------------------------------------------------------------------------------------------
# Connector JSON. Emits the CHILD dict for MultiConnector, not a whole --kv-transfer-config.
# --------------------------------------------------------------------------------------------------
lmcache_mp_connector_json() {
    # kv_role is kv_both regardless of the node's PD role: LMCache is a cache tier on whichever side
    # it runs, not a leg of the transfer. MultiConnector builds a fresh KVTransferConfig per child, so
    # the child does not inherit the outer role.
    #
    # kv_connector_module_path is emitted only when set. Left empty, LMCacheMPConnector resolves
    # through vLLM's factory to the in-tree implementation; set to
    # lmcache.integration.vllm.lmcache_mp_connector to use LMCache's own. That choice decides whether
    # the hybrid refusal in vLLM's copy is on the path, so it is a knob and not a constant.
    LMCACHE_MODULE_PATH="${LMCACHE_CONNECTOR_MODULE_PATH:-}" \
    LMCACHE_PORT="$LMCACHE_PORT" \
    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}" \
    LMCACHE_MQ_TIMEOUT="${LMCACHE_MQ_TIMEOUT:-300}" \
    python3 -c '
import json, os
extra = {
    "lmcache.mp.port": int(os.environ["LMCACHE_PORT"]),
    "lmcache.mp.mq_timeout": int(os.environ["LMCACHE_MQ_TIMEOUT"]),
}
entry = {
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": extra,
}
mod = os.environ.get("LMCACHE_MODULE_PATH") or ""
if mod:
    entry["kv_connector_module_path"] = mod
    # The module-path form is the one that takes a ZMQ-style host, matching how the k2.7 profile
    # passes it.
    extra["lmcache.mp.host"] = "tcp://" + os.environ["LMCACHE_HOST"]
print(json.dumps(entry))
'
}

# --------------------------------------------------------------------------------------------------
# Serve-arg overrides, scoped to this arm so the no-offload curve is untouched.
# --------------------------------------------------------------------------------------------------
lmcache_mp_serve_args() {
    local -a args=(
        --max-num-batched-tokens "$LMCACHE_MNBT"
        # `align` is the Mamba cache mode the KDA backend supports and the one that keeps reusable
        # state snapshots. vLLM picks it implicitly when prefix caching is on, but an implicit default
        # that silently flips to 'none' is exactly how this arm failed before, so pass it.
        --mamba-cache-mode align
    )
    # LMCacheMPConnector holds GPU staging buffers on top of the chunked-prefill workspace and roughly
    # halved the allocation threshold single-node. Under fp8 the chunked MLA workspace had already
    # doubled (98,304 -> 196,608 rows) because the page went 768 -> 1536. Both levers are scoped here.
    [[ -n "${LMCACHE_MAX_NUM_SEQS:-}" ]] && args+=(--max-num-seqs "$LMCACHE_MAX_NUM_SEQS")
    [[ -n "${LMCACHE_GPU_MEM_UTIL:-}" ]] && args+=(--gpu-memory-utilization "$LMCACHE_GPU_MEM_UTIL")
    printf '%s\n' "${args[@]}"
}

# Required by upstream's K3 recipe on the LMCache path; defaults to 0 and gates the fused
# latent-MoE tail in the K3 model definition.
lmcache_mp_export_env() {
    export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-1}"
    echo "[lmcache] VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=$VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION"
}
