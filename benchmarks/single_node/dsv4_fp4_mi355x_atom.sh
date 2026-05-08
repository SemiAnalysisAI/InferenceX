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

if [ "$EP_SIZE" -ne 1 ]; then
    echo "FATAL: DSv4 ATOM benchmark expects EP_SIZE=1, got $EP_SIZE" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1
export AITER_LOG_LEVEL=WARNING
# Use torch/RCCL TP reductions for DSv4. AITER's current TP reduce path was
# observed to corrupt equal-prompt batches at high concurrency.
export ATOM_DSV4_TP_REDUCE_BACKEND="${ATOM_DSV4_TP_REDUCE_BACKEND:-torch}"

# Keep the runtime overlay narrow: this benchmark uses the updated ATOM image
# from amd-master.yaml and overlays a pinned AITER fork with ROCm/aiter#2998
# plus the DSv4 FP8 blockscale projection dispatch fix. Install AITER before ATOM
# because the ATOM fork imports the DSv4 top-k/logits kernels at module load.
if [ "${AITER_DSV4_PR2998:-1}" = "1" ]; then
    AITER_PR2998_REPO=${AITER_PR2998_REPO:-https://github.com/Oseltamivir/aiter.git}
    AITER_PR2998_REF=${AITER_PR2998_REF:-dsv4-fp8-blockscale-wkv-fix}
    AITER_PR2998_SHA=${AITER_PR2998_SHA:-c3b57b6acdbb0e534c0fb8a6ff2f063b33a895eb}
    AITER_PR2998_DIR=${AITER_PR2998_DIR:-/tmp/aiter-dsv4-pr2998}

    rm -rf "$AITER_PR2998_DIR"
    git clone --filter=blob:none "$AITER_PR2998_REPO" "$AITER_PR2998_DIR"
    (
        cd "$AITER_PR2998_DIR"
        git fetch --depth=1 origin "$AITER_PR2998_REF"
        fetched_sha="$(git rev-parse FETCH_HEAD)"
        if [ "$fetched_sha" != "$AITER_PR2998_SHA" ]; then
            echo "FATAL: $AITER_PR2998_REF resolved to $fetched_sha, expected $AITER_PR2998_SHA" >&2
            exit 1
        fi
        git checkout --force FETCH_HEAD

        if [ ! -d 3rdparty/composable_kernel/include ]; then
            git submodule update --init --recursive --depth=1 3rdparty/composable_kernel \
                || git submodule update --init --recursive 3rdparty/composable_kernel
        fi

        PREBUILD_KERNELS=${AITER_PREBUILD_KERNELS:-0} \
        python3 -m pip install --no-deps --no-build-isolation --force-reinstall -e .
    )

    python3 - <<'PYEOF'
import inspect
import ast
from pathlib import Path
import aiter.ops.topk as topk_module
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer_topk
from aiter.ops.triton.attention.sparse_mqa_sink import sparse_mqa_sink
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

for fn in (
    top_k_per_row_decode,
    top_k_per_row_prefill,
    fp8_mqa_logits,
    dsv4_indexer_topk,
    sparse_mqa_sink,
):
    if not callable(fn):
        raise SystemExit(f"FATAL: imported non-callable AITER symbol: {fn!r}")

indexer_params = inspect.signature(dsv4_indexer_topk).parameters
missing = [name for name in ("seq_ids", "kv_lens") if name not in indexer_params]
if missing:
    raise SystemExit(f"FATAL: AITER PR2998 DSv4 Indexer API missing {missing}")
topk_tree = ast.parse(Path(topk_module.__file__).read_text())
topk_defs = {
    node.name: node for node in topk_tree.body if isinstance(node, ast.FunctionDef)
}
missing_k = [
    name
    for name in ("top_k_per_row_decode", "top_k_per_row_prefill")
    if name not in topk_defs
    or not any(arg.arg == "k" for arg in topk_defs[name].args.args)
]
if missing_k:
    raise SystemExit(f"FATAL: AITER top-k wrappers missing dynamic k parameter: {missing_k}")
print("AITER PR2998 DSv4 sparse/top-k/indexer ops imported successfully")
PYEOF
else
    echo "WARN: AITER_DSV4_PR2998=0; using image-provided AITER"
fi

# The updated ATOM image still does not ship DeepSeek-V4 model registration.
# Overlay the ATOM branch stacked on ROCm/ATOM#650 that wires the DSv4 Indexer
# path to ROCm/aiter#2998 top-k/logits kernels.
if [ "${ATOM_DSV4_PR650:-1}" = "1" ]; then
    ATOM_PR650_REPO=${ATOM_PR650_REPO:-https://github.com/Oseltamivir/ATOM.git}
    ATOM_PR650_REF=${ATOM_PR650_REF:-dsv4-deep-l0-diag}
    ATOM_PR650_SHA=${ATOM_PR650_SHA:-cf9a14e8f0a8af39fd257fa0dfafda22e89f3443}
    ATOM_PR650_DIR=${ATOM_PR650_DIR:-/tmp/atom-dsv4-pr650}

    rm -rf "$ATOM_PR650_DIR"
    git clone --filter=blob:none "$ATOM_PR650_REPO" "$ATOM_PR650_DIR"
    (
        cd "$ATOM_PR650_DIR"
        git fetch --depth=1 origin "$ATOM_PR650_REF"
        fetched_sha="$(git rev-parse FETCH_HEAD)"
        if [ "$fetched_sha" != "$ATOM_PR650_SHA" ]; then
            echo "FATAL: $ATOM_PR650_REF resolved to $fetched_sha, expected $ATOM_PR650_SHA" >&2
            exit 1
        fi
        git checkout --force FETCH_HEAD

        python3 - <<'PYEOF'
from pathlib import Path

v4_model_types = '("deepseek_v4", "deepseek_v4_pro", "deepseek_v4_flash")'

path = Path("atom/model_engine/model_runner.py")
source = path.read_text()
old = '''    def is_deepseek_v4(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        return self.hf_text_config.model_type == "deepseek_v4"
'''
new = f'''    def is_deepseek_v4(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        architectures = getattr(self.hf_text_config, "architectures", []) or []
        return model_type in {v4_model_types} or any(
            "DeepseekV4" in arch for arch in architectures
        )
'''
if old in source:
    source = source.replace(old, new, 1)
elif "deepseek_v4_pro" not in source[source.find("def is_deepseek_v4"): source.find("def is_mimo_v2")]:
    raise SystemExit("FATAL: model_runner.py is_deepseek_v4 did not match expected source")
old = '''            mt = self.config.hf_config.model_type
            known = _IOProc._per_req_cache_model_types()  # noqa: SLF001
            assert mt in known, (
                f"Attention builder {type(self.attn_metadata_builder).__name__} "
                f"reports per_req_cache_bytes>0 but model_type={mt!r} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({sorted(known)}). "
                "Add it to the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
'''
new = f'''            mt = self.config.hf_config.model_type
            architectures = getattr(self.config.hf_config, "architectures", []) or []
            known = _IOProc._per_req_cache_model_types()  # noqa: SLF001
            is_v4 = mt in {v4_model_types} or any(
                "DeepseekV4" in arch for arch in architectures
            )
            assert mt in known or is_v4, (
                f"Attention builder {{type(self.attn_metadata_builder).__name__}} "
                f"reports per_req_cache_bytes>0 but model_type={{mt!r}} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({{sorted(known)}}) "
                "and is not a recognized DeepSeek-V4 architecture. Add it to "
                "the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
'''
if old in source:
    source = source.replace(old, new, 1)
elif "is not a recognized DeepSeek-V4 architecture" not in source:
    raise SystemExit("FATAL: model_runner.py per-req cache assertion anchor missing")
path.write_text(source)

path = Path("atom/model_engine/llm_engine.py")
source = path.read_text()
old = '''                "deepseek_v4",
'''
new = '''                "deepseek_v4",
                "deepseek_v4_pro",
                "deepseek_v4_flash",
'''
if "deepseek_v4_pro" not in source:
    if old not in source:
        raise SystemExit("FATAL: llm_engine.py per-req cache model list anchor missing")
    source = source.replace(old, new, 1)
old = '''        if self.config.hf_config.model_type in self._per_req_cache_model_types():
            self.has_per_req_cache = True
'''
new = '''        hf_model_type = getattr(self.config.hf_config, "model_type", None)
        hf_architectures = getattr(self.config.hf_config, "architectures", []) or []
        if (
            hf_model_type in self._per_req_cache_model_types()
            or any("DeepseekV4" in arch for arch in hf_architectures)
        ):
            self.has_per_req_cache = True
'''
if old in source:
    source = source.replace(old, new, 1)
elif "hf_architectures = getattr(self.config.hf_config" not in source:
    raise SystemExit("FATAL: llm_engine.py per-req cache detection anchor missing")
path.write_text(source)

path = Path("atom/config.py")
source = path.read_text()
if '"deepseek_v4_pro": "deepseek_v3"' not in source:
    anchor = '''    "deepseek_v4": "deepseek_v3",  # V4 reuses V3 schema; V4-specific fields
'''
    insert = '''    "deepseek_v4": "deepseek_v3",  # V4 reuses V3 schema; V4-specific fields
    "deepseek_v4_pro": "deepseek_v3",
    "deepseek_v4_flash": "deepseek_v3",
'''
    if anchor not in source:
        raise SystemExit("FATAL: config.py V4 registry anchor missing")
    source = source.replace(anchor, insert, 1)
old = '''        if getattr(self.hf_config, "model_type", None) == "deepseek_v4":
'''
new = f'''        hf_model_type = getattr(self.hf_config, "model_type", None)
        hf_architectures = getattr(self.hf_config, "architectures", []) or []
        if hf_model_type in {v4_model_types} or any(
            "DeepseekV4" in arch for arch in hf_architectures
        ):
'''
if old in source:
    source = source.replace(old, new, 1)
elif "hf_model_type in" not in source:
    raise SystemExit("FATAL: config.py V4 block-size guard did not match expected source")
path.write_text(source)
PYEOF

        python3 -m pip install --no-deps --no-build-isolation --force-reinstall -e .
    )

    python3 - <<'PYEOF'
import inspect
from atom.model_engine.model_runner import support_model_arch_dict
from atom.models.deepseek_v4 import Indexer

target = support_model_arch_dict.get("DeepseekV4ForCausalLM")
if target != "atom.models.deepseek_v4.DeepseekV4ForCausalLM":
    raise SystemExit(f"FATAL: DeepseekV4ForCausalLM maps to {target!r}")
source = inspect.getsource(Indexer.forward_batched)
source += inspect.getsource(Indexer)
missing = [
    name
    for name in ("top_k_per_row_prefill", "top_k_per_row_decode", "fp8_mqa_logits")
    if name not in source
]
if missing:
    raise SystemExit(f"FATAL: ATOM DSv4 Indexer is not wired to AITER symbols {missing}")
print("ATOM DSv4 architecture registration and AITER top-k/logits wiring imported successfully")
PYEOF
else
    echo "WARN: ATOM_DSV4_PR650=0; using image-provided ATOM"
fi

# DSv4-Pro advertises a 1M native context. Set the benchmark context
# explicitly so ATOM does not reserve KV cache for the full native length.
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
    if [ "${ATOM_DSV4_PREFILL_DIAG_ENABLE:-1}" = "1" ]; then
        export ATOM_DSV4_PREFILL_DIAG="${ATOM_DSV4_PREFILL_DIAG:-1}"
        export ATOM_DSV4_PREFILL_DIAG_LAYERS="${ATOM_DSV4_PREFILL_DIAG_LAYERS:-0}"
        export ATOM_DSV4_PREFILL_DIAG_DEEP_LAYERS="${ATOM_DSV4_PREFILL_DIAG_DEEP_LAYERS:-none}"
        export ATOM_DSV4_PREFILL_DIAG_REF_LAYERS="${ATOM_DSV4_PREFILL_DIAG_REF_LAYERS:-0}"
        export ATOM_DSV4_PREFILL_DIAG_LINEAR_COMPARE="${ATOM_DSV4_PREFILL_DIAG_LINEAR_COMPARE:-0}"
        export ATOM_DSV4_PREFILL_DIAG_VERBOSE="${ATOM_DSV4_PREFILL_DIAG_VERBOSE:-1}"
        export ATOM_DSV4_PREFILL_DIAG_TOL="${ATOM_DSV4_PREFILL_DIAG_TOL:-1e-3}"
        export ATOM_DSV4_PREFILL_DIAG_SPARSE_COMPARE="${ATOM_DSV4_PREFILL_DIAG_SPARSE_COMPARE:-0}"
        export ATOM_DSV4_PREFILL_DIAG_SPARSE_LAYERS="${ATOM_DSV4_PREFILL_DIAG_SPARSE_LAYERS:-none}"
        export ATOM_DSV4_PREFILL_DIAG_SPARSE_HEAD="${ATOM_DSV4_PREFILL_DIAG_SPARSE_HEAD:-32}"
        export ATOM_DSV4_PREFILL_DIAG_SPARSE_ROWS="${ATOM_DSV4_PREFILL_DIAG_SPARSE_ROWS:-16}"
        echo "DSv4 prefill diagnostics enabled: layers=${ATOM_DSV4_PREFILL_DIAG_LAYERS}, deep_layers=${ATOM_DSV4_PREFILL_DIAG_DEEP_LAYERS}, sparse_layers=${ATOM_DSV4_PREFILL_DIAG_SPARSE_LAYERS}, verbose=${ATOM_DSV4_PREFILL_DIAG_VERBOSE}"
    fi
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

if [[ -z "${ATOM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    # Keep ATOM startup/warmup bounded without carrying a fork-only warmup knob.
    # Do not set this below ISL, otherwise an 8k prefill may never be admitted.
    ATOM_MAX_NUM_BATCHED_TOKENS=4096
    if [ "$ISL" -gt "$ATOM_MAX_NUM_BATCHED_TOKENS" ]; then
        ATOM_MAX_NUM_BATCHED_TOKENS="$ISL"
    fi
fi

run_dsv4_atom_eval_diagnostics() {
    local diag_file="sample_dsv4_atom_eval_diag_${RESULT_FILENAME}.jsonl"
    local diag_conc_list="${ATOM_DSV4_DIAG_CONCURRENCY_LIST:-1,$CONC}"
    local diag_script
    diag_script="$(dirname "$0")/../../utils/dsv4_atom_eval_diag.py"
    if [ ! -f "$diag_script" ]; then
        diag_script="/workspace/utils/dsv4_atom_eval_diag.py"
    fi
    echo "[DSv4 diag] Running concurrency isolation matrix; output: ${diag_file}"
    DIAG_PORT="$PORT" \
    DIAG_MODEL="$MODEL" \
    DIAG_CONC_LIST="$diag_conc_list" \
    DIAG_ISL="$ISL" \
    DIAG_OUT="$diag_file" \
    python3 "$diag_script"
}

start_gpu_monitor

set -x

BLOCK_SIZE=${BLOCK_SIZE:-128}
setup_atom_profile_args
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --server-port "$PORT" \
    -tp "$TP" \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size "$BLOCK_SIZE" \
    --max-num-batched-tokens "$ATOM_MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager \
    --trust-remote-code \
    "${ATOM_PROFILE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

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
    --server-pid "$SERVER_PID" \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --limit "${EVAL_LIMIT:-640}"
    append_lm_eval_summary
    if [ "${ATOM_DSV4_EVAL_DIAG:-1}" = "1" ]; then
        run_dsv4_atom_eval_diagnostics || echo "WARN: DSv4 eval diagnostics failed" >&2
    fi
fi

stop_gpu_monitor
set +x
