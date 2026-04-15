#!/usr/bin/env bash

resolve_single_node_benchmark_script() {
    local model_code="$1"
    local precision="$2"
    local runner_code="$3"
    local framework="${4:-}"
    local spec_decoding="${5:-}"
    local script_base="benchmarks/single_node/${model_code}_${precision}_${runner_code}"

    if [[ "${BENCHMARK_TYPE:-}" == "isb1_replay" ]] && [[ "$framework" == "sglang" || "$framework" == "vllm" ]]; then
        local runtime_candidate="${script_base}_${framework}.sh"
        if [[ -f "$runtime_candidate" ]]; then
            printf '%s\n' "$runtime_candidate"
            return 0
        fi
    fi

    local framework_suffix=""
    local spec_suffix=""
    if [[ "$framework" == "trt" ]]; then
        framework_suffix="_trt"
    fi
    if [[ "$spec_decoding" == "mtp" ]]; then
        spec_suffix="_mtp"
    fi

    local legacy_candidate="${script_base}${framework_suffix}${spec_suffix}.sh"
    if [[ -f "$legacy_candidate" ]]; then
        printf '%s\n' "$legacy_candidate"
        return 0
    fi

    echo "ERROR: Could not resolve single-node benchmark script." >&2
    echo "  model=$model_code precision=$precision runner=$runner_code framework=${framework:-<unset>} spec_decoding=${spec_decoding:-<unset>} benchmark_type=${BENCHMARK_TYPE:-<unset>}" >&2
    if [[ "${BENCHMARK_TYPE:-}" == "isb1_replay" ]] && [[ "$framework" == "sglang" || "$framework" == "vllm" ]]; then
        echo "  checked runtime-aware candidate: ${script_base}_${framework}.sh" >&2
    fi
    echo "  checked legacy candidate: $legacy_candidate" >&2
    return 1
}
