#!/usr/bin/env bash
set -o pipefail

if [[ $# != 2 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint infmax_workspace" >&2
    exit 1
fi
endpoint=$1
workspace=$2
source "$workspace/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars MODEL_NAME EVAL_CONC EVAL_LIMIT || exit 1
[[ "$EVAL_LIMIT" == 32 ]] || exit 1
mkdir -p /logs/cross_tp_gate || exit 1
cd "$workspace" || exit 1
gsm_rc=0
bash "$workspace/benchmarks/multi_node/srt_eval.sh" "$endpoint" "$workspace" || gsm_rc=$?
score_rc=0
python3 "$workspace/utils/evals/validate_scores.py" --model-prefix qwen3.5 \
    --expected-concs "$EVAL_CONC" > /logs/cross_tp_gate/gsm-score-validator.txt 2>&1 || score_rc=$?
probe_rc=0
python3 "$workspace/experimental/qwen35_cross_tp/transport_probe.py" \
    --endpoint "$endpoint" --model "$MODEL_NAME" --tokenizer /model \
    --timeout 600 --output /logs/cross_tp_gate/long-inputs || probe_rc=$?
printf '{"gsm_execution_exit":%d,"gsm_score_exit":%d,"long_probe_exit":%d,"formal_power_measurement":false}\n' \
    "$gsm_rc" "$score_rc" "$probe_rc" > /logs/cross_tp_gate/gate-status.json
[[ "$gsm_rc" == 0 && "$score_rc" == 0 && "$probe_rc" == 0 ]]
