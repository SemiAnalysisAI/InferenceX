# CI Command Reference

## Validate Matrix Generation

Use `test-config` for one config:

```bash
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/amd-kimi-mori-agentic-sweep-lmcachemp-1p2d.yaml \
  --config-keys kimik2.5-fp4-mi355x-vllm-disagg-mori-lmcache-agentic \
  --scenario-type agentic-coding \
  --conc 32 | python3 -m json.tool
```

Check expected fields:

```bash
python3 utils/matrix_logic/generate_sweep_configs.py test-config ... \
  | python3 -m json.tool \
  | grep -E 'kv-offloading|kv-offload-backend|total-cpu-dram-gb|exp-name'
```

Use `full-sweep` for dispatch dry runs:

```bash
python3 utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/amd-kimi-mori-agentic-sweep-lmcachemp-1p2d.yaml \
  --model-prefix kimik2.5 \
  --framework vllm-disagg \
  --runner-type cluster:mi355x-amds \
  --scenario-type agentic-coding \
  --min-conc 32 \
  --max-conc 32
```

## Dispatch e2e-tests.yml

`run-sweep.yml` is label-triggered, not workflow-dispatchable. For manual one-offs:

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=yichaozhu/kimi-mori-lmcache-agentx-v1.0' \
  -f 'inputs[test-name]=Kimi MoRI LMCache agentic 1P2D conc32' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/amd-kimi-mori-agentic-sweep-lmcachemp-1p2d.yaml --model-prefix kimik2.5 --framework vllm-disagg --runner-type cluster:mi355x-amds --scenario-type agentic-coding --min-conc 32 --max-conc 32' \
  -f 'inputs[duration-override]=900'
```

The dispatch API returns no run ID. Find the latest run:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')
```

## Monitor

```bash
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json status,conclusion,createdAt,updatedAt,url
```

List jobs:

```bash
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | [.databaseId, .name, .status, .conclusion] | @tsv'
```

## Download Artifacts

```bash
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/"$RUN_ID"/artifacts \
  --jq '.artifacts[] | [.name, .size_in_bytes, .expired] | @tsv'
```

```bash
mkdir -p /tmp/inferencex-ci-"$RUN_ID"
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX -D /tmp/inferencex-ci-"$RUN_ID"
```

Parse aggregate benchmark results:

```bash
jq -r '.[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round), .mean_ttft, .mean_tpot] | @tsv' \
  /tmp/inferencex-ci-"$RUN_ID"/results_bmk/agg_bmk.json | column -t
```

Parse agentic result JSON:

```bash
find /tmp/inferencex-ci-"$RUN_ID" -name '*agentic*.json' -o -name 'agentic_bench.json'
jq '{conc, num_requests_successful, num_requests_total, input_tput_tps, output_tput_tps, total_tput_tps, server_gpu_cache_hit_rate, server_external_cache_hit_rate}' \
  path/to/agentic_bench.json
```

## Safe Force Push After Squash

```bash
git fetch origin +refs/heads/yichaozhu/kimi-mori-lmcache-agentx-v1.0:refs/remotes/origin/yichaozhu/kimi-mori-lmcache-agentx-v1.0
git log --oneline --left-right --cherry-pick origin/yichaozhu/kimi-mori-lmcache-agentx-v1.0...HEAD
git push --force-with-lease=refs/heads/yichaozhu/kimi-mori-lmcache-agentx-v1.0:$(git rev-parse origin/yichaozhu/kimi-mori-lmcache-agentx-v1.0) \
  origin HEAD:yichaozhu/kimi-mori-lmcache-agentx-v1.0
```
