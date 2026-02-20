## vLLM DecodeBenchConnector

```
vllm serve Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 1 --kv-transfer-config '{"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0}}' 
```
```
vllm serve Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 1
```

```
vllm bench serve --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-30B-A3B-FP8 --dataset-name random --random-input-len 32768 --random-output-len 100  --max-concurrency 10 --seed 56075618 --num-prompts 20 --ignore-eos
```

## experiment done on h200 sxm

| Metric (Qwen3-30B-A3B, 32k ISL, TP1) | Without DecodeBenchConnector | With DecodeBenchConnector | Delta |
| --- | --- | --- | --- |
| TTFT (ms) | 5,220 | 341.57 | **15x faster** |
| TPOT (ms) | 84.35 | 16.02 | **5.3x faster** |
| Interactivity (tok/s) | 12 | 62 | **5x better** |
| Output throughput (tok/s) | 73.18 | 513.73 | **7x higher** |
