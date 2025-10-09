# InferenceMAX

test

## System Design

We hook GPU nodes into the GitHub Action of this repo as **runners**, and a GitHub Action **workflow** assigns work to the runners.

- Benchmark template: The core logic of launching a single benchmark run
- Model templates (LLaMA 70B, DeepSeek R1, gpt-oss): The logic for declaring GPUs and their benchmark run configs
- Full sweep template: The workflow that contains the full benchmark sweep with configurations
- Full sweep scheduler: The workflow that schedules launches the full benchmark sweep

A runner loads code from this repo and executes them.
- `benchmarks/`: Scripts of benchmark configs, including the benchmark server and the benchmark client setup
- `runners/`: Scripts that launches the runners

The flow of operations is as follows:
1. GitHub Action workflow assigns a runner a config to benchmark, specified by the workflow YAML file.  
   Config includes: GPU type, Model, ISL / OSL, TP
1. The runner pulls the repo code and launchs the respective GPU node script in `runners/`,
   specifying the benchmark config: Model, ISL / OSL, TP
1. The script executes the corresponding script in `benchmarks/`, configuring it with the benchmark config


## Benchmark Client Configuration

| Parameter | Values |
| :-: | :- |
| (`ISL`, `OSL`) | (1024, 1024), (1024, 8192), (8192, 1024) |
| `CONC` | 4, 8, 16, 32, 64 |
| `RANDOM_RANGE_RATIO` | 0.8 |

```bash
git clone https://github.com/kimbochen/bench_serving.git 
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el'
```


## Server Configurations

Server configurations evolve quickly.  
Please checkout the scripts in `benchmarks/` for the most up-to-date configs.


## Sponsors

- SemiAnalysis
- NVIDIA
- AMD
- Crusoe
- Nebius
- CoreWeave
- TogetherAI
- TensorWave
- Oracle
