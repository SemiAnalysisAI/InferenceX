## Experimental WIP: Multi turn with/without CPU KVCache Offloading

lit review
- https://lmsys.org/blog/2025-09-10-sglang-hicache/
-  sglang calls GPU HBM as (L1) and CPU DRAM as (L2)
- https://lmsys.org/images/blog/hicache/mooncake_benchmark.png
- single turn long context Q&A  https://arxiv.org/abs/2311.04939 (seems more like an shared prefix style similar to cascade attention (pre cursor to sglang radix attention )) https://flashinfer.ai/2024/02/02/cascade-inference.html
- synethic & sharegpt vllm multi turn datasets https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn
- Production Alibiba Multi turn dataset https://arxiv.org/abs/2506.02634 (seem to not provide the acutal prompts and outputs tho, more just prompt lengths and output lengths, etc.)
- sglang synthetic multi turn benchmark script here https://github.com/sgl-project/sglang/tree/main/benchmark/hicache
- interestingly sglang blog simulates PD disagg via just setting OSL as 1
- MT-bench https://arxiv.org/abs/2402.14762
```bash
python3 benchmark/hicache/bench_multiturn.py --model-path $MODEL_PATH --disable-random-sample \
--output-length 1 --request-length 2048 \ # simulate P-D disaggregation
```

## Cam Experimentation

On B200 Nebius cluster. To reproduce:

```
cam@login-1:~$ sudo su - gharunner0

gharunner0@login-1:~$ salloc --partition=main --gres=gpu:8 --exclusive --time=360
salloc: Granted job allocation 5186
salloc: Nodes worker-1 are ready for job

srun --jobid=5186 --container-image=$IMAGE   --container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER: -1}   --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE   --no-container-mount-home   --container-remap-root   --container-writable   --container-workdir=/workspace/   --no-container-entrypoint   --export=ALL,PORT=$(( 8888 + ${USER: -1} )),UCX_NET_DEVICES=eth0   --pty bash
```
