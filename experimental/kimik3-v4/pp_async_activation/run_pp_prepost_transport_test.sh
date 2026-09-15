#!/usr/bin/env bash
set -euo pipefail
rank="${SLURM_PROCID:?}"
name="k3-pp-prepost-transport-test-${rank}"
set +o pipefail
host_ip="$(hostname -I | awk '{print $1}')"
net_if="$(ip route | awk '/^default/ {print $5; exit}')"
set -o pipefail
docker rm -f "$name" >/dev/null 2>&1 || true
exec docker run --rm --name "$name" \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --add-host "$(hostname):${host_ip}" \
  --shm-size 64G \
  -e GLOO_SOCKET_IFNAME="$net_if" \
  -e NCCL_SOCKET_IFNAME="$net_if" \
  -e NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e K3_PP_ASYNC_ACTIVATION=1 \
  -e K3_PP_PREPOST_RECV=1 \
  -v /it-share/charwu/k3-throughput-screen/pp_async_activation:/ppasync:ro \
  --entrypoint bash \
  vllm/vllm-openai-rocm:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb \
  -lc "python3 /ppasync/apply_pp_async_activation_v5.py &&
       torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$rank \
         --master-addr=10.28.104.181 --master-port=29640 \
         /ppasync/test_pp_prepost_activation.py"
