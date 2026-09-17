FROM c56best:archived

USER root

COPY apply_vllm_50514_pp_spec.sh /opt/k3-dspark/
COPY vllm_pr50514_pp_only/ /opt/k3-dspark/vllm_pr50514_pp_only/
COPY vllm_pr50514_overlay/ /opt/k3-dspark/vllm_pr50514_overlay/
COPY dspark_cache_reuse/ /opt/k3-dspark/dspark_cache_reuse/
COPY c56best_dspark_image/verify_c56best_dspark.py /opt/k3-dspark/
COPY c56best_dspark_image/upstream/vllm-pr52921.patch /opt/k3-dspark/
COPY c56best_dspark_image/patch_vllm_pr52921_worker_compat.py /opt/k3-dspark/
COPY c56best_dspark_image/patch_vllm_pr56621_no_forward_store.py /opt/k3-dspark/

RUN PATCH_DIR=/opt/k3-dspark/vllm_pr50514_pp_only \
      bash /opt/k3-dspark/apply_vllm_50514_pp_spec.sh \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_aux_contract_v028.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_pp_draft_transfer_decision.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_pp_draft_collective_stream.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_pp_combined_sample_draft_broadcast.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_pp_combined_broadcast_sync.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_pp_scheduler_eagle_groups.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_scheduler_kv_cache_eagle_merge.py \
    && DSPARK_REPLAY_PATCH=/opt/k3-dspark/dspark_cache_reuse/vllm-51295-replay-boundary.patch \
       python3 /opt/k3-dspark/dspark_cache_reuse/apply_dspark_replay_boundary_fix.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_disable_eagle_block_drop_v028.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_eagle_group_annotate_fixup.py \
    && python3 /opt/k3-dspark/dspark_cache_reuse/patch_dspark_kv_group_diagnostics.py \
    && cd /usr/local/lib/python3.12/dist-packages \
    && git apply --exclude="tests/**" \
         --exclude="vllm/v1/simple_kv_offload/worker.py" \
         /opt/k3-dspark/vllm-pr52921.patch \
    && python3 /opt/k3-dspark/patch_vllm_pr52921_worker_compat.py \
    && python3 /opt/k3-dspark/patch_vllm_pr56621_no_forward_store.py \
    && python3 /opt/k3-dspark/verify_c56best_dspark.py

LABEL org.opencontainers.image.title="Kimi-K3 C56 best with DSpark PP" \
      org.opencontainers.image.description="C56best image plus PR 53901, PR 52921, PR 56621, and global draft KV-group propagation" \
      inferencex.vllm.pr="53901,52921,56621" \
      inferencex.dspark.kv-group-fix="global-projection+scheduler-merge"
