#!/usr/bin/env bash
# ============================================================================
# apply_k3_cb8104839c_patches.sh
#
# Reproduces ALL in-container filesystem changes for the Kimi-K3 DSpark FP8 MLA
# (MTP) noprefixcache stack on image:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#
# Changes applied (idempotent):
#   1. vLLM PR #50619  - "Fix Kimi-K3 DSpark FP8 MLA verification":
#        kimi_k3/nvidia/mla.py bf16-query branch (do_kv_cache_update instead of
#        asserting when the backend can't take an fp8 query); rocm_aiter_mla.py
#        supports_quant_query_input=False for <16h+fp8 and forward_mqa native
#        4-D Gluon for <16-head decode + MTP verify; worker cudagraph
#        draft-exclude. (vllm/* only; the PR's tests/* are omitted.)
#   2. aiter mla_gluon.py bh16bn128 batch relaxation (1 <= batch_size <= 128).
#   3. aiter PR #4474  - int64 KV stride widening on the >2GB global_load path.
#   4. Triton 3.7.0 (AMD ROCm 7.2.0 build) + tabulate (Gluon MLA needs
#      PaddedSharedLayout(cga_layout=...)).
#
# Run INSIDE a fresh container of the image above:
#   docker exec -i <container> bash < apply_k3_cb8104839c_patches.sh
#
# Server-side runtime config for the passing benchmark (NOT container changes):
#   --kv-cache-dtype fp8, TP8, --gpu-memory-utilization 0.8,
#   --speculative-config num_speculative_tokens=2 attention_backend=TRITON_MLA,
#   --no-enable-prefix-caching  (prefix caching is unsupported on this stack).
# ============================================================================
set -euo pipefail

VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
AITER_DIR="$(python -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')"
echo "[patch] VLLM_DIR=$VLLM_DIR"
echo "[patch] AITER_DIR=$AITER_DIR"

apply_idem() {  # $1=dir  $2=diff  $3=label
  cd "$1"
  if git apply --reverse --check -p1 "$2" 2>/dev/null; then
    echo "[patch] $3: already applied, skipping"
  else
    git apply --check -p1 "$2"
    git apply -p1 "$2"
    echo "[patch] $3: applied"
  fi
}

# Same edit, two source baselines. vllm#50476 ("Mask the AITER MLA small-head
# verify flatten causally") rewrote the exact block that the rocm_aiter_mla.py
# hunk replaces, so the context differs between nightlies on either side of it.
# Both variants converge on a byte-identical file, so the reverse-check for
# idempotency works whichever one landed. Fails loudly if neither applies --
# a silently skipped fp8 MLA verify fix is worse than a dead cell.
apply_one_of() {  # $1=dir  $2=label  $3..=candidate diffs
  local dir="$1" label="$2"; shift 2
  cd "$dir"
  local d
  for d in "$@"; do
    if git apply --reverse --check -p1 "$d" 2>/dev/null; then
      echo "[patch] $label: already applied ($(basename "$d")), skipping"
      return 0
    fi
  done
  for d in "$@"; do
    if git apply --check -p1 "$d" 2>/dev/null; then
      git apply -p1 "$d"
      echo "[patch] $label: applied ($(basename "$d"))"
      return 0
    fi
  done
  echo "[patch] $label: NO VARIANT APPLIES -- tried: $*" >&2
  for d in "$@"; do echo "[patch] --- $(basename "$d") ---" >&2; git apply --check -p1 "$d" >&2 2>&1 || true; done
  return 1
}

cat > /tmp/pr50619_vllm.diff <<'PR50619_EOF'
diff --git a/vllm/models/kimi_k3/nvidia/mla.py b/vllm/models/kimi_k3/nvidia/mla.py
index 3334f3bbbb49..f46e98624e37 100644
--- a/vllm/models/kimi_k3/nvidia/mla.py
+++ b/vllm/models/kimi_k3/nvidia/mla.py
@@ -594,8 +594,7 @@ def _decode_concat_cache(
         cos_sin_cache: torch.Tensor | None,
         slot_mapping: torch.Tensor,
     ) -> torch.Tensor:
-        """Fused decode query-concat + latent cache insert, dispatched by cache
-        dtype (same policy as prefill: fp8 cache -> fp8 query)."""
+        """Build the decode query and update the cache for its dtype/backend."""
         if self.kv_cache_dtype == "fp8_ds_mla":
             cache = self.kv_cache
             if cache.dtype != torch.uint8:
@@ -612,10 +611,21 @@ def _decode_concat_cache(
                 cos_sin_cache=cos_sin_cache,
             )
         if is_quantized_kv_cache(self.kv_cache_dtype):
-            assert self.impl.supports_quant_query_input, (  # type: ignore[attr-defined]
-                "Kimi-K3 fp8 KV cache decode requires a backend that accepts an "
-                "fp8 (quantized) query input."
-            )
+            if not self.impl.supports_quant_query_input:  # type: ignore[attr-defined]
+                if positions is not None:
+                    assert self.rotary_emb is not None
+                    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
+                    q_pe = q_pe.to(ql_nope.dtype)
+                    k_pe = k_pe.to(kv_c_normed.dtype)
+                self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
+                    kv_c_normed,
+                    k_pe,
+                    self.kv_cache,
+                    slot_mapping,
+                    self.kv_cache_dtype,
+                    self._k_scale,
+                )
+                return torch.cat((ql_nope, q_pe), dim=-1)
             cache = self.kv_cache
             if cache.dtype != torch.float8_e4m3fn:
                 cache = cache.view(torch.float8_e4m3fn)
diff --git a/vllm/v1/worker/gpu/attn_utils.py b/vllm/v1/worker/gpu/attn_utils.py
index 598bddf47ab0..010d53210115 100644
--- a/vllm/v1/worker/gpu/attn_utils.py
+++ b/vllm/v1/worker/gpu/attn_utils.py
@@ -92,6 +92,7 @@ def init_attn_backend(
     kv_cache_config: KVCacheConfig,
     vllm_config: VllmConfig,
     device: torch.device,
+    cg_support_exclude_layers: set[str] | None = None,
     active_layer_names: set[str] | None = None,
 ) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
     # Phase 1: discover attention groups for each kv cache group.
@@ -165,6 +166,15 @@ def init_attn_backend(
             else:
                 if hasattr(builder, "set_workspace_buffer"):
                     builder.set_workspace_buffer(attn_backend_workspace)
+            # A group owned entirely by a separately-managed model part must
+            # not constrain this runner: a spec-decode draft gets its own
+            # CudaGraphManager and has a first-class eager fallback, so letting
+            # it in here downgrades the target for a decision it does not share.
+            if (
+                cg_support_exclude_layers is not None
+                and set(group.layer_names) <= cg_support_exclude_layers
+            ):
+                continue
             # Check cudagraph support for the attention backend
             cg_support = builder.get_cudagraph_support(
                 vllm_config,
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
index cea735c0e804..492dd4fcac7a 100644
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -470,7 +470,14 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
             max_num_blocks_per_group.append(max_num_blocks)
 
         self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
-            self.kv_cache_config, self.vllm_config, self.device
+            self.kv_cache_config,
+            self.vllm_config,
+            self.device,
+            cg_support_exclude_layers=(
+                self.speculator.draft_attn_layer_names
+                if isinstance(self.speculator, DraftModelSpeculator)
+                else None
+            ),
         )
         attn_cg_support = attn_cg_support.narrow(
             *self.model_state.get_additional_cg_support()
PR50619_EOF

# rocm_aiter_mla.py is split out because its context straddles vllm#50476.
# _post50476: nightlies at/after cb8104839 (causal flatten present upstream).
# _pre50476:  nightlies before it, e.g. 124154a8843d (non-causal flatten).
cat > /tmp/pr50619_ram_post50476.diff <<'RAM_POST_EOF'
diff --git a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
index 43584998a0bb..e502dccc12cf 100644
--- a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
+++ b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -20,6 +20,7 @@
     QueryLenSupport,
 )
 from vllm.triton_utils import tl, triton
+from vllm.utils.torch_utils import is_quantized_kv_cache
 from vllm.v1.attention.backend import (
     AttentionCGSupport,
     AttentionLayer,
@@ -861,6 +862,17 @@ def __init__(
         )
         AiterMLAHelper.check_num_heads_validity(num_heads)
 
+        # Decode with fewer than 16 heads is served by AITER's Gluon MLA
+        # kernel. Its fp8 KV regime (bh16bn128) takes a bf16 query and folds
+        # the KV dequant scale into the QK temperature, so a pre-quantized
+        # query is rejected with "q_nope/q_pe must be bf16". Tell the common
+        # layer to leave the query alone, as TritonMLAImpl already does for
+        # its own fp8 KV path.
+        if num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and is_quantized_kv_cache(
+            self.kv_cache_dtype
+        ):
+            self.supports_quant_query_input = False
+
         unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
         if any(unsupported_features):
             raise NotImplementedError(
@@ -1073,118 +1085,51 @@ def forward_mqa(
         assert decode.max_qo_len is not None
         assert decode.paged_kv_indptr is not None
         assert decode.paged_kv_indices is not None
-        if decode.use_gluon_decode:
-            if type(q) is tuple:
-                q_nope, q_pe = q
-            else:
-                q_nope, q_pe = torch.split(
-                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
-                )
-            B, num_q_heads, _ = q_nope.shape
-            o = torch.empty(
-                B,
-                num_q_heads,
-                self.kv_lora_rank,
-                dtype=decode.attn_out_dtype,
-                device=q_nope.device,
-            )
-            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
-            mla_gluon = _get_mla_gluon()
-            mla_gluon(
-                q_nope=q_nope,
-                q_pe=q_pe,
-                kv_c=kv_buffer,
-                o=o,
-                page_table=decode.paged_kv_indices,
-                seq_info=decode.paged_kv_indptr,
-                sm_scale=self.scale,
-                k_pe=None,
-                kv_pe_offset=self.kv_lora_rank,
-                use_2d_view=False,
-                kv_scale=1.0,
-                min_kv_seq_len=decode.min_kv_seq_len,
-            )
-            return o, None
-
-        # 12-head (<16) multi-token verify (DSpark): the asm path has no
-        # gqa<16, qseqlen>1 kernel. Flatten each verify token to its own
-        # qseqlen=1 gluon decode, mirroring the TRITON_MLA / sparse-backend
-        # flatten but on the fast gluon kernel. The block is causal -- the
-        # target is checking draft tokens, so position t must not see t+1 --
-        # and attention rows are independent, so giving row t the KV range
-        # [0, context + t] is exactly causal multi-token attention.
-        if (
+        if decode.use_gluon_decode or (
             self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
             and int(decode.max_qo_len) > 1
         ):
-            qlen = int(decode.max_qo_len)
             if type(q) is tuple:
                 q_nope, q_pe = q
             else:
                 q_nope, q_pe = torch.split(
                     q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                 )
-            B, num_q_heads, _ = q_nope.shape
+            num_tokens, num_q_heads, _ = q_nope.shape
+            qlen = int(decode.max_qo_len)
+            if num_tokens % qlen != 0:
+                raise ValueError(
+                    f"Gluon MLA requires uniform query lengths, got "
+                    f"{num_tokens} tokens and query length {qlen}"
+                )
+            batch_size = num_tokens // qlen
+            if qlen > 1:
+                # Gluon applies the causal tail for native 4-D MTP queries.
+                q_nope = q_nope.view(batch_size, qlen, num_q_heads, self.kv_lora_rank)
+                q_pe = q_pe.view(batch_size, qlen, num_q_heads, self.qk_rope_head_dim)
             o = torch.empty(
-                B,
-                num_q_heads,
+                *q_nope.shape[:-1],
                 self.kv_lora_rank,
                 dtype=decode.attn_out_dtype,
                 device=q_nope.device,
             )
             kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
-            # Expand per-request paged-KV to per-verify-token. Row r*qlen+t is
-            # request r's verify token t, and seq_lens counts the tokens
-            # scheduled in this step, so a request's KV range already spans its
-            # whole verify block and context_r = seq_len_r - qlen. Token t may
-            # attend to [0, context_r + t], i.e. seq_len_r - (qlen - 1) + t
-            # entries. paged_kv_indices lists a request's pages in ascending
-            # position order, so each row's causal window is a prefix of that
-            # request's slice and only the row length changes. Rows clamp to
-            # zero for cudagraph padding requests, whose seq_len is 0. Fully
-            # vectorized (no host loop).
-            old_indptr = decode.paged_kv_indptr
-            per_req_len = old_indptr[1:] - old_indptr[:-1]
-            dev = q_nope.device
-            row_req = torch.arange(per_req_len.shape[0], device=dev).repeat_interleave(
-                qlen
-            )
-            row_len = (
-                (
-                    per_req_len.unsqueeze(1)
-                    - (qlen - 1)
-                    + torch.arange(qlen, device=dev, dtype=per_req_len.dtype)
-                )
-                .clamp_(min=0)
-                .flatten()
-            )
-            new_indptr = torch.cat([old_indptr.new_zeros(1), row_len.cumsum(0)]).to(
-                torch.int32
-            )
-            total = int(new_indptr[-1].item())
-            within = torch.arange(total, device=dev, dtype=torch.int64) - new_indptr[
-                :-1
-            ].to(torch.int64).repeat_interleave(row_len)
-            src = (
-                old_indptr[row_req].to(torch.int64).repeat_interleave(row_len) + within
-            )
-            new_indices = decode.paged_kv_indices[src]
             mla_gluon = _get_mla_gluon()
             mla_gluon(
                 q_nope=q_nope,
                 q_pe=q_pe,
                 kv_c=kv_buffer,
                 o=o,
-                page_table=new_indices,
-                seq_info=new_indptr,
+                page_table=decode.paged_kv_indices,
+                seq_info=decode.paged_kv_indptr,
                 sm_scale=self.scale,
                 k_pe=None,
                 kv_pe_offset=self.kv_lora_rank,
                 use_2d_view=False,
                 kv_scale=1.0,
-                min_kv_seq_len=int(row_len.min()),
+                min_kv_seq_len=decode.min_kv_seq_len,
             )
-            return o, None
+            return o.view(num_tokens, num_q_heads, self.kv_lora_rank), None
 
         if type(q) is tuple:
             q = torch.cat(q, dim=-1)
RAM_POST_EOF

cat > /tmp/pr50619_ram_pre50476.diff <<'RAM_PRE_EOF'
diff --git a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
--- a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
+++ b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -20,6 +20,7 @@
     QueryLenSupport,
 )
 from vllm.triton_utils import tl, triton
+from vllm.utils.torch_utils import is_quantized_kv_cache
 from vllm.v1.attention.backend import (
     AttentionCGSupport,
     AttentionLayer,
@@ -861,6 +862,17 @@
         )
         AiterMLAHelper.check_num_heads_validity(num_heads)
 
+        # Decode with fewer than 16 heads is served by AITER's Gluon MLA
+        # kernel. Its fp8 KV regime (bh16bn128) takes a bf16 query and folds
+        # the KV dequant scale into the QK temperature, so a pre-quantized
+        # query is rejected with "q_nope/q_pe must be bf16". Tell the common
+        # layer to leave the query alone, as TritonMLAImpl already does for
+        # its own fp8 KV path.
+        if num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and is_quantized_kv_cache(
+            self.kv_cache_dtype
+        ):
+            self.supports_quant_query_input = False
+
         unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
         if any(unsupported_features):
             raise NotImplementedError(
@@ -1073,101 +1085,51 @@
         assert decode.max_qo_len is not None
         assert decode.paged_kv_indptr is not None
         assert decode.paged_kv_indices is not None
-        if decode.use_gluon_decode:
-            if type(q) is tuple:
-                q_nope, q_pe = q
-            else:
-                q_nope, q_pe = torch.split(
-                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
-                )
-            B, num_q_heads, _ = q_nope.shape
-            o = torch.empty(
-                B,
-                num_q_heads,
-                self.kv_lora_rank,
-                dtype=decode.attn_out_dtype,
-                device=q_nope.device,
-            )
-            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
-            mla_gluon = _get_mla_gluon()
-            mla_gluon(
-                q_nope=q_nope,
-                q_pe=q_pe,
-                kv_c=kv_buffer,
-                o=o,
-                page_table=decode.paged_kv_indices,
-                seq_info=decode.paged_kv_indptr,
-                sm_scale=self.scale,
-                k_pe=None,
-                kv_pe_offset=self.kv_lora_rank,
-                use_2d_view=False,
-                kv_scale=1.0,
-                min_kv_seq_len=decode.min_kv_seq_len,
-            )
-            return o, None
-
-        # 12-head (<16) non-causal multi-token verify (DSpark): the asm path has
-        # no gqa<16, qseqlen>1 kernel. Flatten each verify token to a qseqlen=1
-        # gluon decode over the same committed prefix (non-causal), mirroring the
-        # TRITON_MLA / sparse-backend flatten but on the fast gluon kernel. Each
-        # request's KV metadata is repeated max_qo_len times.
-        if (
+        if decode.use_gluon_decode or (
             self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
             and int(decode.max_qo_len) > 1
         ):
-            qlen = int(decode.max_qo_len)
             if type(q) is tuple:
                 q_nope, q_pe = q
             else:
                 q_nope, q_pe = torch.split(
                     q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                 )
-            B, num_q_heads, _ = q_nope.shape
+            num_tokens, num_q_heads, _ = q_nope.shape
+            qlen = int(decode.max_qo_len)
+            if num_tokens % qlen != 0:
+                raise ValueError(
+                    f"Gluon MLA requires uniform query lengths, got "
+                    f"{num_tokens} tokens and query length {qlen}"
+                )
+            batch_size = num_tokens // qlen
+            if qlen > 1:
+                # Gluon applies the causal tail for native 4-D MTP queries.
+                q_nope = q_nope.view(batch_size, qlen, num_q_heads, self.kv_lora_rank)
+                q_pe = q_pe.view(batch_size, qlen, num_q_heads, self.qk_rope_head_dim)
             o = torch.empty(
-                B,
-                num_q_heads,
+                *q_nope.shape[:-1],
                 self.kv_lora_rank,
                 dtype=decode.attn_out_dtype,
                 device=q_nope.device,
             )
             kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
-            # Expand per-request paged-KV to per-verify-token: each request's KV
-            # block is repeated qlen times (row r*qlen+t attends request r's
-            # committed prefix). Fully vectorized (no host loop).
-            old_indptr = decode.paged_kv_indptr
-            per_req_len = old_indptr[1:] - old_indptr[:-1]
-            dev = q_nope.device
-            row_req = torch.arange(per_req_len.shape[0], device=dev).repeat_interleave(
-                qlen
-            )
-            row_len = per_req_len[row_req]
-            new_indptr = torch.cat([old_indptr.new_zeros(1), row_len.cumsum(0)]).to(
-                torch.int32
-            )
-            total = int(new_indptr[-1].item())
-            within = torch.arange(total, device=dev, dtype=torch.int64) - new_indptr[
-                :-1
-            ].to(torch.int64).repeat_interleave(row_len)
-            src = (
-                old_indptr[row_req].to(torch.int64).repeat_interleave(row_len) + within
-            )
-            new_indices = decode.paged_kv_indices[src]
             mla_gluon = _get_mla_gluon()
             mla_gluon(
                 q_nope=q_nope,
                 q_pe=q_pe,
                 kv_c=kv_buffer,
                 o=o,
-                page_table=new_indices,
-                seq_info=new_indptr,
+                page_table=decode.paged_kv_indices,
+                seq_info=decode.paged_kv_indptr,
                 sm_scale=self.scale,
                 k_pe=None,
                 kv_pe_offset=self.kv_lora_rank,
                 use_2d_view=False,
                 kv_scale=1.0,
-                min_kv_seq_len=int(per_req_len.min()),
+                min_kv_seq_len=decode.min_kv_seq_len,
             )
-            return o, None
+            return o.view(num_tokens, num_q_heads, self.kv_lora_rank), None
 
         if type(q) is tuple:
             q = torch.cat(q, dim=-1)
RAM_PRE_EOF

cat > /tmp/aiter_mla_gluon_batch.diff <<'BATCH_EOF'
diff --git a/aiter/ops/triton/gluon/mla_gluon.py b/aiter/ops/triton/gluon/mla_gluon.py
index 39e93b78b..19d21f025 100644
--- a/aiter/ops/triton/gluon/mla_gluon.py
+++ b/aiter/ops/triton/gluon/mla_gluon.py
@@ -10,10 +10,10 @@
 #                        Fast path: when NUM_KV_SPLITS==1, stage-1 writes the
 #                        final output directly to O and stage-2 reduce is skipped.
 #   REGIME='bh16bn128' - bf16 Q + fp8 KV, BLOCK_H=16, BLOCK_N=128,
-#                        nhead <= 16, batch_size=1, NUM_KV_SPLITS=256.
-#                        2-D (batch, split) grid. Always splits + always
-#                        reduces. NHEAD < BLOCK_H masks OOB heads on Q load
-#                        and O store.
+#                        nhead <= 16, 1 <= batch_size <= 128, 2-D (batch, split)
+#                        grid, NUM_KV_SPLITS = max(1, min(256 // batch_size,
+#                        cdiv(min_kv_seq_len, BLOCK_N))). NHEAD < BLOCK_H masks
+#                        OOB heads on Q load and O store.
 #   REGIME='bh16bn64'  - bf16 Q + bf16 KV, BLOCK_H=16, BLOCK_N=64,
 #                        nhead <= 16, batch_size >= 1, 2-D (batch, split) grid,
 #                        NUM_KV_SPLITS = max(1, 256 // batch_size). Full decode
@@ -934,9 +934,15 @@ def mla_gluon(
         # NUM_KV_SPLITS >= 1). Each clamp below keeps NUM_KV_SPLITS <= min_kv_seq_len,
         if REGIME == "bh16bn128":
             assert (
-                batch_size == 1
-            ), f"mla_gluon[bh16bn128] requires batch_size=1, got {batch_size}"
-            NUM_KV_SPLITS = max(1, min(256 // (batch_size * qlen), min_kv_seq_len))
+                1 <= batch_size <= 128
+            ), f"mla_gluon[bh16bn128] requires 1 <= batch_size <= 128, got {batch_size}"
+            # Same split policy as bh16bn64: fill ~256 WGs (B * NUM_KV_SPLITS <=
+            # 256) but never split a sequence into more blocks than it has, so
+            # every split holds >= 1 BLOCK_N block (floor split size >= 1 keeps
+            # num_iter >= 1). For batch_size=1 + long ctx this still yields 256.
+            NUM_KV_SPLITS = max(
+                1, min(256 // (batch_size * qlen), triton.cdiv(min_kv_seq_len, BLOCK_N))
+            )
         else:  # bh16bn64
             # Fill ~256 WGs (total WGs = B * NUM_KV_SPLITS <= 256, one MI350 wave),
             # but never split a sequence into more blocks than it has: bound by the
BATCH_EOF

cat > /tmp/aiter_pr4474.diff <<'PR4474_EOF'
diff --git a/aiter/ops/triton/gluon/mla_gluon.py b/aiter/ops/triton/gluon/mla_gluon.py
index 64dc4bec418..d36f1fca3b7 100644
--- a/aiter/ops/triton/gluon/mla_gluon.py
+++ b/aiter/ops/triton/gluon/mla_gluon.py
@@ -164,6 +164,11 @@ def _mla_gluon(
     num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
     start_n = split_kv_start
 
+    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
+    if not WITHIN_2GB:
+        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
+        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)
+
     # early return with empty kv slice to save compute
     if split_kv_start >= split_kv_end:
         return
PR4474_EOF

# ---- apply source patches (order: vllm, then aiter batch, then aiter #4474) ----
apply_idem "$VLLM_DIR"  /tmp/pr50619_vllm.diff          "vLLM PR #50619 (K3 fp8 MLA verify)"
apply_one_of "$VLLM_DIR" "vLLM PR #50619 (rocm_aiter_mla.py)" \
    /tmp/pr50619_ram_post50476.diff /tmp/pr50619_ram_pre50476.diff
apply_idem "$AITER_DIR" /tmp/aiter_mla_gluon_batch.diff "aiter mla_gluon bh16bn128 batch relax"
apply_idem "$AITER_DIR" /tmp/aiter_pr4474.diff          "aiter PR #4474 (int64 KV stride)"

# ---- Triton 3.7.0 + tabulate ----
TRITON_ROCM_VERSION="${TRITON_ROCM_VERSION:-7.2.0}"
TRITON_VERSION="${TRITON_VERSION:-3.7.0}"
echo "[patch] current triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"
python -m pip install --extra-index-url "https://pypi.amd.com/triton/release/rocm-${TRITON_ROCM_VERSION}/simple/" "triton==${TRITON_VERSION}"
python -m pip install tabulate

# ---- verify + import sanity ----
python - <<'PY'
import inspect
import vllm, aiter, triton
import vllm.models.kimi_k3.nvidia.mla
import vllm.v1.attention.backends.mla.rocm_aiter_mla
import aiter.ops.triton.gluon.mla_gluon
from triton.experimental.gluon.language import PaddedSharedLayout
assert "cga_layout" in inspect.signature(PaddedSharedLayout.__init__).parameters, \
    f"triton {triton.__version__} lacks PaddedSharedLayout(cga_layout=...)"
print("[patch] IMPORT_OK  vllm", vllm.__version__, " triton", triton.__version__)
PY
echo "[patch] DONE"