#!/usr/bin/env bash
# Apply all K3 patches on top of a stock
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
# container: vLLM PR #51171 + aiter PR #4474 (KV int64 stride) + Triton 3.7.0/tabulate.
#
# Usage INSIDE the container:
#     source apply_k3_cb8104839_patches.sh     # or: bash apply_k3_cb8104839_patches.sh
#
# Idempotent (-N skips already-applied hunks). Safe to source (no set -e / exit / cd).

_K3_DP="${K3_DP:-/usr/local/lib/python3.12/dist-packages}"
echo "[k3-cb] target dist-packages: ${_K3_DP}"

patch -p1 -N -d "${_K3_DP}" <<'K3CB_EOF'
--- a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
+++ b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -134,6 +134,13 @@
     use_gluon_decode: bool = False
     # Whether persistent MLA metadata was computed
     has_persistent_metadata: bool = False
+    # Small-head multi-token verify: paged-KV metadata with one row per verify
+    # token holding that token's causal KV window, built in _build_decode so
+    # forward_mqa stays free of device->host syncs.
+    # flat_kv_indptr is [num_reqs * max_qo_len + 1]; flat_kv_indices is the
+    # whole persistent buffer, indexed through flat_kv_indptr.
+    flat_kv_indptr: torch.Tensor | None = None
+    flat_kv_indices: torch.Tensor | None = None
 
 
 @dataclass
@@ -267,6 +274,74 @@
         self.paged_kv_indices = torch.zeros(
             max_num_pages, dtype=torch.int32, device=device
         )
+
+        # Small-head (< 16) multi-token verify expands each request's paged-KV
+        # range into one row per verify token, each holding that token's causal
+        # window. reorder_batch_threshold is the longest query block the decode
+        # path admits, so it bounds the row count per request. Sizing the
+        # buffers here keeps the expansion at fixed addresses, which is what
+        # lets the mla_gluon call in forward_mqa be captured in a full CUDA
+        # graph.
+        #
+        # The flatten is selected by the *impl's* per-layer query head count, so
+        # the buffers are reserved for any multi-token decode block this group
+        # can admit rather than from this builder's own num_heads, which is not
+        # required to agree with it. That reserves them for >= 16-head
+        # deployments too, where mla_decode_fwd serves the block and never reads
+        # them.
+        self._flat_max_qo_len = max(1, int(self.reorder_batch_threshold or 1))
+        self._flat_kv_enabled = self._flat_max_qo_len > 1
+        if self._flat_kv_enabled:
+            # The rows write at most max_qo_len times the sum of the batch's
+            # sequence lengths. max_num_pages bounds that sum by assuming every
+            # request is max_model_len long at the same time, which needs many
+            # times more entries than the KV cache can hold. Without prefix
+            # caching no two requests share a slot, so the pool's own token
+            # capacity is the real bound.
+            #
+            # cache_config.kv_cache_size_tokens is that capacity,
+            # max_concurrency * max_model_len, and it is a genuine upper bound
+            # on the sum even though every group draws block ids from one
+            # shared pool: a group's block count for a request of L tokens is
+            # either constant in L or concave in L through the origin, so it is
+            # never below L / max_model_len of what a full-length request
+            # takes. Summing that over the pool gives exactly this figure.
+            # num_gpu_blocks * block_size counts only this group's slots and so
+            # overstates the bound on hybrid layouts, where the other groups'
+            # blocks come out of the same pool; it is kept as the fallback for
+            # engines that have not published the group-aware capacity.
+            cache_config = vllm_config.cache_config
+            flat_pages = max_num_pages
+            if not cache_config.enable_prefix_caching:
+                kv_capacity = cache_config.kv_cache_size_tokens
+                if not kv_capacity and cache_config.num_gpu_blocks:
+                    kv_capacity = (
+                        int(cache_config.num_gpu_blocks) * self.kernel_block_size
+                    )
+                if kv_capacity:
+                    flat_pages = min(flat_pages, int(kv_capacity))
+            self.flat_kv_indptr = torch.zeros(
+                max_num_reqs * self._flat_max_qo_len + 1,
+                dtype=torch.int32,
+                device=device,
+            )
+            self.flat_kv_indices = torch.zeros(
+                flat_pages * self._flat_max_qo_len, dtype=torch.int32, device=device
+            )
+            # [0, 1, ..., max_qo_len - 1]. Added to a request's context length
+            # this gives each verify row its own causal KV bound; materialised
+            # once so the per-step build allocates nothing.
+            self._flat_causal_offsets = torch.arange(
+                self._flat_max_qo_len, dtype=torch.int32, device=device
+            )
+            logger.info(
+                "AITER MLA small-head verify buffers allocated "
+                "(max_qo_len=%d, pages=%d of %d, %.1f MiB)",
+                self._flat_max_qo_len,
+                flat_pages,
+                max_num_pages,
+                self.flat_kv_indices.numel() * 4 / (1024 * 1024),
+            )
 
         from aiter import dtypes, get_mla_metadata_info_v1
 
@@ -596,9 +671,9 @@
             block_table_tensor,
             block_table_tensor.stride(0),
             paged_kv_indptr,
-            seq_lens_for_kernel,
             KERNEL_BLOCK_SIZE=self.kernel_block_size,
             BLOCK_SIZE=1024,
+            QLEN=1,
         )
         paged_kv_indices = self.paged_kv_indices
 
@@ -688,6 +763,79 @@
             )
             has_persistent_metadata = True
 
+        # Small-head multi-token verify: build the per-verify-token causal
+        # paged-KV view here, once per step, instead of once per MLA layer in
+        # forward_mqa. That removes four device->host syncs per layer (an
+        # .item(), two tensor-driven repeat_interleave calls and a .min()) plus
+        # a data-dependent allocation, all of which abort HIP graph capture.
+        flat_kv_indptr = None
+        flat_kv_indices = None
+        min_kv_seq_len = 1
+        if self._flat_kv_enabled and max_qo_len > 1:
+            qlen = int(max_qo_len)
+            assert qlen <= self._flat_max_qo_len, (
+                f"verify block {qlen} exceeds the reserved maximum "
+                f"{self._flat_max_qo_len}"
+            )
+            num_rows = num_kernel_reqs * qlen
+            # Row r * qlen + t is request r's verify token t. seq_lens counts
+            # the tokens scheduled in this step, so a request's KV range already
+            # spans its whole verify block and context_r = seq_len_r - qlen.
+            # Causal masking lets token t attend to KV positions
+            # [0, context_r + t], i.e. seq_len_r - (qlen - 1) + t entries, so
+            # only the last row of a block may see the full range. Rows clamp to
+            # zero for cudagraph padding requests, whose seq_len is 0.
+            per_req_len = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
+            row_len = (
+                (
+                    per_req_len.unsqueeze(1)
+                    - (qlen - 1)
+                    + self._flat_causal_offsets[:qlen]
+                )
+                .clamp_(min=0)
+                .flatten()
+            )
+            # Element 0 stays zero from the initial torch.zeros; assigning a
+            # Python scalar to it would be a blocking host->device copy.
+            self.flat_kv_indptr[1 : num_rows + 1].copy_(
+                row_len.cumsum(dim=0, dtype=torch.int32), non_blocking=True
+            )
+            # A replayed cudagraph reads seq_info out to its captured row count,
+            # which can exceed num_rows. Repeating the final offset rather than
+            # zeroing makes every such row report length 0 instead of a large
+            # negative one, the same reason paged_kv_indptr's tail above is
+            # filled with its last entry.
+            self.flat_kv_indptr[num_rows + 1 :].fill_(self.flat_kv_indptr[num_rows])
+            flat_kv_indptr = self.flat_kv_indptr[: num_rows + 1]
+            # One device->host read serves both uses below; a sync is legal here
+            # because the builder runs outside the captured region. Gluon turns
+            # min_kv_seq_len into its split count, so it has to be the shortest
+            # row actually submitted, not the shortest per-request length those
+            # rows were cut from.
+            min_kv_seq_len, total_entries = torch.stack(
+                (row_len.min(), self.flat_kv_indptr[num_rows])
+            ).tolist()
+            # flat_kv_indices is reserved from the KV pool's token capacity,
+            # which bounds this sum. Check it rather than let a bound that is
+            # wrong for some future layout corrupt memory silently.
+            assert total_entries <= self.flat_kv_indices.numel(), (
+                f"verify KV view needs {total_entries} entries but only "
+                f"{self.flat_kv_indices.numel()} are reserved"
+            )
+            # No need to clear flat_kv_indices: the kernel writes exactly the
+            # [flat_kv_indptr[row], flat_kv_indptr[row + 1]) range that
+            # mla_gluon reads back for that row.
+            _expand_page_indices_kernel[(num_rows,)](
+                self.flat_kv_indices,
+                block_table_tensor,
+                block_table_tensor.stride(0),
+                flat_kv_indptr,
+                KERNEL_BLOCK_SIZE=self.kernel_block_size,
+                BLOCK_SIZE=1024,
+                QLEN=qlen,
+            )
+            flat_kv_indices = self.flat_kv_indices
+
         attn_metadata = AiterMLADecodeMetadata(
             block_table=block_table_tensor,
             seq_lens=seq_lens_for_kernel,
@@ -697,9 +845,12 @@
             qo_indptr=qo_indptr,
             dcp_tot_seq_lens=dcp_tot_seq_lens_device,
             max_qo_len=max_qo_len,
+            min_kv_seq_len=min_kv_seq_len,
             use_gluon_decode=use_gluon_decode,
             attn_out_dtype=self.decode_attn_out_dtype,
             has_persistent_metadata=has_persistent_metadata,
+            flat_kv_indptr=flat_kv_indptr,
+            flat_kv_indices=flat_kv_indices,
         )
 
         return attn_metadata
@@ -734,9 +885,9 @@
     block_table,
     block_table_stride,
     cu_num_tokens,
-    seq_lens,
     KERNEL_BLOCK_SIZE: tl.constexpr,
     BLOCK_SIZE: tl.constexpr,
+    QLEN: tl.constexpr,
 ):
     """Expand block table entries into per-token flat page indices.
 
@@ -750,11 +901,19 @@
 
     When KERNEL_BLOCK_SIZE=K: block table entry b (covering K tokens)
     is expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
+
+    QLEN is the number of output rows per request: 1 for ordinary decode, and
+    the verify block length for the small-head multi-token verify expansion,
+    where output row ``r * QLEN + t`` is request ``r``'s verify token ``t`` and
+    takes the first ``cu_num_tokens[row + 1] - cu_num_tokens[row]`` tokens of
+    that request -- its causal window, since block_table lists a request's
+    blocks in ascending position order.
     """
-    req_idx = tl.program_id(0)
+    row_idx = tl.program_id(0)
+    req_idx = row_idx // QLEN
     row_ptr = block_table + req_idx * block_table_stride
-    start_idx = tl.load(cu_num_tokens + req_idx)
-    num_tokens = tl.load(seq_lens + req_idx)
+    start_idx = tl.load(cu_num_tokens + row_idx)
+    num_tokens = tl.load(cu_num_tokens + row_idx + 1) - start_idx
 
     offset = tl.arange(0, BLOCK_SIZE)
     for i in tl.range(0, num_tokens, BLOCK_SIZE):
@@ -1117,7 +1276,6 @@
             self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
             and int(decode.max_qo_len) > 1
         ):
-            qlen = int(decode.max_qo_len)
             if type(q) is tuple:
                 q_nope, q_pe = q
             else:
@@ -1133,56 +1291,35 @@
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
+            # The per-verify-token view -- row r*qlen+t reads request r's
+            # committed prefix plus verify tokens 0..t, i.e. its causal window --
+            # is built once per step in _build_decode, where device->host syncs
+            # are legal. Reading it back here keeps this path free of the syncs
+            # that previously aborted HIP graph capture.
+            assert decode.flat_kv_indptr is not None
+            assert decode.flat_kv_indices is not None
+            # A non-causal block would need the untruncated range instead, and
+            # cannot arrive here: this builder leaves
+            # supports_non_causal_multi_token_decode False, so
+            # MLACommonMetadataBuilder.build rejects causal=False before
+            # _build_decode ever runs.
+            assert attn_metadata.causal, (
+                "AITER MLA small-head verify flatten is causal-only"
+            )
             mla_gluon = _get_mla_gluon()
             mla_gluon(
                 q_nope=q_nope,
                 q_pe=q_pe,
                 kv_c=kv_buffer,
                 o=o,
-                page_table=new_indices,
-                seq_info=new_indptr,
+                page_table=decode.flat_kv_indices,
+                seq_info=decode.flat_kv_indptr,
                 sm_scale=self.scale,
                 k_pe=None,
                 kv_pe_offset=self.kv_lora_rank,
                 use_2d_view=False,
                 kv_scale=1.0,
-                min_kv_seq_len=int(row_len.min()),
+                min_kv_seq_len=decode.min_kv_seq_len,
             )
             return o, None
 
--- a/vllm/v1/attention/backends/mla/triton_mla.py
+++ b/vllm/v1/attention/backends/mla/triton_mla.py
@@ -6,6 +6,7 @@
 import torch
 
 import vllm.envs as envs
+from vllm.config import VllmConfig
 from vllm.config.cache import CacheDType
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention.mla_attention import (
@@ -25,6 +26,7 @@
     MultipleOf,
 )
 from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
+from vllm.v1.kv_cache_interface import KVCacheSpec
 from vllm.v1.worker.workspace import (
     current_workspace_manager,
     is_workspace_manager_initialized,
@@ -54,6 +56,34 @@
     # Non-causal DSpark block is flattened to one decode row per query token in
     # forward_mqa, so no intra-block causal masking is required.
     supports_non_causal_multi_token_decode: ClassVar[bool] = True
+
+    @classmethod
+    def get_cudagraph_support(
+        cls,
+        vllm_config: VllmConfig,
+        kv_cache_spec: KVCacheSpec,
+    ) -> AttentionCGSupport:
+        """Report UNIFORM_BATCH where a non-causal multi-token block is served.
+
+        ``_cudagraph_support`` is a class constant, so serving the DSpark
+        draft's (1 + num_spec) block through the decode path reports
+        UNIFORM_SINGLE_TOKEN_DECODE and, because the engine takes the minimum
+        over all attention groups, downgrades the *whole* engine off full
+        cudagraphs. ``forward_mqa`` flattens that block with
+        ``repeat_interleave`` on a Python int and performs no device->host
+        sync, so it does satisfy the UNIFORM_BATCH contract.
+
+        ``non_causal_multi_token_decode`` is a KV-cache-group property, not a
+        per-layer one: ``MLAAttentionSpec.merge`` ORs it over every layer in
+        the group, so a group holding both a draft and its target reports it
+        for both. That is the same predicate ``__init__`` below already uses to
+        raise ``reorder_batch_threshold``, so the two stay consistent, but it
+        does mean this lifts a causal target sharing the draft's KV cache group
+        as well.
+        """
+        if getattr(kv_cache_spec, "non_causal_multi_token_decode", False):
+            return AttentionCGSupport.UNIFORM_BATCH
+        return cls._cudagraph_support
 
     def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
         super().__init__(kv_cache_spec, layer_names, vllm_config, device)
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -64,6 +64,7 @@
 from vllm.utils.mem_constants import GiB_bytes
 from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
 from vllm.utils.torch_utils import set_random_seed
+from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
 from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
 from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
 from vllm.v1.outputs import (
@@ -653,6 +654,19 @@
         # Update local config with adjusted num blocks after profiling,
         # so that it's available to the warmup stage.
         self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
+        # num_gpu_blocks * block_size is not the pool's token capacity when a
+        # request occupies more than one KV cache group, which is why
+        # kv_cache_size_tokens exists. It is only ever filled in by the engine
+        # core and the front end, so the worker's copy stays None and anything
+        # sizing a buffer off the KV pool during warmup -- the AITER MLA verify
+        # view, for one -- silently falls back to a far looser bound. Fill it in
+        # here too; get_kv_cache_capacity is documented to give the same answer
+        # for the worker's config as for the scheduler's.
+        if kv_cache_config.kv_cache_groups:
+            (
+                self.cache_config.kv_cache_size_tokens,
+                self.cache_config.kv_cache_max_concurrency,
+            ) = get_kv_cache_capacity(self.vllm_config, kv_cache_config)
 
         # Init kv cache connector here, because it requires
         # `kv_cache_config`.
--- a/aiter/ops/triton/gluon/mla_gluon.py
+++ b/aiter/ops/triton/gluon/mla_gluon.py
@@ -156,6 +156,11 @@
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
K3CB_EOF
echo "[k3-cb] source patch exit=$? (with -N, nonzero usually means already applied)"

# ---- Triton 3.7.0 + tabulate ----
_TRITON_ROCM_VERSION="${TRITON_ROCM_VERSION:-7.2.0}"
_TRITON_VERSION="${TRITON_VERSION:-3.7.0}"
echo "[k3-cb] current triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"
if python -c 'import triton,sys; sys.exit(0 if triton.__version__.startswith("'"${_TRITON_VERSION}"'") else 1)' 2>/dev/null; then
  echo "[k3-cb] triton ${_TRITON_VERSION} already installed"
else
  python -m pip install --extra-index-url "https://pypi.amd.com/triton/release/rocm-${_TRITON_ROCM_VERSION}/simple/" "triton==${_TRITON_VERSION}"
fi
python -m pip install tabulate
echo "[k3-cb] new triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"

# ---- sanity: byte-compile patched files ----
for _f in \
  vllm/v1/attention/backends/mla/rocm_aiter_mla.py \
  vllm/v1/attention/backends/mla/triton_mla.py \
  vllm/v1/worker/gpu_worker.py \
  aiter/ops/triton/gluon/mla_gluon.py ; do
  if python -m py_compile "${_K3_DP}/${_f}" 2>/dev/null; then
    echo "[k3-cb] pycompile OK: ${_f}"
  else
    echo "[k3-cb] PYCOMPILE FAILED: ${_f}"
  fi
done
echo "[k3-cb] done"
unset _K3_DP _TRITON_ROCM_VERSION _TRITON_VERSION _f