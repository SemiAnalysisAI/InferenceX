#!/usr/bin/env bash
# =============================================================================
# apply_minimaxm3_container_patches.sh   (PINNED / offline)
#
# Enables the AITER gluon paged-attention decode kernel for MiniMax-M3 on a
# FRESH container of:
#   vllm/vllm-openai-rocm:v0.27.1
#
# WHY THE STOCK IMAGE CANNOT SERVE THIS CONFIG
#   At TP4 the recipe runs ROCM_AITER_FA with VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
#   (the sparse-attention path needs the shuffled layout) and EAGLE3 MTP with 3
#   speculative tokens. MTP makes every target decode a 4-token query, and the
#   image's decode path hard-asserts on exactly that combination:
#
#     vllm/v1/attention/backends/rocm_aiter_fa.py
#       assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
#           "Shuffle KV cache layout is not supported with sliding "
#           "window, sinks, or speculative decoding (multi-token decode).")
#
#   So the server dies on the first multi-token decode. The image already ships
#   the kernel that handles this (aiter/ops/triton/gluon/pa_decode_gluon.py);
#   only vLLM's dispatch is missing.
#
# NET EFFECT
#   vllm #52849  Enable AITER PA gluon decode for MiniMax-M3 MTP and dense
#                layers. Routes uniform multi-token decodes to pa_decode_gluon
#                instead of asserting, switches the shuffled KV cache to a
#                K/V-separated layout (+ stride order), advertises the 128-token
#                page only when gluon can serve it, and sizes the fp8 scales
#                from a layer the builder owns so the draft model's own KV cache
#                no longer mis-sizes them.
#                https://github.com/vllm-project/vllm/pull/52849
#   PREREQUISITE (upstream main, NOT part of #52849): gate
#                fused_qk_norm_rope_kvcache_supported() on the shuffle layout.
#                MiniMax-M3 uses per-head QK norm, so the QK-norm+RoPE+KVCache
#                fusion pass is live; upstream disables that fusion under the
#                shuffle layout so the write goes through the dedicated
#                reshape_and_cache_shuffle_triton path. #52849 was validated
#                with that gate in place, and v0.27.1 predates it.
#
# PORTING NOTE
#   #52849 is written against vllm main. 11 of its 13 hunks apply to v0.27.1
#   unchanged; two are hand-ported because v0.27.1 differs structurally:
#     * _split_kv_cache does not exist yet -- v0.27.1 inlines the same
#       transpose/split at three call sites, so the helper is hoisted (exactly
#       as main has it) and those sites now call it.
#     * the KV block zeroer in vllm/v1/worker/utils.py has different
#       surrounding code, so the outer-dim classification fix is re-expressed
#       against v0.27.1's context.
#   The result is byte-identical to the PR's own post-image in every function
#   the PR touches (verified by diffing against the PR head).
#
# Run INSIDE the container, before `vllm serve`:
#   bash apply_minimaxm3_container_patches.sh
#
# Idempotent: re-running is a no-op once the markers are present.
# =============================================================================
set -uo pipefail

IMAGE_EXPECTED="vllm/vllm-openai-rocm:v0.27.1"

die() { echo "[pa_gluon] ERROR: $*" >&2; exit 1; }

# Nothing in the result JSON, the recipe fingerprint or the result filename
# records that the engine was patched, and this script's stdout goes to the job
# log rather than to an artifact. Leave the provenance beside the result instead,
# the way the recipe already does for the server command line.
PATCH_ID="vllm-pr-52849-pa-gluon"
record() {
    local status="$1"
    local dir="${RESULT_DIR:-}"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
    printf 'image=%s patch=%s status=%s files=%s\n' \
        "$IMAGE_EXPECTED" "$PATCH_ID" "$status" \
        "vllm/v1/attention/backends/rocm_aiter_fa.py,vllm/v1/worker/utils.py" \
        >> "$dir/container_patches.txt"
}

# Resolve the install root WITHOUT importing (importing aiter runs rocminfo and
# aborts on a GPU-less container). vllm and aiter share one dist-packages dir.
ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))' 2>/dev/null)"
[ -n "$ROOT" ] && [ -d "$ROOT/vllm" ] || die "could not resolve dist-packages (ROOT='$ROOT')"
echo "[pa_gluon] ROOT=$ROOT"

FA_REL="vllm/v1/attention/backends/rocm_aiter_fa.py"
UTILS_REL="vllm/v1/worker/utils.py"
for rel in "$FA_REL" "$UTILS_REL"; do
    [ -f "$ROOT/$rel" ] || die "$ROOT/$rel not found (expected image $IMAGE_EXPECTED)"
done

# This patch only teaches vLLM to call the kernel; the kernel itself must
# already ship in the image's aiter. Fail loudly rather than at the first
# decode if the image ever drops it.
GLUON_SRC="$ROOT/aiter/ops/triton/gluon/pa_decode_gluon.py"
[ -f "$GLUON_SRC" ] || die "$GLUON_SRC not found: image ships no AITER gluon PA decode kernel"
for sym in pa_decode_gluon get_recommended_splits; do
    grep -q "^def ${sym}(" "$GLUON_SRC" || die "$GLUON_SRC does not define $sym()"
done

MARKER_FA="_PA_GLUON_MAX_QUERY_LEN"
MARKER_UTILS="d != block_dim and kv.stride(d)"
if grep -qF "$MARKER_FA" "$ROOT/$FA_REL" && grep -qF "$MARKER_UTILS" "$ROOT/$UTILS_REL"; then
    echo "[pa_gluon] already applied (skip)"
    record already-present
    exit 0
fi

WS="${WS:-/tmp/minimaxm3_pa_gluon}"
mkdir -p "$WS" || die "could not create $WS"

cat > "$WS/pa_gluon.diff" <<'DIFF_PA_GLUON'
--- a/vllm/v1/attention/backends/rocm_aiter_fa.py
+++ b/vllm/v1/attention/backends/rocm_aiter_fa.py
@@ -8,7 +8,11 @@
 import torch
 
 from vllm._aiter_ops import rocm_aiter_ops
-from vllm.config import VllmConfig, get_layers_from_vllm_config
+from vllm.config import (
+    VllmConfig,
+    get_current_vllm_config_or_none,
+    get_layers_from_vllm_config,
+)
 from vllm.config.cache import CacheDType
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention import Attention
@@ -33,9 +37,42 @@
 from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
 from vllm.v1.kv_cache_interface import AttentionSpec
 
+_PA_GLUON_MAX_QUERY_LEN = 4
+_PA_GLUON_MAX_QUERY_GROUP_SIZE = 64
+# Query group sizes the gluon paged-attention decode kernel is validated for: 8, 16
+_PA_GLUON_QUERY_GROUP_SIZES = (8, 16)
+
+# The kernel is only validated for this head size and kernel block size.
+_PA_GLUON_HEAD_SIZE = 128
+_PA_GLUON_BLOCK_SIZE = 128
+
+
+def _pa_gluon_supports(num_heads_q: int, num_heads_kv: int, head_size: int) -> bool:
+    """Whether the head config can use the PA decode gluon kernel.
+
+    Requires the shuffle KV cache layout (the kernel reads K/V in that layout)
+    plus a head config the kernel is validated for. Both the advertised kernel
+    block sizes and the decode dispatch go through this, so a config can never
+    be offered a 128-token page that gluon will then decline to serve.
+    """
+    return (
+        rocm_aiter_ops.is_shuffle_kv_cache_enabled()
+        and num_heads_kv > 0
+        and num_heads_q % num_heads_kv == 0
+        and num_heads_q // num_heads_kv in _PA_GLUON_QUERY_GROUP_SIZES
+        and head_size == _PA_GLUON_HEAD_SIZE
+    )
+
 _PARTITION_SIZE_ROCM = 256
 _CP_TOKENS_PER_ITER_ROCM = 32 * 1024
 if current_platform.is_rocm():
+    from aiter.ops.triton.gluon.pa_decode_gluon import (
+        get_recommended_splits,
+    )
+    from aiter.ops.triton.gluon.pa_decode_gluon import (
+        pa_decode_gluon as _pa_decode_gluon,
+    )
+
     from vllm.triton_utils import tl, triton
 
     def block_size(x, head_dim):
@@ -318,6 +355,7 @@
 @dataclass
 class AiterFlashAttentionDecodeMetadata:
     max_query_len: int
+    uniform_query_len: int | None
 
 
 @dataclass
@@ -474,10 +512,11 @@
             and self.scale.numel() == 1
             and is_quantized_kv_cache(self.vllm_config.cache_config.cache_dtype)
         ):
-            layers = get_layers_from_vllm_config(self.vllm_config, Attention)
-            first_layer_name = [k for k in layers][0]
+            # Size the scales from a layer this builder owns. The draft model
+            # runs its own builder over its own KV cache, so the first layer of
+            # the whole config can carry an unrelated block count.
             kv_cache_shape = self.vllm_config.compilation_config.static_forward_context[
-                first_layer_name
+                self.layer_names[0]
             ].kv_cache.shape
             num_blocks = kv_cache_shape[0]
             self.scale = torch.ones(
@@ -508,8 +547,15 @@
 
         decode_metadata = None
         if num_decodes > 0:
+            decode_max_query_len = query_lens_cpu[:num_decodes].max().item()
+            uniform_query_len = (
+                decode_max_query_len
+                if num_decode_tokens == num_decodes * decode_max_query_len
+                else None
+            )
             decode_metadata = AiterFlashAttentionDecodeMetadata(
-                max_query_len=query_lens_cpu[:num_decodes].max().item(),
+                max_query_len=decode_max_query_len,
+                uniform_query_len=uniform_query_len,
             )
 
         prefill_metadata = None
@@ -680,9 +726,27 @@
         """
         num_reqs = common_attn_metadata.num_reqs
         num_tokens = common_attn_metadata.num_actual_tokens
+        max_query_len = common_attn_metadata.max_query_len
+
+        # Uniform-decode assumption does not hold for the
+        # drafter's first forward after a target step: it inherits the target's
+        # per-request query lengths, so rows can be longer than gluon's limit or
+        # ragged. Those batches need the real split, which costs a sync.
+        # _PA_GLUON_MAX_QUERY_LEN only binds when gluon is the decode consumer,
+        # so test that rather than the shuffle layout alone.
+        if _pa_gluon_supports(self.num_heads_q, self.num_heads_kv, self.headdim) and (
+            max_query_len > _PA_GLUON_MAX_QUERY_LEN
+            or num_tokens != num_reqs * max_query_len
+        ):
+            return self.build(
+                common_prefix_len=0, common_attn_metadata=common_attn_metadata
+            )
 
         decode_metadata = AiterFlashAttentionDecodeMetadata(
-            max_query_len=common_attn_metadata.max_query_len,
+            max_query_len=max_query_len,
+            uniform_query_len=(
+                max_query_len if num_tokens == num_reqs * max_query_len else None
+            ),
         )
 
         return AiterFlashAttentionMetadata(
@@ -737,6 +801,21 @@
 
     @staticmethod
     def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
+        if not rocm_aiter_ops.is_shuffle_kv_cache_enabled():
+            return [16, 32]
+        # Only gluon serves 128-token pages; the pa_fwd_asm/ll4mi decode
+        # fallback is limited to 16 and 32. Advertise 128 only when gluon can
+        # run so selection never picks a page we cannot serve.
+        vllm_config = get_current_vllm_config_or_none()
+        if vllm_config is not None and vllm_config.model_config is not None:
+            mc = vllm_config.model_config
+            pc = vllm_config.parallel_config
+            if _pa_gluon_supports(
+                mc.get_num_attention_heads(pc),
+                mc.get_num_kv_heads(pc),
+                mc.get_head_size(),
+            ):
+                return [16, 32, 128]
         return [16, 32]
 
     @classmethod
@@ -771,9 +850,25 @@
     ) -> tuple[int, ...]:
         if block_size % 16 != 0:
             raise ValueError("Block size must be a multiple of 16.")
+
+        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
+            return (num_blocks, 2, block_size, num_kv_heads, head_size)
         # K and V are packed into the content dim: logical (B, H, N, 2*hs).
         return (num_blocks, num_kv_heads, block_size, 2 * head_size)
 
+    @staticmethod
+    def get_kv_cache_stride_order(
+        include_num_layers_dimension: bool = False,
+    ) -> tuple[int, ...]:
+        if not rocm_aiter_ops.is_shuffle_kv_cache_enabled():
+            # Physical layout matches the logical packed shape.
+            raise NotImplementedError
+        if include_num_layers_dimension:
+            raise NotImplementedError
+        # Hoist the K/V dim out so kv_cache[:, 0] and kv_cache[:, 1] are each a
+        # contiguous (num_blocks, block_size, num_kv_heads, head_size) range.
+        return (1, 0, 2, 3, 4)
+
     @classmethod
     def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
         from vllm.platforms.rocm import get_cdna_version
@@ -1065,8 +1160,7 @@
         # Whenever making a change in this method, please benchmark the
         # performance to make sure it does not introduce any overhead.
         num_actual_tokens = attn_metadata.num_actual_tokens
-        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
-        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
+        key_cache, value_cache = self._split_kv_cache(kv_cache)
 
         if is_quantized_kv_cache(self.kv_cache_dtype):
             key_cache = key_cache.view(current_platform.fp8_dtype())
@@ -1155,11 +1249,25 @@
             if num_decodes > 0:
                 assert attn_metadata.decode_metadata is not None
                 decode_max_query_len = attn_metadata.decode_metadata.max_query_len
+                decode_query_len = attn_metadata.decode_metadata.uniform_query_len
 
-                # Use unified_attention for speculative decoding (multi-token),
-                # sliding window, or sinks
-                # (pa_fwd_asm and paged_attention_v1 don't support sinks)
-                if (
+                # check if we can use the gluon paged-attention decode kernel
+                use_gluon = (
+                    _pa_gluon_supports(
+                        self.num_heads, self.num_kv_heads, self.head_size
+                    )
+                    and key_cache.shape[1] == _PA_GLUON_BLOCK_SIZE
+                    and decode_query_len is not None
+                    and decode_query_len <= _PA_GLUON_MAX_QUERY_LEN
+                    and decode_query_len * (self.num_heads // self.num_kv_heads)
+                    <= _PA_GLUON_MAX_QUERY_GROUP_SIZE
+                    and (decode_query_len == 1 or attn_metadata.causal)
+                )
+                # Use unified_attention for the decodes the paged kernels can't
+                # take: sliding window, sinks, or a multi-token batch that gluon
+                # declined (pa_fwd_asm and paged_attention_v1 don't support
+                # sinks).
+                if not use_gluon and (
                     self.sliding_window[0] != -1
                     or decode_max_query_len > 1
                     or self.sinks is not None
@@ -1282,21 +1390,6 @@
                     )
                 elif rocm_aiter_ops.is_shuffle_kv_cache_enabled():
                     _, num_heads, head_size = query.shape
-                    num_seqs = attn_metadata.seq_lens.shape[0]
-                    max_num_partitions = (
-                        attn_metadata.max_seq_len + _PARTITION_SIZE_ROCM - 1
-                    ) // _PARTITION_SIZE_ROCM
-                    tmp_out = torch.empty(
-                        (num_seqs, num_heads, max_num_partitions, head_size),
-                        dtype=query.dtype,
-                        device=query.device,
-                    )
-                    exp_sums = torch.empty(
-                        (num_seqs, num_heads, max_num_partitions),
-                        dtype=torch.float32,
-                        device=query.device,
-                    )
-                    max_logits = torch.empty_like(exp_sums)
                     num_blocks, block_size, num_kv_heads, _ = key_cache.shape
                     x = 16 // key_cache.element_size()
                     new_key_cache = key_cache.reshape(
@@ -1305,37 +1398,136 @@
                     new_value_cache = value_cache.reshape(
                         num_blocks, num_kv_heads, block_size // x, head_size, x
                     )
-                    k_qscale = (
-                        layer._k_scale
-                        if attn_metadata.k_scale is None
-                        else attn_metadata.k_scale
-                    )
-                    v_qscale = (
-                        layer._v_scale
-                        if attn_metadata.v_scale is None
-                        else attn_metadata.v_scale
-                    )
-                    rocm_aiter_ops.paged_attention_common(
-                        Q=query[:num_decode_tokens],
-                        K=new_key_cache,
-                        V=new_value_cache,
-                        tmp_out=tmp_out,
-                        max_logits=max_logits,
-                        exp_sums=exp_sums,
-                        max_seq_len=attn_metadata.max_seq_len,
-                        block_tables=attn_metadata.block_table[:num_decodes],
-                        context_lens=attn_metadata.seq_lens[:num_decodes],
-                        block_tables_stride0=attn_metadata.block_table[
-                            :num_decodes
-                        ].stride(0),
-                        scale=self.scale,
-                        K_QScale_hip=k_qscale,
-                        V_QScale_hip=v_qscale,
-                        K_QScale_asm=k_qscale,
-                        V_QScale_asm=v_qscale,
-                        out_=output[:num_decode_tokens],
-                        kv_cache_dtype=self.kv_cache_dtype,
-                    )
+
+                    if use_gluon:
+                        is_fp8_kv = is_quantized_kv_cache(self.kv_cache_dtype)
+                        # Per-tensor descale, as a float32 [1] tensor.
+                        k_scale_gluon = (
+                            layer._k_scale.reshape(1).to(torch.float32)
+                            if is_fp8_kv
+                            else None
+                        )
+                        v_scale_gluon = (
+                            layer._v_scale.reshape(1).to(torch.float32)
+                            if is_fp8_kv
+                            else None
+                        )
+                        compute_type = (
+                            current_platform.fp8_dtype() if is_fp8_kv else query.dtype
+                        )
+                        # The kernel folds the query positions into the group
+                        # dim, so the intermediate buffers are sized by the
+                        # combined extent.
+                        query_group_size = decode_query_len * (
+                            num_heads // num_kv_heads
+                        )
+
+                        sliding_window_int = (
+                            self.sliding_window[0] + 1
+                            if self.sliding_window[0] > 0
+                            else 0
+                        )
+                        if sliding_window_int > 0:
+                            max_context_partition_num = 1
+                            context_partition_size = 128
+                        else:
+                            max_context_partition_num = get_recommended_splits(
+                                num_decodes, num_kv_heads
+                            )
+                            context_partition_size = _PARTITION_SIZE_ROCM
+
+                        intermediate_shape = (
+                            num_decodes,
+                            num_kv_heads,
+                            max_context_partition_num,
+                            query_group_size,
+                        )
+                        exp_sums = torch.empty(
+                            intermediate_shape,
+                            dtype=torch.float32,
+                            device=query.device,
+                        )
+                        max_logits = torch.empty_like(exp_sums)
+                        temporary_output = torch.empty(
+                            (*intermediate_shape, head_size),
+                            dtype=output.dtype,
+                            device=query.device,
+                        )
+
+                        _pa_decode_gluon(
+                            output=output[:num_decode_tokens],
+                            query=query[:num_decode_tokens],
+                            key_cache=new_key_cache,
+                            value_cache=new_value_cache,
+                            context_lengths=attn_metadata.seq_lens[:num_decodes].to(
+                                torch.int32
+                            ),
+                            block_tables=attn_metadata.block_table[:num_decodes].to(
+                                torch.int32
+                            ),
+                            softmax_scale=self.scale,
+                            query_length=decode_query_len,
+                            max_context_partition_num=max_context_partition_num,
+                            context_partition_size=context_partition_size,
+                            compute_type=compute_type,
+                            query_scale=None,
+                            key_scale=k_scale_gluon,
+                            value_scale=v_scale_gluon,
+                            exp_sums=exp_sums,
+                            max_logits=max_logits,
+                            temporary_output=temporary_output,
+                            alibi_slopes=self.alibi_slopes,
+                            sinks=self.sinks,
+                            sliding_window=sliding_window_int,
+                            ps=True,
+                        )
+                    else:
+                        num_seqs = attn_metadata.seq_lens.shape[0]
+                        max_num_partitions = (
+                            attn_metadata.max_seq_len + _PARTITION_SIZE_ROCM - 1
+                        ) // _PARTITION_SIZE_ROCM
+                        tmp_out = torch.empty(
+                            (num_seqs, num_heads, max_num_partitions, head_size),
+                            dtype=query.dtype,
+                            device=query.device,
+                        )
+                        exp_sums = torch.empty(
+                            (num_seqs, num_heads, max_num_partitions),
+                            dtype=torch.float32,
+                            device=query.device,
+                        )
+                        max_logits = torch.empty_like(exp_sums)
+                        k_qscale = (
+                            layer._k_scale
+                            if attn_metadata.k_scale is None
+                            else attn_metadata.k_scale
+                        )
+                        v_qscale = (
+                            layer._v_scale
+                            if attn_metadata.v_scale is None
+                            else attn_metadata.v_scale
+                        )
+                        rocm_aiter_ops.paged_attention_common(
+                            Q=query[:num_decode_tokens],
+                            K=new_key_cache,
+                            V=new_value_cache,
+                            tmp_out=tmp_out,
+                            max_logits=max_logits,
+                            exp_sums=exp_sums,
+                            max_seq_len=attn_metadata.max_seq_len,
+                            block_tables=attn_metadata.block_table[:num_decodes],
+                            context_lens=attn_metadata.seq_lens[:num_decodes],
+                            block_tables_stride0=attn_metadata.block_table[
+                                :num_decodes
+                            ].stride(0),
+                            scale=self.scale,
+                            K_QScale_hip=k_qscale,
+                            V_QScale_hip=v_qscale,
+                            K_QScale_asm=k_qscale,
+                            V_QScale_asm=v_qscale,
+                            out_=output[:num_decode_tokens],
+                            kv_cache_dtype=self.kv_cache_dtype,
+                        )
                 else:
                     _, num_heads, head_size = query.shape
                     nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
@@ -1385,6 +1577,17 @@
 
         return output
 
+    def _split_kv_cache(
+        self, kv_cache: torch.Tensor
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
+            # (B, 2, N, H, hs) -> two contiguous (B, N, H, hs), which is what
+            # the shuffle read/write kernels reinterpret in place.
+            key_cache, value_cache = kv_cache.unbind(1)
+            return key_cache, value_cache
+        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
+        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
+
     def do_kv_cache_update(
         self,
         layer: AttentionLayer,
@@ -1393,8 +1596,7 @@
         kv_cache: torch.Tensor,
         slot_mapping: torch.Tensor,
     ):
-        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
-        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
+        key_cache, value_cache = self._split_kv_cache(kv_cache)
 
         # key and value may be None in the case of cross attention. They are
         # calculated once based on the output from the encoder and then cached
@@ -1449,7 +1651,12 @@
         )
 
     def fused_qk_norm_rope_kvcache_supported(self):
-        return rocm_aiter_ops.is_enabled()
+        # Only fuse when shuffle layout is off; the shuffle write path uses a
+        # dedicated cache update, mirroring fused_rope_kvcache_supported.
+        return (
+            rocm_aiter_ops.is_enabled()
+            and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
+        )
 
     def do_qk_norm_rope_kvcache_update(
         self,
@@ -1466,7 +1673,7 @@
         kv_cache: torch.Tensor,
         layer_slot_mapping: torch.Tensor,
     ):
-        key_cache, value_cache = kv_cache.unbind(1)
+        key_cache, value_cache = self._split_kv_cache(kv_cache)
         rocm_aiter_ops.do_qk_norm_rope_kvcache_update(
             qkv=qkv,
             q_weight=q_weight,
@@ -1501,8 +1708,7 @@
         kv_cache: torch.Tensor,
         layer_slot_mapping: torch.Tensor,
     ):
-        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
-        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
+        key_cache, value_cache = self._split_kv_cache(kv_cache)
         flash_layout = True
 
         is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
--- a/vllm/v1/worker/utils.py
+++ b/vllm/v1/worker/utils.py
@@ -161,10 +161,14 @@
                 cur_page_el = kernel_block_el * ratio
 
                 block_stride_bytes = cur_bytes
+                # A dim that strides further than one block encloses the block
+                # axis, so it is iterated rather than zeroed as part of a page.
+                # Enclosing dims need not precede the block dim: a K/V-separated
+                # layout can sit K/V after it and still span the whole cache.
                 outer_dims = [
                     d
-                    for d in range(block_dim)
-                    if kv.stride(d) * el > block_stride_bytes
+                    for d in range(kv.ndim)
+                    if d != block_dim and kv.stride(d) * el > block_stride_bytes
                 ]
                 outer_strides = [kv.stride(d) * el for d in outer_dims]
                 for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
DIFF_PA_GLUON

# All-or-nothing: a half-applied backend would serve wrong numbers rather than
# fail, so anything other than a clean apply is fatal.
if ( cd "$ROOT" && git apply -p1 "$WS/pa_gluon.diff" ) 2>/dev/null; then
    echo "[pa_gluon] APPLIED (git apply)"
elif patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch \
        < "$WS/pa_gluon.diff" >/dev/null 2>&1; then
    echo "[pa_gluon] APPLIED (patch)"
else
    die "patch did not apply under $ROOT -- image contents differ from $IMAGE_EXPECTED"
fi

grep -qF "$MARKER_FA" "$ROOT/$FA_REL" || die "$FA_REL is missing $MARKER_FA after apply"
grep -qF "$MARKER_UTILS" "$ROOT/$UTILS_REL" || die "$UTILS_REL is missing the outer-dim fix after apply"
for rel in "$FA_REL" "$UTILS_REL"; do
    python3 -m py_compile "$ROOT/$rel" || die "$rel does not compile after apply"
done

record applied
echo "[pa_gluon] gluon PA decode enabled for ROCM_AITER_FA (vllm #52849)"
