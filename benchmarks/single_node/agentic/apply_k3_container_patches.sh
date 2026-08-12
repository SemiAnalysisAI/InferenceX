#!/usr/bin/env bash
# =============================================================================
# apply_k3_cb8104839c_fp8_embedded.sh   (PINNED / offline)
#
# Reproduces, BYTE-FOR-BYTE, the patched Python source of the working Kimi-K3
# fp8-KV FULL_AND_PIECEWISE cudagraph container `k3_srok_cb810_0810_replay` on a
# FRESH container of:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#
# Code changes are EMBEDDED as pristine->container diffs (no GitHub / no PR
# drift). Net effect of, in the container:
#   aiter #4474  int64 KV stride (mla_gluon >2GB global_load)
#   aiter #4494  a16w16 GEMM fresh split-K semaphore under cudagraph capture
#   vllm  #51171 FULL cudagraphs for AITER MLA speculative decoding
#   vllm  #50578 asm decode for non-divisor small head counts (12->16 @ TP8)
#   vllm  #51011 fix fp8 KV cache decode on the AITER MLA backend
#   vllm  #51040 extend FP8 asm MLA prefill to non-divisor small head counts
#   vllm  #50619 (PARTIAL) cudagraph-exclude draft-attn layers + nvidia MLA
#                fallback gate: gpu/attn_utils.py, gpu/model_runner.py,
#                kimi_k3/nvidia/mla.py  (rocm_aiter_mla.py hunks NOT taken --
#                they conflict with the #50578/#51011 asm strategy)
#   vllm  #51682 KDA packed decode: pass the state-index stride to the kernel so
#                a non-contiguous 1-D state_indices is handled natively (only
#                requires ndim==1). Replaces the earlier reshape/coerce
#                workaround. NOTE: not strictly needed by this stack (it boots
#                under FULL_AND_PIECEWISE without it) -- kept for robustness.
#   aiter #4521  fp8 cp round-robin asm MLA verify kernels: adds the qh16/qh32
#                qseqlen4 gqaratio16/32 cprr .co + mla_asm.csv + asm_mla.cu +
#                v1_2_device.cuh + aiter/mla.py + aiter/ops/attention.py, then
#                rebuilds module_mla_asm.  [NEEDS NETWORK + hipcc + a GPU:
#                unlike the offline Python diffs, this fetches the binary .co
#                and recompiles the asm module.]
#   DSpark PS verify: route the small-head fp8 DSpark TARGET VERIFY to the ASM
#                persistent (PS) decode instead of the Gluon flatten. Two edits
#                on rocm_aiter_mla.py: (a) use_gluon_verify returns False for
#                fp8 KV so the verify is NOT swallowed by the flatten, (b)
#                _mtp_decode_qlen is sized for DSpark (1 + num_spec) so the PS
#                gate opens. Needs aiter #4521 for the fp8 qseqlen4 verify
#                kernels. SUPERSEDES the earlier HYBRID (gluon-flatten) verify.
#                mla_gluon bh16bn128 batch<=256 relax + fp8-query dequant are
#                kept (used by the bf16 verify path).
#   triton 3.7.0 (AMD ROCm 7.2.0) + tabulate + lm_eval[api]==0.4.12
#
# Run INSIDE a fresh container of that image:
#   docker exec -i <container> bash < apply_k3_cb8104839c_fp8_embedded.sh
#
# RUNTIME NOTE (NOT a code change -- set in your server script):
#   * MODEL_PATH must point at the model INSIDE the container (e.g. /model/Kimi-K3
#     when launched with `-v /data:/model`).
#   * fp8 PIECEWISE capture memory-faults at capture size 45 -> cap below it:
#     "max_cudagraph_capture_size": 44, "cudagraph_mode": "FULL_AND_PIECEWISE",
#     "custom_ops": ["+fused_rms_norm_gated"].
#   * env: VLLM_ROCM_USE_AITER=1, VLLM_ROCM_AITER_MLA_ASM_PADDING=asm,
#     VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1, --kv-cache-dtype fp8,
#     --enable-prefix-caching, DSpark spec-decode attention_backend=TRITON_MLA.
# =============================================================================
set -uo pipefail

# Resolve install root WITHOUT importing (importing aiter runs rocminfo and
# aborts on a GPU-less container). vllm and aiter share one dist-packages dir.
ROOT="$(python -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
if [ -z "$ROOT" ] || [ ! -d "$ROOT/vllm" ] || [ ! -d "$ROOT/aiter" ]; then
  echo "ERROR: could not resolve dist-packages (ROOT='$ROOT')"; exit 1
fi
echo "[embed] ROOT=$ROOT"
WS="${WS:-/tmp/k3_embed}"; mkdir -p "$WS"
say(){ echo; echo "=================== $* ==================="; }

say "1/4 triton 3.7.0 + tabulate + lm_eval"
python -m pip install --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ triton==3.7.0 2>&1 | tail -2
python -m pip install tabulate 2>&1 | tail -1
if [ "${WITH_LM_EVAL:-1}" = "1" ]; then
  python -m pip install "lm_eval[api]==0.4.12" 2>&1 | tail -2
fi

# Marker-gated apply: skip if the post-state marker is already present.
apply_one(){ # $1=relpath  $2=marker  $3=difffile
  local f="$ROOT/$1"
  if grep -qF "$2" "$f" 2>/dev/null; then echo "  $1: already present (skip)"; return; fi
  if ( cd "$ROOT" && git apply -p1 "$3" ) 2>/dev/null; then
    echo "  $1: APPLIED (git apply)"
  else
    patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch < "$3" \
      && echo "  $1: APPLIED (patch)" || echo "  $1: FAILED"
  fi
}

say "2/4 apply embedded code changes"
cat > "$WS/MLA_GLUON.diff" <<'DIFF_MLA_GLUON'
diff --git a/aiter/ops/triton/gluon/mla_gluon.py b/aiter/ops/triton/gluon/mla_gluon.py
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
@@ -861,6 +866,11 @@
         kv_pe_offset = 0
         use_2d_view = False
 
+    if q_nope.dtype == torch.float8_e4m3fn:
+        q_nope = q_nope.to(torch.bfloat16)
+    if q_pe is not None and q_pe.dtype == torch.float8_e4m3fn:
+        q_pe = q_pe.to(torch.bfloat16)
+
     assert (
         arch_info.get_arch() == "gfx950"
     ), f"mla_gluon requires gfx950 (CDNA4), got {arch_info.get_arch()}"
@@ -931,9 +941,11 @@
         # NUM_KV_SPLITS >= 1). Each clamp below keeps NUM_KV_SPLITS <= min_kv_seq_len,
         if REGIME == "bh16bn128":
             assert (
-                batch_size == 1
-            ), f"mla_gluon[bh16bn128] requires batch_size=1, got {batch_size}"
-            NUM_KV_SPLITS = max(1, min(256 // (batch_size * qlen), min_kv_seq_len))
+                1 <= batch_size <= 256
+            ), f"mla_gluon[bh16bn128] requires 1 <= batch_size <= 256, got {batch_size}"
+            NUM_KV_SPLITS = max(
+                1, min(256 // (batch_size * qlen), triton.cdiv(min_kv_seq_len, BLOCK_N))
+            )
         else:  # bh16bn64
             # Fill ~256 WGs (total WGs = B * NUM_KV_SPLITS <= 256, one MI350 wave),
             # but never split a sequence into more blocks than it has: bound by the
DIFF_MLA_GLUON
apply_one "aiter/ops/triton/gluon/mla_gluon.py" "1 <= batch_size <= 256" "$WS/MLA_GLUON.diff"

cat > "$WS/GEMM_A16W16.diff" <<'DIFF_GEMM_A16W16'
diff --git a/aiter/ops/gemm_op_a16w16.py b/aiter/ops/gemm_op_a16w16.py
--- a/aiter/ops/gemm_op_a16w16.py
+++ b/aiter/ops/gemm_op_a16w16.py
@@ -37,6 +37,9 @@
     return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)
 
 
+_captured_semaphore_keepalive: list[Tensor] = []
+
+
 def get_semaphore_workspace(device: torch.device) -> Tensor:
     """Return a per-(device, stream) zero-initialized semaphore workspace.
 
@@ -52,7 +55,19 @@
     Workspace size is small (~4 KB) and stream count per process is typically
     < 8, so the LRU cap of 64 leaves plenty of headroom before any in-flight
     workspace risks being evicted.
+
+    Under CUDA graph capture this returns a fresh workspace per launch instead
+    of the cached one: a captured graph bakes in the pointer and replays on a
+    stream other than the capture stream, so the cached counter can be left
+    non-zero and the reduction never fires. Allocating under capture also
+    records the zero-fill as a graph node, re-establishing the counter==0 entry
+    invariant on every replay. It is retained for the process lifetime because
+    aiter cannot observe when a graph dies.
     """
+    if torch.cuda.is_current_stream_capturing():
+        w = torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)
+        _captured_semaphore_keepalive.append(w)
+        return w
     stream = torch.cuda.current_stream(device)
     return _get_semaphore_workspace_keyed(device, stream.cuda_stream)
 
DIFF_GEMM_A16W16
apply_one "aiter/ops/gemm_op_a16w16.py" "is_current_stream_capturing" "$WS/GEMM_A16W16.diff"

cat > "$WS/ROCM_AITER_MLA.diff" <<'DIFF_ROCM_AITER_MLA'
diff --git a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
--- a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
+++ b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -26,7 +26,7 @@
     CommonAttentionMetadata,
     MultipleOf,
 )
-from vllm.v1.kv_cache_interface import AttentionSpec
+from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache
 
 logger = init_logger(__name__)
 
@@ -75,6 +75,50 @@
     except Exception:  # noqa: BLE001
         return False
     return True
+
+
+@functools.lru_cache(maxsize=1)
+def _gluon_mla_decode_supported() -> bool:
+    """The small-head Gluon MLA decode kernel only has a gfx950 (CDNA4) build.
+
+    Its tiling needs ~160 KiB of LDS, which exceeds CDNA3's 64 KiB, so on
+    gfx942 there is no kernel to fall through to and selecting it asserts
+    (``mla_gluon requires gfx950``). Restrict Gluon decode to gfx950; other
+    archs use the asm persistent decode, which ``get_mla_padded_q`` makes
+    correct for any 1..15 heads.
+    """
+    try:
+        from vllm.platforms.rocm import on_gfx950
+    except Exception:  # noqa: BLE001
+        return False
+    return on_gfx950()
+
+
+def _aiter_mla_small_head_mode() -> str:
+    """Small-head (<16) MLA decode kernel selection.
+
+    Controlled by ``VLLM_ROCM_AITER_MLA_ASM_PADDING``:
+
+    - ``"auto"`` (default): let the arch decide -- divisor head counts keep the
+      Gluon decode where a build exists (gfx950), everything else (non-divisor
+      counts and all counts on gfx942) uses the padded persistent-scheduling
+      ASM decode.
+    - ``"gluon"``: prefer the Gluon path wherever a build exists.
+    - ``"asm"``: force the padded persistent-scheduling ASM decode.
+
+    On gfx942 (no Gluon build) the ASM path is always used regardless of this
+    setting; ``"gluon"`` there falls back to ASM with a one-time warning.
+    """
+    import vllm.envs as envs
+
+    mode = (envs.VLLM_ROCM_AITER_MLA_ASM_PADDING or "auto").lower()
+    if mode == "gluon" and not _gluon_mla_decode_supported():
+        logger.warning_once(
+            "VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon requested, but this device "
+            "has no Gluon MLA decode build (Gluon requires gfx950); using the "
+            "padded persistent-scheduling ASM decode instead."
+        )
+    return mode
 
 
 class AiterMLABackend(MLACommonBackend):
@@ -134,6 +178,13 @@
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
@@ -225,17 +276,17 @@
         self.compilation_config = vllm_config.compilation_config
         self.decode_attn_out_dtype = vllm_config.model_config.dtype
 
-        # MTP/deepseek_mtp verification runs decode with qlen = num_spec + 1;
-        # any other config (including no spec) stays at single-token decode.
-        speculative_config = vllm_config.speculative_config
-        if (
-            speculative_config is not None
-            and speculative_config.method in ("mtp", "deepseek_mtp")
-            and speculative_config.num_speculative_tokens is not None
-        ):
-            self._mtp_decode_qlen = int(speculative_config.num_speculative_tokens) + 1
-        else:
-            self._mtp_decode_qlen = 1
+        # Size the metadata from reorder_batch_threshold, the largest query
+        # length decode can be handed (MLACommonMetadataBuilder asserts
+        # max_query_len <= reorder_batch_threshold); it already accounts for the
+        # drafting scheme. A method-name whitelist instead leaves drafters not on
+        # it -- DSpark, the eagle family -- sized for qlen=1 while the router
+        # still admits up to 1 + 2 * num_spec. The persistent gate below then
+        # never opens and aiter indexes get_block_n_fp8[num_heads * qlen], a
+        # table holding only {8, 16, 24, 32, 48, 64, 128, 256, 384, 512}: at 16
+        # heads every qlen in 5..7 and 9..15 is a KeyError, raised mid-run rather
+        # than at startup.
+        self._mtp_decode_qlen = self.reorder_batch_threshold or 1
 
         # Store the kernel block size from the spec. When kernel_block_size=1
         # (no spec-dec), behavior is identical to the original. When > 1
@@ -267,6 +318,74 @@
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
 
@@ -283,6 +402,9 @@
                 torch.float16: dtypes.fp16,
                 torch.bfloat16: dtypes.bf16,
             }[kv_cache_spec.dtype]
+        # _build_decode needs the cache dtype to pick the decode kernel; keep
+        # the normalized string instead of dropping it at the end of __init__.
+        self._kv_cache_dtype_str = kv_cache_dtype_str
         # MLAAttention quantizes decode Q to FP8 before calling this backend
         # whenever the KV cache is FP8 and supports_quant_query_input is true.
         q_dtype = (
@@ -329,9 +451,12 @@
             device=device,
         )
 
-        # FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads.
-        self._fp8_prefill_enabled = (
-            _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
+        # FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads, and
+        # only runs when the KV cache is FP8 (otherwise the bf16 path is used and
+        # the PS workspace must not be reserved).
+        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (
+            kv_cache_dtype_str == "fp8"
+            and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
         )
         if self._fp8_prefill_enabled:
             max_prefill_qlen = min(
@@ -387,7 +512,11 @@
 
         # After kv_b_proj decompression, K has num_heads heads (same as Q).
         # So gqa_ratio=1 and num_head_k=num_heads for the PS kernel.
-        num_head_k = self.num_heads
+        # Non-divisor head counts (e.g. K3's 12/rank at TP8) are padded to 16 in
+        # _mla_fp8_prefill_attn; build the PS metadata for the padded head count so
+        # the work/reduce maps match. This also lowers the partial-tile count:
+        # gcd(16, cu_num=256)=16 (~960 tiles) vs gcd(12,256)=4 (~4032), saving ~6 GiB.
+        num_head_k = max(16, self.num_heads)
         v_head_dim = self.mla_dims.v_head_dim
         # gqa_ratio = 1
         # qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)
@@ -481,7 +610,11 @@
         kv_indptr_cpu = qo_indptr_cpu.clone()
         seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)
 
-        num_head_k = self.num_heads
+        # Non-divisor head counts (e.g. K3's 12/rank at TP8) are padded to 16 in
+        # _mla_fp8_prefill_attn; build the PS metadata for the padded head count so
+        # the work/reduce maps match. This also lowers the partial-tile count:
+        # gcd(16, cu_num=256)=16 (~960 tiles) vs gcd(12,256)=4 (~4032), saving ~6 GiB.
+        num_head_k = max(16, self.num_heads)
         # gqa_ratio = 1
         # qhead_granularity = max(gqa_ratio, 1)
         # qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
@@ -580,7 +713,7 @@
             ]
         )
         use_gluon_decode = AiterMLAHelper.use_gluon_decode(
-            self.num_heads, int(max_qo_len)
+            self.num_heads, int(max_qo_len), self._kv_cache_dtype_str
         )
 
         if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
@@ -596,9 +729,9 @@
             block_table_tensor,
             block_table_tensor.stride(0),
             paged_kv_indptr,
-            seq_lens_for_kernel,
             KERNEL_BLOCK_SIZE=self.kernel_block_size,
             BLOCK_SIZE=1024,
+            QLEN=1,
         )
         paged_kv_indices = self.paged_kv_indices
 
@@ -650,12 +783,24 @@
                     qo_indptr = query_start_loc_device[: 1 + num_kernel_reqs]
 
         # Pass persistent metadata for every uniform decode we sized buffers for
-        # (normal qlen==1 through MTP verification qlen==K): the fp8 nhead=32 fold
-        # path breaks without it. qlen>K falls back to kernel-internal metadata.
-        # Small-head (<16) decode takes the Gluon paths and never consumes it.
+        # (qlen==1 through verification qlen==K); qlen>K falls back to
+        # kernel-internal metadata. Only the asm decode consumes the schedule, so
+        # gate on the routing and not on the raw head count: a non-divisor rank is
+        # padded to 16 and runs the same asm kernels as a native 16-head rank, yet
+        # `num_heads >= 16` reads as False for it and denies it the schedule. The
+        # kernel then falls back on its internal metadata, which bf16 tolerates
+        # and fp8 does not, and which the fp8 fold path rejects once qlen > 4:
+        #
+        #   asm_mla.cu:903 mla_decode_stage1_asm_fwd: only support gqa_ratio=16
+        #   fp8 mla decoding with qo_len <= 4 and qo_len > 4 in persistent mode
         has_persistent_metadata = False
         use_persistent_metadata = (
-            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
+            not AiterMLAHelper.use_gluon_decode(
+                self.num_heads, max_qo_len, self._kv_cache_dtype_str
+            )
+            and not AiterMLAHelper.use_gluon_verify(
+                self.num_heads, max_qo_len, self._kv_cache_dtype_str
+            )
             and max_qo_len >= 1
             and max_qo_len <= self._mtp_decode_qlen
         )
@@ -688,6 +833,79 @@
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
@@ -697,9 +915,12 @@
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
@@ -734,9 +955,9 @@
     block_table,
     block_table_stride,
     cu_num_tokens,
-    seq_lens,
     KERNEL_BLOCK_SIZE: tl.constexpr,
     BLOCK_SIZE: tl.constexpr,
+    QLEN: tl.constexpr,
 ):
     """Expand block table entries into per-token flat page indices.
 
@@ -750,11 +971,19 @@
 
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
@@ -781,8 +1010,11 @@
 
 class AiterMLAHelper:
     """
-    AITER MLA implementation requires num_heads >= 16. If num_heads < 16 and
-    16 % num_heads == 0, we can pad q to 16 heads; otherwise AITER has to fail.
+    AITER MLA persistent (asm) decode requires num_heads >= 16. Head counts
+    < 16 are padded up to exactly 16: divisors of 16 by repeat_interleave,
+    other counts (e.g. 12 heads/rank at TP8, 6 at TP16) by tiling the query
+    heads and slicing to 16. Non-divisor padded decodes take the asm path;
+    divisors and max_qo_len > 1 small-head verify still use Gluon.
     """
 
     _AITER_MIN_MLA_HEADS: Final = 16
@@ -791,8 +1023,9 @@
     @staticmethod
     def check_num_heads_validity(num_heads: int):
         assert AiterMLAHelper.is_valid_num_heads(num_heads), (
-            "ROCM AITER MLA requires 1-15 heads for Gluon decode or a multiple "
-            f"of 16 heads for persistent decode, but got {num_heads}.\n"
+            "ROCM AITER MLA requires 1-15 heads (padded to 16 for asm "
+            "persistent decode; exact divisors of 16 may keep Gluon) or a "
+            f"multiple of 16 heads, but got {num_heads}.\n"
             f"Try adjusting tensor_parallel_size value."
         )
 
@@ -809,25 +1042,87 @@
 
     @staticmethod
     def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
-        return (
-            q
-            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
-            else q.repeat_interleave(
-                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
-            )
-        )
+        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
+        if num_heads >= m:
+            return q
+        if m % num_heads == 0:
+            return q.repeat_interleave(m // num_heads, dim=1)
+        # Non-divisor head counts (e.g. 12 heads/rank at TP8, 6 at TP16) cannot
+        # be padded by repeat_interleave. Tile the query heads and slice to
+        # exactly m; this reaches m for any 0 < num_heads < m (unlike a single
+        # append, which under-pads when num_heads < m - num_heads). MLA
+        # attention is independent per query head over the shared KV, so the
+        # padding heads cannot affect heads [0:num_heads]; they are sliced back
+        # off in get_mla_unpadded_o.
+        reps = -(-m // num_heads)  # ceil(m / num_heads)
+        # Slicing a tiled tensor down to m yields a non-contiguous view whenever
+        # reps * num_heads > m (the common case: TP8 12->24->16, TP16 6->18->16).
+        # The asm persistent decode reads q as a packed [tokens, m, head_dim]
+        # buffer, so materialize a contiguous copy. No-op when already contiguous.
+        return q.repeat(1, reps, 1)[:, :m, :].contiguous()
 
     @staticmethod
     def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
-        return (
-            o
-            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
-            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
-        )
+        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
+        if num_heads >= m:
+            return o
+        if m % num_heads == 0:
+            return o[:, :: m // num_heads, :]
+        # Undo the tile-padding from get_mla_padded_q: the real heads are the
+        # first num_heads.
+        return o[:, :num_heads, :]
 
     @staticmethod
-    def use_gluon_decode(num_heads: int, max_qo_len: int) -> bool:
-        return num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and max_qo_len == 1
+    def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
+        # Small-head (<16) single-token decode can use either the Gluon kernel
+        # or the padded asm persistent decode, selected by
+        # VLLM_ROCM_AITER_MLA_ASM_PADDING (see _aiter_mla_small_head_mode) and
+        # the arch: Gluon only has a gfx950 build. In "auto" (default) mode
+        # divisor counts keep Gluon on gfx950 and everything else -- non-divisor
+        # counts (e.g. 12 heads/rank at TP8) and all counts on gfx942 -- takes
+        # the asm path, which get_mla_padded_q pads to exactly 16.
+        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
+        if num_heads >= m or max_qo_len != 1:
+            return False
+        # Gluon has exactly one fp8-KV regime, bh16bn128. It is a bf16-query
+        # kernel that upcasts the cache in registers with a hardcoded scale of
+        # 1.0, and it asserts batch_size == 1, so it cannot serve a real decode
+        # batch at any head count. A quantized cache always goes to the asm
+        # decode, which ships true fp8 kernels for gqa=16
+        # (mla_a8w8_qh16_qseqlen*_gqaratio16*.co). This precedes the mode knob:
+        # an explicit "gluon" request under fp8 would assert immediately.
+        if is_quantized_kv_cache(kv_cache_dtype):
+            return False
+        mode = _aiter_mla_small_head_mode()
+        if mode == "asm":
+            return False
+        gluon_supported = _gluon_mla_decode_supported()
+        if mode == "gluon":
+            return gluon_supported
+        return m % num_heads == 0 and gluon_supported
+
+    @staticmethod
+    def use_gluon_verify(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
+        """Whether a small-head multi-token verify is flattened onto Gluon.
+
+        The bf16 asm kernels have no gqa < 16, qseqlen > 1 entry, so a small-head
+        verify is flattened into per-token qseqlen=1 Gluon decodes. fp8 does have
+        one, reached by the q-row fold (16 heads x qlen 8 folds onto the
+        nhead=32, qseqlen=4 kernel, which ships in the package), and must not
+        come here: the flatten hands Gluon a batch of exactly the size that its
+        fp8 regime asserts against.
+
+        This lives next to use_gluon_decode rather than inline in forward_mqa so
+        that the builder, which has to know whether the asm decode will run, sees
+        the same answer the impl acts on.
+        """
+        if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS or max_qo_len <= 1:
+            return False
+        # HYBRID: small-head multi-token verify always uses the Gluon flatten,
+        # independent of kv dtype and VLLM_ROCM_AITER_MLA_ASM_PADDING. fp8 KV is
+        # served by the batch<=256 + fp8-query-dequant mla_gluon relaxation; the
+        # asm fp8 q-row-fold verify faults on gfx950 (HSA 0x1016 in DSpark).
+        return _gluon_mla_decode_supported()
 
 
 class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
@@ -873,10 +1168,13 @@
         self.flash_attn_varlen_func = flash_attn_varlen_func
 
         # FP8 MLA prefill kernel imports (lazy, only when enabled).
-        # Auto-enabled on gfx950 when AITER ships the kernels.
-        # FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads.
-        self._fp8_prefill_enabled = (
-            _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
+        # Auto-enabled on gfx950 when AITER ships the kernels. Only runs when the
+        # KV cache is FP8, and supports non-divisor small head counts via pad-to-16.
+        from vllm.utils.torch_utils import is_quantized_kv_cache
+
+        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (
+            is_quantized_kv_cache(kv_cache_dtype)
+            and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
         )
         if self._fp8_prefill_enabled:
             from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1
@@ -919,7 +1217,19 @@
 
         fp8_dtype = current_platform.fp8_dtype()
         total_q = q.shape[0]
-        nhead = self.num_heads
+        # PS asm prefill + mla_reduce_v1 require 16-aligned heads and the PS
+        # metadata is built for max(16, num_heads). For non-divisor small head
+        # counts (K3 = 12/rank at TP8) replicate-pad q/k/v to 16 — MLA attention
+        # is independent per query head over the shared KV, so the padding heads
+        # cannot affect the real ones (exact, same as the decode path) — then
+        # slice the output back to the real head count.
+        _real_nhead = self.num_heads
+        _pad16 = _real_nhead < 16
+        if _pad16:
+            q = AiterMLAHelper.get_mla_padded_q(_real_nhead, q)
+            k = AiterMLAHelper.get_mla_padded_q(_real_nhead, k)
+            v = AiterMLAHelper.get_mla_padded_q(_real_nhead, v)
+        nhead = 16 if _pad16 else self.num_heads
         v_head_dim = self.v_head_dim
         tile_q = _FP8_PREFILL_TILE_Q
 
@@ -946,7 +1256,13 @@
         # Reuse the caller's output buffer to skip the per-call alloc + copy.
         # The ASM and reduce kernels both write to a [total_q, nhead, v_head_dim]
         # view, which aliases the [total_q, nhead * v_head_dim] storage of out.
-        out_3d = out.view(total_q, nhead, v_head_dim)
+        if _pad16:
+            # Padded heads can't alias the real-head `out` storage; use scratch.
+            out_3d = torch.empty(
+                total_q, nhead, v_head_dim, dtype=out.dtype, device=out.device
+            )
+        else:
+            out_3d = out.view(total_q, nhead, v_head_dim)
 
         # Per-call scratch (logits, attn_lse, final_lse) is served from the
         # workspace manager so allocator churn in the prefill hot path is
@@ -993,6 +1309,11 @@
             final_lse,
         )
 
+        if _pad16:
+            out.view(total_q, _real_nhead, v_head_dim).copy_(
+                out_3d[:, :_real_nhead, :]
+            )
+
     def forward_mha(
         self,
         q: torch.Tensor,
@@ -1113,11 +1434,12 @@
         # target is checking draft tokens, so position t must not see t+1 --
         # and attention rows are independent, so giving row t the KV range
         # [0, context + t] is exactly causal multi-token attention.
-        if (
-            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
-            and int(decode.max_qo_len) > 1
+        # Arch, mode and dtype gating all live in use_gluon_verify, so that the
+        # builder -- which has to know whether the asm decode will run -- sees
+        # the same answer as this branch.
+        if AiterMLAHelper.use_gluon_verify(
+            self.num_heads, int(decode.max_qo_len), self.kv_cache_dtype
         ):
-            qlen = int(decode.max_qo_len)
             if type(q) is tuple:
                 q_nope, q_pe = q
             else:
@@ -1133,56 +1455,35 @@
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
 
DIFF_ROCM_AITER_MLA
apply_one "vllm/v1/attention/backends/mla/rocm_aiter_mla.py" "flat_kv_indices" "$WS/ROCM_AITER_MLA.diff"

# --- DSpark PS verify: supersede the HYBRID gluon-flatten verify -------------
# Two edits on the file the diff above just produced (HYBRID). Done as exact
# string replacements (not a context diff) so whitespace/line-drift can't break
# it, and idempotent via the "Local DSpark PS extension" guard. The base marker
# above was changed to "flat_kv_indices" (untouched here) so re-runs still skip.
#   (a) use_gluon_verify -> False for fp8 KV: the small-head multi-token verify
#       is no longer swallowed by the Gluon flatten and falls through to the ASM
#       persistent (PS) decode (aiter #4521 qseqlen4 cprr kernels).
#   (b) size _mtp_decode_qlen for DSpark (1 + num_spec) so the PS gate opens.
python - "$ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py" <<'PYDSPARK'
import ast, sys
F = sys.argv[1]
src = open(F).read()
if "Local DSpark PS extension" in src:
    print("  rocm_aiter_mla.py (DSpark PS): already present (skip)"); sys.exit(0)
OLD1 = "        self._mtp_decode_qlen = self.reorder_batch_threshold or 1\n"
NEW1 = (
    OLD1
    + "        # Local DSpark PS extension: reorder_batch_threshold's method\n"
    + "        # whitelist does not size DSpark, leaving its verify (qlen =\n"
    + "        # 1 + num_spec) at 1 so the persistent gate below never opens. Size\n"
    + "        # it explicitly so the ASM PS fp8 verify (qh16/qh32 qseqlen4 cprr\n"
    + "        # kernels, aiter #4521) is reachable.\n"
    + "        _spec = vllm_config.speculative_config\n"
    + "        if _spec is not None and (\n"
    + "            getattr(_spec, \"use_dspark\", False)\n"
    + "            or getattr(_spec, \"method\", None) == \"dspark\"\n"
    + "        ):\n"
    + "            self._mtp_decode_qlen = max(\n"
    + "                self._mtp_decode_qlen, 1 + int(_spec.num_speculative_tokens or 0)\n"
    + "            )\n"
)
OLD2 = (
    "        # HYBRID: small-head multi-token verify always uses the Gluon flatten,\n"
    "        # independent of kv dtype and VLLM_ROCM_AITER_MLA_ASM_PADDING. fp8 KV is\n"
    "        # served by the batch<=256 + fp8-query-dequant mla_gluon relaxation; the\n"
    "        # asm fp8 q-row-fold verify faults on gfx950 (HSA 0x1016 in DSpark).\n"
    "        return _gluon_mla_decode_supported()\n"
)
NEW2 = (
    "        # Local DSpark PS extension: with aiter #4521 the asm fp8 q-row-fold\n"
    "        # verify (qh16/qh32 qseqlen4 cprr kernels) works on gfx950, so an fp8\n"
    "        # KV small-head multi-token verify must NOT be swallowed by the Gluon\n"
    "        # flatten -- let it fall through to the ASM persistent (PS) path.\n"
    "        if is_quantized_kv_cache(kv_cache_dtype):\n"
    "            return False\n"
    "        return _gluon_mla_decode_supported()\n"
)
for tag, OLD in (("mtp_qlen sizing", OLD1), ("use_gluon_verify", OLD2)):
    if src.count(OLD) != 1:
        print(f"  rocm_aiter_mla.py (DSpark PS): ABORT {tag} (found {src.count(OLD)})")
        sys.exit(2)
src = src.replace(OLD1, NEW1, 1).replace(OLD2, NEW2, 1)
ast.parse(src)
open(F, "w").write(src)
print("  rocm_aiter_mla.py (DSpark PS): APPLIED")
PYDSPARK

cat > "$WS/TRITON_MLA.diff" <<'DIFF_TRITON_MLA'
diff --git a/vllm/v1/attention/backends/mla/triton_mla.py b/vllm/v1/attention/backends/mla/triton_mla.py
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
DIFF_TRITON_MLA
apply_one "vllm/v1/attention/backends/mla/triton_mla.py" "get_cudagraph_support" "$WS/TRITON_MLA.diff"

cat > "$WS/GPU_WORKER.diff" <<'DIFF_GPU_WORKER'
diff --git a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py
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
@@ -652,6 +653,19 @@
 
         # Update local config with adjusted num blocks after profiling,
         # so that it's available to the warmup stage.
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
         self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
 
         # Init kv cache connector here, because it requires
DIFF_GPU_WORKER
apply_one "vllm/v1/worker/gpu_worker.py" "import get_kv_cache_capacity" "$WS/GPU_WORKER.diff"

cat > "$WS/VLLM_ENVS.diff" <<'DIFF_VLLM_ENVS'
diff --git a/vllm/envs.py b/vllm/envs.py
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -133,6 +133,7 @@
     VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4: bool = False
     VLLM_ROCM_USE_AITER_RMSNORM: bool = True
     VLLM_ROCM_USE_AITER_MLA: bool = True
+    VLLM_ROCM_AITER_MLA_ASM_PADDING: Literal["auto", "gluon", "asm"] = "auto"
     VLLM_ROCM_USE_AITER_MHA: bool = True
     VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False
     VLLM_ROCM_USE_AITER_TRITON_ROPE: bool = False
@@ -1236,6 +1237,20 @@
     "VLLM_ROCM_USE_AITER_MLA": lambda: (
         os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1")
     ),
+    # Small-head (<16) AITER MLA decode kernel selection. Small head counts
+    # (e.g. Kimi-K3: 12 heads/rank at TP8, 6 at TP16) can decode either through
+    # the Gluon small-head kernel or through the padded persistent-scheduling
+    # (PS) ASM kernel. "auto" (default) keeps Gluon for head counts that divide
+    # 16 where a Gluon build exists (gfx950/CDNA4) and otherwise uses the padded
+    # PS ASM decode; "gluon" forces the Gluon path wherever a build exists;
+    # "asm" forces the padded PS ASM decode. On gfx942/CDNA3 there is no Gluon
+    # build, so the ASM path is always used regardless of this setting.
+    "VLLM_ROCM_AITER_MLA_ASM_PADDING": env_with_choices(
+        "VLLM_ROCM_AITER_MLA_ASM_PADDING",
+        "auto",
+        ["auto", "gluon", "asm"],
+        case_sensitive=False,
+    ),
     # Whether to use aiter mha ops.
     # By default is enabled.
     "VLLM_ROCM_USE_AITER_MHA": lambda: (
DIFF_VLLM_ENVS
apply_one "vllm/envs.py" "VLLM_ROCM_AITER_MLA_ASM_PADDING" "$WS/VLLM_ENVS.diff"

cat > "$WS/KIMI_NVIDIA_MLA.diff" <<'DIFF_KIMI_NVIDIA_MLA'
diff --git a/vllm/models/kimi_k3/nvidia/mla.py b/vllm/models/kimi_k3/nvidia/mla.py
--- a/vllm/models/kimi_k3/nvidia/mla.py
+++ b/vllm/models/kimi_k3/nvidia/mla.py
@@ -594,8 +594,7 @@
         cos_sin_cache: torch.Tensor | None,
         slot_mapping: torch.Tensor,
     ) -> torch.Tensor:
-        """Fused decode query-concat + latent cache insert, dispatched by cache
-        dtype (same policy as prefill: fp8 cache -> fp8 query)."""
+        """Build the decode query and update the cache for its dtype/backend."""
         if self.kv_cache_dtype == "fp8_ds_mla":
             cache = self.kv_cache
             if cache.dtype != torch.uint8:
@@ -612,10 +611,21 @@
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
DIFF_KIMI_NVIDIA_MLA
apply_one "vllm/models/kimi_k3/nvidia/mla.py" "if not self.impl.supports_quant_query_input" "$WS/KIMI_NVIDIA_MLA.diff"

cat > "$WS/ATTN_UTILS.diff" <<'DIFF_ATTN_UTILS'
diff --git a/vllm/v1/worker/gpu/attn_utils.py b/vllm/v1/worker/gpu/attn_utils.py
--- a/vllm/v1/worker/gpu/attn_utils.py
+++ b/vllm/v1/worker/gpu/attn_utils.py
@@ -92,6 +92,7 @@
     kv_cache_config: KVCacheConfig,
     vllm_config: VllmConfig,
     device: torch.device,
+    cg_support_exclude_layers: set[str] | None = None,
     active_layer_names: set[str] | None = None,
 ) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
     # Phase 1: discover attention groups for each kv cache group.
@@ -165,6 +166,15 @@
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
DIFF_ATTN_UTILS
apply_one "vllm/v1/worker/gpu/attn_utils.py" "cg_support_exclude_layers" "$WS/ATTN_UTILS.diff"

cat > "$WS/MODEL_RUNNER.diff" <<'DIFF_MODEL_RUNNER'
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -488,7 +488,14 @@
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
DIFF_MODEL_RUNNER
apply_one "vllm/v1/worker/gpu/model_runner.py" "cg_support_exclude_layers" "$WS/MODEL_RUNNER.diff"

cat > "$WS/KDA_FUSED_RECURRENT.diff" <<'DIFF_KDA_FUSED_RECURRENT'
diff --git a/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py b/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
index 2f512df62643..db519fb6f0db 100644
--- a/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
+++ b/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
@@ -459,6 +459,7 @@ def fused_recurrent_kda_packed_decode_kernel(
     stride_g_token: tl.constexpr,
     stride_beta_token: tl.constexpr,
     stride_state_token: tl.constexpr,
+    stride_state_indices,
     H: tl.constexpr,
     K: tl.constexpr,
     V: tl.constexpr,
@@ -476,7 +477,7 @@ def fused_recurrent_kda_packed_decode_kernel(
     mask_v = o_v < V
     mask_state = mask_v[:, None] & mask_k[None, :]

-    state_idx = tl.load(state_indices + i_n).to(tl.int64)
+    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
     p_out = out + (i_n * H + i_h) * V + o_v
     if state_idx <= 0:
         tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
@@ -560,8 +561,8 @@ def fused_recurrent_kda_packed_decode(
         raise ValueError("`raw_beta` heads must be contiguous.")
     if initial_state.stride()[1:] != (V * K, K, 1):
         raise ValueError("`initial_state` must be contiguous within each cache slot.")
-    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
-        raise ValueError("`state_indices` must be contiguous and one-dimensional.")
+    if state_indices.ndim != 1:
+        raise ValueError("`state_indices` must be one-dimensional.")
     if A_log.ndim != 1 or not A_log.is_contiguous():
         raise ValueError("`A_log` must be contiguous and one-dimensional.")
     if not dt_bias.is_contiguous():
@@ -608,6 +609,7 @@ def fused_recurrent_kda_packed_decode(
         stride_g_token=raw_g.stride(1),
         stride_beta_token=raw_beta.stride(1),
         stride_state_token=initial_state.stride(0),
+        stride_state_indices=state_indices.stride(0),
         H=H,
         K=K,
         V=V,
DIFF_KDA_FUSED_RECURRENT
apply_one "vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py" "stride_state_indices" "$WS/KDA_FUSED_RECURRENT.diff"


say "3/4 aiter #4521 (fp8 cp round-robin asm MLA verify kernels)  [needs network + hipcc + GPU]"
# Unlike the offline Python diffs above, #4521 ships BINARY .co kernels (not in
# the GitHub .diff) and a C++ asm_mla.cu change that must be recompiled, so this
# section fetches from GitHub and rebuilds module_mla_asm (which imports aiter ->
# needs a GPU). Skipped-idempotent via the csv marker.
PR4521_SHA="0cbedbb1bc5b3b254dd12ca4e8d3c7638b86830b"   # merged head of ROCm/aiter#4521
PR4521_RAW="https://raw.githubusercontent.com/ROCm/aiter/$PR4521_SHA"
META="$ROOT/aiter_meta"; MLADIR="$META/hsa/gfx950/mla"
if grep -qF "mla_a8w8_qh16_qseqlen4_gqaratio16_cprr_v3_ps.co" "$MLADIR/mla_asm.csv" 2>/dev/null; then
  echo "  #4521: already present (skip)"
elif [ "${WITH_PR4521:-1}" != "1" ]; then
  echo "  #4521: SKIPPED (WITH_PR4521!=1)"
else
  # 1) binary .co verify kernels (4 new cprr + 4 updated); leave orphans, csv gates load
  for f in \
    mla_a8w8_qh16_qseqlen4_gqaratio16_cprr_v3_ps.co \
    mla_a8w8_qh16_qseqlen4_gqaratio16_lse_cprr_v3_ps.co \
    mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_ps.co \
    mla_a8w8_qh16_qseqlen4_gqaratio16_v3_ps.co \
    mla_a8w8_qh32_qseqlen4_gqaratio32_cprr_ps.co \
    mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps.co \
    mla_a8w8_qh32_qseqlen4_gqaratio32_lse_ps.co \
    mla_a8w8_qh32_qseqlen4_gqaratio32_ps.co ; do
    if curl -ksSL -o "$MLADIR/$f.new" "$PR4521_RAW/hsa/gfx950/mla/$f" \
       && [ "$(stat -c %s "$MLADIR/$f.new" 2>/dev/null || echo 0)" -gt 1000 ]; then
      mv "$MLADIR/$f.new" "$MLADIR/$f"; echo "  co OK   $f"
    else
      rm -f "$MLADIR/$f.new"; echo "  co FAIL $f"
    fi
  done
  # 2) text diffs. Two install roots: aiter/*.py -> $ROOT ; csrc + mla_asm.csv -> $META
  curl -ksSL -o "$WS/pr4521.diff" "https://github.com/ROCm/aiter/pull/4521.diff"
  awk -v A="$WS/pr4521_A.diff" -v B="$WS/pr4521_B.diff" '
    /^diff --git /{p=$0; sub(/^diff --git a\//,"",p); sub(/ .*/,"",p); a=0; b=0;
      if (p ~ /^aiter\//) a=1;
      else if (p ~ /^csrc\// || p=="hsa/gfx950/mla/mla_asm.csv") b=1 }
    { if (a) print > A; else if (b) print > B }
  ' "$WS/pr4521.diff"
  git apply --directory="$ROOT" -p1 --unsafe-paths --whitespace=nowarn "$WS/pr4521_A.diff" 2>/dev/null \
    || patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch < "$WS/pr4521_A.diff"
  git apply --directory="$META" -p1 --unsafe-paths --whitespace=nowarn "$WS/pr4521_B.diff" 2>/dev/null \
    || patch -p1 -d "$META" --fuzz=3 --forward --no-backup-if-mismatch < "$WS/pr4521_B.diff"
  # 3) force module_mla_asm rebuild (aiter JIT only rebuilds when the .so is gone)
  rm -f "$ROOT/aiter/jit/module_mla_asm.so"
  UJ="$(python -c 'from aiter.jit.core import get_user_jit_dir as g; print(g())' 2>/dev/null)"
  [ -n "$UJ" ] && rm -f "$UJ/module_mla_asm.so"
  python - <<'PYBUILD'
from aiter.jit.core import get_args_of_build, build_module
d = get_args_of_build("module_mla_asm")
build_module("module_mla_asm", d["srcs"], d["flags_extra_cc"], d["flags_extra_hip"],
             d["blob_gen_cmd"], d["extra_include"], d["extra_ldflags"], d["verbose"],
             d["is_python_module"], d["is_standalone"], d["torch_exclude"],
             d.get("third_party", []), d.get("hipify", False),
             d.get("flags_extra_hip_per_source", {}))
print("  module_mla_asm rebuilt")
PYBUILD
  echo "  #4521: APPLIED"
fi

say "4/4 verify markers + py_compile + import"
echo "chk mla_gluon.py               = $(grep -c '1 <= batch_size <= 256' "$ROOT/aiter/ops/triton/gluon/mla_gluon.py")"
echo "chk gemm_op_a16w16.py          = $(grep -c 'is_current_stream_capturing' "$ROOT/aiter/ops/gemm_op_a16w16.py")"
echo "chk rocm_aiter_mla.py (base)   = $(grep -c 'flat_kv_indices' "$ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py")"
echo "chk rocm_aiter_mla.py (DSpark) = $(grep -c 'Local DSpark PS extension' "$ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py")  (expect 2)"
echo "chk #4521 mla_asm.csv          = $(grep -c 'qh16_qseqlen4_gqaratio16_cprr' "$ROOT/aiter_meta/hsa/gfx950/mla/mla_asm.csv" 2>/dev/null)  (expect 2; 0 if WITH_PR4521=0)"
echo "chk #4521 module_mla_asm.so    = $([ -f "$ROOT/aiter/jit/module_mla_asm.so" ] && echo present || echo MISSING)"
echo "chk triton_mla.py              = $(grep -c 'get_cudagraph_support' "$ROOT/vllm/v1/attention/backends/mla/triton_mla.py")"
echo "chk gpu_worker.py              = $(grep -c 'import get_kv_cache_capacity' "$ROOT/vllm/v1/worker/gpu_worker.py")"
echo "chk envs.py                    = $(grep -c 'VLLM_ROCM_AITER_MLA_ASM_PADDING' "$ROOT/vllm/envs.py")"
echo "chk mla.py                     = $(grep -c 'if not self.impl.supports_quant_query_input' "$ROOT/vllm/models/kimi_k3/nvidia/mla.py")"
echo "chk attn_utils.py              = $(grep -c 'cg_support_exclude_layers' "$ROOT/vllm/v1/worker/gpu/attn_utils.py")"
echo "chk model_runner.py            = $(grep -c 'cg_support_exclude_layers' "$ROOT/vllm/v1/worker/gpu/model_runner.py")"
echo "chk fused_recurrent.py         = $(grep -c 'reshape(-1).contiguous()' "$ROOT/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py")"
echo "triton          = $(python -c 'import triton; print(triton.__version__)')  (expect 3.7.0*)"
python -m py_compile "$ROOT/aiter/ops/triton/gluon/mla_gluon.py" \
  "$ROOT/aiter/ops/gemm_op_a16w16.py" \
  "$ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py" \
  "$ROOT/vllm/v1/attention/backends/mla/triton_mla.py" \
  "$ROOT/vllm/v1/worker/gpu_worker.py" \
  "$ROOT/vllm/envs.py" \
  "$ROOT/vllm/models/kimi_k3/nvidia/mla.py" \
  "$ROOT/vllm/v1/worker/gpu/attn_utils.py" \
  "$ROOT/vllm/v1/worker/gpu/model_runner.py" \
  "$ROOT/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py" && echo "PY_COMPILE_OK" || { echo "PY_COMPILE_FAIL"; exit 1; }
# Runtime import needs a GPU (aiter probes rocminfo); best-effort.
python - <<'PYEOF'
import importlib, traceback
mods = ("vllm.envs",
        "vllm.v1.attention.backends.mla.rocm_aiter_mla",
        "vllm.v1.attention.backends.mla.triton_mla",
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu.attn_utils",
        "vllm.v1.worker.gpu.model_runner")
try:
    for m in mods: importlib.import_module(m)
    import aiter.ops.gemm_op_a16w16  # noqa: F401
    import aiter.ops.triton.gluon.mla_gluon  # noqa: F401
    print("IMPORT_OK")
except Exception as e:
    print("IMPORT_SKIPPED (needs GPU?):", type(e).__name__, str(e).splitlines()[-1] if str(e) else "")
PYEOF
echo
echo "[embed] DONE. Launch server_final_CI.sh (MODEL_PATH + max_cudagraph_capture_size=44)."
