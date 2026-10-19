#!/usr/bin/env bash
set -eo pipefail
package=$(python3 -P -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
echo "Patching SGLang at $package"
python3 - "$package/srt/disaggregation" <<'PY'
import sys
from pathlib import Path

package = Path(sys.argv[1])
edits = [
    (
        "common/utils.py",
        """@dataclasses.dataclass(frozen=True)
class DCPTokenTransferPlan:
""",
        """def kv_entry_head_slice(
    src_tp: int,
    dst_tp: int,
    src_rank: int,
    dst_rank: int,
    src_heads: int,
    src_item_len: int,
    dst_item_len: int,
    page_size: int,
) -> Optional[Tuple[int, int, int, int, int]]:
    src_rank %= src_tp
    dst_rank %= dst_tp
    if src_heads <= 0 or src_item_len % page_size or dst_item_len % page_size:
        raise ValueError(
            f"KV head slicing needs per-token rows: heads={src_heads} "
            f"src_item_len={src_item_len} dst_item_len={dst_item_len} page_size={page_size}"
        )
    src_tok = src_item_len // page_size
    dst_tok = dst_item_len // page_size
    if src_tok % src_heads:
        raise ValueError(f"{src_tok} bytes per token do not split into {src_heads} heads")
    head_bytes = src_tok // src_heads
    if dst_tok % head_bytes:
        raise ValueError(
            f"decode token row {dst_tok} is not a whole number of {head_bytes}-byte heads"
        )
    dst_heads = dst_tok // head_bytes
    src_span = src_heads * src_tp
    dst_span = dst_heads * dst_tp
    if src_span != dst_span and src_heads == dst_heads == 1:
        raise ValueError(
            f"cannot infer the KV head count: both TP{src_tp} and TP{dst_tp} replicate heads"
        )
    total = min(src_span, dst_span)

    def first_head(tp: int, rank: int, heads: int) -> int:
        if heads * tp == total:
            return rank * heads
        if heads == 1 and tp % total == 0:
            return rank // (tp // total)
        raise ValueError(f"{heads} heads per rank over TP{tp} do not cover {total} KV heads")

    src_start = first_head(src_tp, src_rank, src_heads)
    dst_start = first_head(dst_tp, dst_rank, dst_heads)
    lo = max(src_start, dst_start)
    hi = min(src_start + src_heads, dst_start + dst_heads)
    if hi <= lo:
        return None
    if src_tp <= dst_tp and hi - lo != dst_heads:
        raise ValueError(
            f"prefill rank {src_rank}/TP{src_tp} holds heads [{src_start}, "
            f"{src_start + src_heads}) but decode rank {dst_rank}/TP{dst_tp} needs "
            f"[{dst_start}, {dst_start + dst_heads})"
        )
    return (
        (lo - src_start) * head_bytes,
        (lo - dst_start) * head_bytes,
        (hi - lo) * head_bytes,
        src_tok,
        dst_tok,
    )


def kv_head_slice_addrs(
    src_ptr: int,
    dst_ptr: int,
    src_pages,
    dst_pages,
    page_size: int,
    src_item_len: int,
    dst_item_len: int,
    head_slice: Tuple[int, int, int, int, int],
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    src_off, dst_off, _, src_tok, dst_tok = head_slice
    tokens = np.arange(page_size, dtype=np.uint64)[None, :]
    src_pages = np.asarray(src_pages, dtype=np.uint64)[:, None]
    dst_pages = np.asarray(dst_pages, dtype=np.uint64)[:, None]
    src = (
        np.uint64(src_ptr)
        + src_pages * np.uint64(src_item_len)
        + tokens * np.uint64(src_tok)
        + np.uint64(src_off)
    )
    dst = (
        np.uint64(dst_ptr)
        + dst_pages * np.uint64(dst_item_len)
        + tokens * np.uint64(dst_tok)
        + np.uint64(dst_off)
    )
    return src.ravel(), dst.ravel()


def transfer_entry_head_nums(pool, num_entries: int) -> List[int]:
    if num_entries <= 0:
        return []
    pool = getattr(pool, "full_kv_pool", pool)
    if (
        pool is None
        or getattr(pool, "kv_cache_layout", None) != "nhd"
        or getattr(pool, "use_hnd", False)
        or num_entries != 2 * getattr(pool, "layer_num", -1)
    ):
        return [0] * num_entries
    return [int(getattr(pool, "head_num", 0))] * num_entries


def state_entry_heads(kv_args, index: int) -> Optional[List[int]]:
    heads = getattr(kv_args, "state_entry_head_nums", None) or []
    entry = heads[index] if index < len(heads) else []
    ptrs = kv_args.state_data_ptrs[index] if index < len(kv_args.state_data_ptrs) else []
    if not entry or len(entry) != len(ptrs) or not all(entry):
        return None
    return list(entry)


@dataclasses.dataclass(frozen=True)
class DCPTokenTransferPlan:
""",
    ),
    (
        "prefill.py",
        """            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )
""",
        """            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )
        from sglang.srt.disaggregation.common.utils import transfer_entry_head_nums

        kv_args.kv_entry_head_nums = transfer_entry_head_nums(
            self.token_to_kv_pool, len(kv_data_ptrs) - num_draft_entries
        ) + transfer_entry_head_nums(draft_kv_pool, num_draft_entries)
        swa_pool = getattr(self.token_to_kv_pool, "swa_kv_pool", None)
        swa_ptrs = (
            list(self.token_to_kv_pool.get_state_buf_infos()[0])
            if swa_pool is not None
            and hasattr(self.token_to_kv_pool, "get_state_buf_infos")
            else None
        )
        kv_args.state_entry_head_nums = [
            (
                transfer_entry_head_nums(swa_pool, len(ptrs))
                if swa_ptrs and list(ptrs) == swa_ptrs
                else []
            )
            for ptrs in kv_args.state_data_ptrs
        ]
""",
    ),
    (
        "nixl/conn.py",
        """from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
""",
        """from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    kv_entry_head_slice,
    kv_head_slice_addrs,
    state_entry_heads,
""",
    ),
    (
        "nixl/conn.py",
        """    def _init_hetero_tp_prep_handle(
        self,
        peer_name: str,
        decode_kv_args: KVArgsRegisterInfo,
        src_mem_kind: str = "VRAM",
        dst_mem_kind: str = "VRAM",
    ):
""",
        """    def _init_hetero_tp_prep_handle(
        self,
        peer_name: str,
        decode_kv_args: KVArgsRegisterInfo,
        src_mem_kind: str = "VRAM",
        dst_mem_kind: str = "VRAM",
    ):
        page_size = self.kv_args.page_size
        src_ptrs = self.kv_args.kv_data_ptrs
        src_lens = self.kv_args.kv_item_lens
        n_src = len(src_ptrs)
        heads = list(getattr(self.kv_args, "kv_entry_head_nums", None) or [])
        if len(heads) != n_src or not all(heads):
            heads = [self.kv_args.kv_head_num] * n_src
        dst_index = self._build_transfer_dst_indices(
            peer_info=decode_kv_args,
            n_src=n_src,
            n_dst=len(decode_kv_args.dst_kv_item_lens),
        )
        plan = []
        for i, j in enumerate(dst_index):
            head_slice = kv_entry_head_slice(
                self.attn_tp_size,
                decode_kv_args.decode_tp_size,
                self.kv_args.engine_rank,
                decode_kv_args.decode_tp_rank,
                heads[i],
                src_lens[i],
                decode_kv_args.dst_kv_item_lens[j],
                page_size,
            )
            if head_slice is None:
                continue
            src_off, dst_off, nbytes, src_tok, dst_tok = head_slice
            if src_off % nbytes or src_tok % nbytes:
                raise RuntimeError(
                    f"KV entry {i}: head slice of {nbytes} bytes at {src_off} does not "
                    f"tile a {src_tok}-byte prefill token row"
                )
            plan.append((i, j, src_tok // nbytes, src_off // nbytes, dst_off, nbytes, dst_tok))
        if not plan:
            raise RuntimeError(f"No KV heads overlap with decode peer {peer_name}")
        groups_by_entry = {p[0]: p[2] for p in plan}
        src_groups = tuple(groups_by_entry.get(i, 1) for i in range(n_src))
        src_slots = [
            self.kv_args.kv_data_lens[i] // src_lens[i] for i in range(n_src)
        ]
        src_cache = self.__dict__.setdefault("prep_handles_entry_slice_src", {})
        if src_groups not in src_cache:
            src_device_id = _nixl_device_id(src_mem_kind, self.kv_args.gpu_id)
            src_runs = []
            src_bases = []
            base = 0
            for i in range(n_src):
                width = src_lens[i] // page_size // src_groups[i]
                count = src_slots[i] * page_size * src_groups[i]
                src_runs.append((src_ptrs[i], width, src_device_id, width, count))
                src_bases.append(base)
                base += count
            if base >= 2**31:
                raise RuntimeError(f"Sliced NIXL source dlist too large: {base}")
            src_cache[src_groups] = (
                self._prep_xfer_dlist("", self._pack_stride_descs(src_runs), src_mem_kind),
                np.asarray(src_bases, dtype=np.int64),
            )
        num_slots_dst = (
            decode_kv_args.dst_num_slots
            if decode_kv_args.dst_num_slots is not None
            else min(src_slots)
        )
        if len(plan) * num_slots_dst * page_size >= 2**31:
            raise RuntimeError("Sliced NIXL destination dlist too large")
        dst_device_id = _nixl_device_id(dst_mem_kind, decode_kv_args.gpu_id)
        dst_runs = [
            (
                decode_kv_args.dst_kv_ptrs[j] + dst_off,
                nbytes,
                dst_device_id,
                dst_tok,
                num_slots_dst * page_size,
            )
            for _, j, _, _, dst_off, nbytes, dst_tok in plan
        ]
        dst_handle = self._prep_xfer_dlist(
            peer_name, self._pack_stride_descs(dst_runs), dst_mem_kind
        )
        self.prep_handles_slice_dst[peer_name] = (
            dst_handle,
            num_slots_dst,
            (
                src_groups,
                np.asarray([p[0] for p in plan], dtype=np.int64),
                np.asarray([p[2] for p in plan], dtype=np.int64),
                np.asarray([p[3] for p in plan], dtype=np.int64),
            ),
        )

    def _init_hetero_tp_prep_handle_uniform(
        self,
        peer_name: str,
        decode_kv_args: KVArgsRegisterInfo,
        src_mem_kind: str = "VRAM",
        dst_mem_kind: str = "VRAM",
    ):
""",
    ),
    (
        "nixl/conn.py",
        """    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ):
""",
        """    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ):
        dst_handle, num_slots_dst, plan = self.prep_handles_slice_dst[peer_name]
        if not isinstance(plan, tuple):
            return self.send_kvcache_slice_uniform(
                peer_name, prefill_kv_indices, dst_kv_indices, notif
            )
        src_groups, entries, groups, group_idx = plan
        src_handle, src_bases = self.prep_handles_entry_slice_src[src_groups]
        page_size = self.kv_args.page_size
        tokens = np.arange(page_size, dtype=np.int64)
        src_tokens = (
            np.asarray(prefill_kv_indices, dtype=np.int64)[:, None] * page_size + tokens
        ).ravel()
        dst_tokens = (
            np.asarray(dst_kv_indices, dtype=np.int64)[:, None] * page_size + tokens
        ).ravel()
        if src_tokens.size != dst_tokens.size:
            raise RuntimeError(
                f"Sliced KV transfer index mismatch: prefill={src_tokens.size} "
                f"decode={dst_tokens.size}"
            )
        src_indices = (
            src_bases[entries][:, None]
            + src_tokens[None, :] * groups[:, None]
            + group_idx[:, None]
        ).ravel()
        dst_indices = (
            np.arange(len(entries), dtype=np.int64)[:, None] * (num_slots_dst * page_size)
            + dst_tokens[None, :]
        ).ravel()
        xfer_handle = self.agent.make_prepped_xfer(
            "WRITE",
            src_handle,
            src_indices.astype(np.int32),
            dst_handle,
            dst_indices.astype(np.int32),
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create prepped slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post prepped slice transfer")
        return xfer_handle

    def send_kvcache_slice_uniform(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ):
""",
    ),
    (
        "nixl/conn.py",
        """    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[List[int]],
""",
        """    def _send_state_head_slice(
        self,
        peer_name: str,
        src_ptrs: List[int],
        src_lens: List[int],
        src_heads: List[int],
        dst_ptrs: List[int],
        dst_lens: List[int],
        src_indices,
        dst_indices,
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int,
        src_layer_ids: List[int],
        dst_layer_ids: List[int],
    ):
        if len(src_indices) != len(dst_indices):
            raise RuntimeError(
                f"State index length mismatch: prefill={len(src_indices)}, "
                f"dst={len(dst_indices)}"
            )
        if len(dst_lens) != len(dst_ptrs):
            raise RuntimeError("Sliced state transfer needs decode item lengths")
        page_size = self.kv_args.page_size
        src_addrs, dst_addrs, lens = [], [], []
        for i, j in build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_ptrs),
            len(dst_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        ):
            head_slice = kv_entry_head_slice(
                self.attn_tp_size,
                decode_tp_size,
                self.kv_args.engine_rank,
                decode_tp_rank,
                src_heads[i],
                src_lens[i],
                dst_lens[j],
                page_size,
            )
            if head_slice is None:
                continue
            src, dst = kv_head_slice_addrs(
                src_ptrs[i],
                dst_ptrs[j],
                src_indices,
                dst_indices,
                page_size,
                src_lens[i],
                dst_lens[j],
                head_slice,
            )
            src_addrs.append(src)
            dst_addrs.append(dst)
            lens.append(np.full(src.shape, head_slice[2], dtype=np.uint64))
        if not src_addrs:
            return None
        lens = np.concatenate(lens)
        src_reqs = np.column_stack(
            (
                np.concatenate(src_addrs),
                lens,
                np.full(lens.shape, _nixl_device_id("VRAM", self.kv_args.gpu_id), dtype=np.uint64),
            )
        )
        dst_reqs = np.column_stack(
            (
                np.concatenate(dst_addrs),
                lens,
                np.full(lens.shape, _nixl_device_id("VRAM", dst_gpu_id), dtype=np.uint64),
            )
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create sliced state transfer")
        if self.agent.transfer(xfer_handle) == "ERR":
            raise Exception("KVSender failed to post sliced state transfer")
        return xfer_handle

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[List[int]],
""",
    ),
    (
        "nixl/conn.py",
        """                if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
""",
        """                if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                    state_heads = state_entry_heads(self.kv_args, i)
                    if state_heads is None or st != StateType.SWA:
                        raise RuntimeError(
                            f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                        )
                    h = self._send_state_head_slice(
                        peer_name,
                        src_ptrs,
                        src_lens,
                        state_heads,
                        dst_ptrs,
                        dst_lens,
                        src_indices,
                        dst_indices,
                        dst_gpu_id,
                        comp_notif,
                        decode_tp_size,
                        decode_tp_rank,
                        src_lids,
                        dst_lids,
                    )
                    if h is not None:
                        handles.append(h)
                    continue
""",
    ),
    (
        "mooncake/conn.py",
        """from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
""",
        """from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    kv_entry_head_slice,
    kv_head_slice_addrs,
    state_entry_heads,
""",
    ),
    (
        "mooncake/conn.py",
        """    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
""",
        """    def _send_head_sliced(
        self,
        mooncake_session_id: str,
        src_ptrs: List[int],
        src_lens: List[int],
        src_heads: List[int],
        dst_ptrs: List[int],
        dst_lens: List[int],
        src_indices,
        dst_indices,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        src_layer_ids: List[int],
        dst_layer_ids: List[int],
    ) -> int:
        page_size = self.kv_args.page_size

        def process(src_ptr, dst_ptr, src_len, dst_len, head_slice) -> int:
            src, dst = kv_head_slice_addrs(
                src_ptr,
                dst_ptr,
                src_indices,
                dst_indices,
                page_size,
                src_len,
                dst_len,
                head_slice,
            )
            if src.size == 0:
                return 0
            return self.engine.batch_transfer_sync(
                mooncake_session_id,
                src.tolist(),
                dst.tolist(),
                [head_slice[2]] * src.size,
            )

        futures = []
        for i, j in build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_ptrs),
            len(dst_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        ):
            head_slice = kv_entry_head_slice(
                self.attn_tp_size,
                dst_attn_tp_size,
                self.kv_args.engine_rank,
                dst_tp_rank,
                src_heads[i],
                src_lens[i],
                dst_lens[j],
                page_size,
            )
            if head_slice is not None:
                futures.append(
                    executor.submit(
                        process, src_ptrs[i], dst_ptrs[j], src_lens[i], dst_lens[j], head_slice
                    )
                )
        return self._await_transfer_futures(futures)

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_kv_item_lens: Optional[List[int]] = None,
    ):
        heads = list(getattr(self.kv_args, "kv_entry_head_nums", None) or [])
        if (
            not dst_kv_item_lens
            or len(dst_kv_item_lens) != len(dst_kv_ptrs)
            or len(heads) != len(self.kv_args.kv_data_ptrs)
            or not all(heads)
            or not (self.kv_args.kv_layer_ids and dst_layer_ids)
        ):
            return self.send_kvcache_slice_uniform(
                mooncake_session_id,
                prefill_kv_indices,
                dst_kv_ptrs,
                dst_kv_indices,
                dst_tp_rank,
                dst_attn_tp_size,
                dst_kv_item_len,
                executor,
                dst_layer_ids,
            )
        return self._send_head_sliced(
            mooncake_session_id,
            self.kv_args.kv_data_ptrs,
            self.kv_args.kv_item_lens,
            heads,
            dst_kv_ptrs,
            dst_kv_item_lens,
            prefill_kv_indices,
            dst_kv_indices,
            dst_tp_rank,
            dst_attn_tp_size,
            executor,
            self.kv_args.kv_layer_ids,
            dst_layer_ids,
        )

    def send_kvcache_slice_uniform(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
""",
    ),
    (
        "mooncake/conn.py",
        """                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                                target_rank_registration_info.dst_kv_layer_ids,
                            )
""",
        """                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                                target_rank_registration_info.dst_kv_layer_ids,
                                dst_kv_item_lens=target_rank_registration_info.dst_kv_item_lens,
                            )
""",
    ),
    (
        "mooncake/conn.py",
        """                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
""",
        """                    state_heads = state_entry_heads(self.kv_args, i)
                    if state_heads is None or st != StateType.SWA:
                        raise RuntimeError(
                            f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                        )
                    src_indices = list(indices)
                    dst_indices_local = list(dst_indices)
                    if len(src_indices) != len(dst_indices_local):
                        logger.warning(
                            f"len(prefill_state_indices) = {len(src_indices)}, len(dst_state_indices) = {len(dst_indices_local)}"
                        )
                    n = min(len(src_indices), len(dst_indices_local))
                    rc = (
                        self._send_head_sliced(
                            req.mooncake_session_id,
                            src_data_ptrs,
                            src_item_lens,
                            state_heads,
                            dst_data_ptrs,
                            dst_item_lens,
                            src_indices[:n],
                            dst_indices_local[:n],
                            target_rank_registration_info.dst_tp_rank,
                            target_rank_registration_info.dst_attn_tp_size,
                            executor,
                            src_state_layer_ids,
                            dst_state_layer_ids,
                        )
                        or rc
                    )
                    if self.request_status.get(req.room, KVPoll.Failed) == KVPoll.Failed:
                        return rc
                    continue
""",
    ),
]
for name, old, new in edits:
    path = package / name
    text = path.read_text()
    if text.count(new) == 1:
        print(f"already patched {path}")
    elif text.count(old) == 1:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
    else:
        sys.exit(f"{path}: anchor found {text.count(old)} times, expected 1:\n{old}")
PY
