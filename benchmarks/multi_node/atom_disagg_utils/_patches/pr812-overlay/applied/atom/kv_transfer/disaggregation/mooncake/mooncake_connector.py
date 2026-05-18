# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses Mooncake TransferEngine for RDMA-based push (WRITE) transfers of
KV cache data from producer (prefill) to consumer (decode) nodes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import msgpack
import msgspec
import torch
import zmq

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    ReqId,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.utils import get_open_port, make_zmq_path, zmq_socket_ctx
from atom.utils.network import get_ip
from aiter.dist.parallel_state import get_dp_group, get_tp_group

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# Mooncake availability check
# ---------------------------------------------------------------------------

_MOONCAKE_AVAILABLE = False
try:
    from mooncake.engine import TransferEngine

    _MOONCAKE_AVAILABLE = True
    logger.info("Mooncake TransferEngine loaded successfully")
except ImportError as _exc:
    # Self-heal: atom's subprocess startup strips the venv site-packages
    # from sys.path. Add it back and retry; if mooncake is in the venv,
    # this transparent fix means the import succeeds without rebuilding
    # the image.
    import os as _os
    import sys as _sys
    import traceback as _tb
    _venv_sp = "/opt/venv/lib/python3.12/site-packages"
    if _venv_sp not in _sys.path and _os.path.isdir(_venv_sp):
        _sys.path.append(_venv_sp)
        try:
            from mooncake.engine import TransferEngine  # noqa: F811
            _MOONCAKE_AVAILABLE = True
            logger.info(
                "Mooncake TransferEngine loaded after re-adding venv site-packages "
                "to sys.path"
            )
        except ImportError as _exc2:
            _exc = _exc2

    if not _MOONCAKE_AVAILABLE:
        logger.warning(
            "Mooncake is not available — KV cache disaggregation via mooncake "
            "will not work. Install the mooncake package to enable push-mode "
            "RDMA transfers."
        )
        logger.warning(
            "[mooncake-diag] ImportError=%r | sys.executable=%s | PYTHONPATH=%r | "
            "venv_sp_in_path=%s | sys.path(full)=%s",
            _exc, _sys.executable,
            _os.environ.get("PYTHONPATH"), (_venv_sp in _sys.path), _sys.path,
        )
        logger.warning("[mooncake-diag] full traceback:\n%s", _tb.format_exc())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOONCAKE_PING_INTERVAL_SECONDS = 5
MOONCAKE_MAX_PING_RETRIES = 100
MOONCAKE_DEFAULT_PROTOCOL = "rdma"
PREFILL_LOOKUP_TIMEOUT = 60
PREFILL_LOOKUP_POLL_INTERVAL = 0.01

# ZMQ side-channel message types
MSG_WRITE_REQUEST = b"write_request"
MSG_WRITE_DONE = b"write_done"
MSG_GET_META = b"get_meta"


# ---------------------------------------------------------------------------
# Metadata struct for bootstrap handshake
# ---------------------------------------------------------------------------


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Serializable metadata exchanged during the mooncake bootstrap."""

    engine_id: str
    rpc_port: int
    kv_caches_base_addr: list[int] | None = None
    num_blocks: int = 0
    block_len: int = 0
    is_v4: bool = False
    v4_block_base_addrs: list[int] | None = None
    v4_block_bpb: list[int] | None = None
    v4_slot_base_addrs: list[int] | None = None
    v4_slot_bps: list[int] | None = None
    num_slots: int = 0


def _port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return dp_rank * tp_size + tp_rank


# ===================================================================
# MooncakeConnectorScheduler — scheduler-side connector
# ===================================================================


class MooncakeConnectorScheduler(KVConnectorSchedulerBase):
    def __init__(self, config: Config) -> None:
        kv_transfer_config = config.kv_transfer_config
        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.handshake_port = get_open_port()
        self.engine_id = "None"
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.parallel_config.data_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.host_ip = get_ip()

        # Pending requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        params = seq.kv_transfer_params or {}

        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True

        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id

        for req_id, (req, block_ids, slot_idx) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            req.kv_transfer_params["local_slot_index"] = slot_idx
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Producer side: pass completed prefill block_ids to worker
        for req_id, (req, block_ids, slot_idx) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            req.kv_transfer_params["local_slot_index"] = slot_idx
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self._reqs_need_recv or self._reqs_need_save:
            logger.info(
                "[SCHEDULER] build_connector_meta: %d recv, %d save, " "id_map=%s",
                len(self._reqs_need_recv),
                len(self._reqs_need_save),
                meta.request_id_to_transfer_id,
            )
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        return meta

    def update_state_after_alloc(self, seq: Sequence) -> None:
        params = seq.kv_transfer_params or {}

        if not self.is_producer:
            transfer_id = params.get("transfer_id")
            if transfer_id is not None:
                self.transfer_id_to_request_id[transfer_id] = seq.id
                self.request_id_to_transfer_id[seq.id] = transfer_id

        slot_index = getattr(seq, "per_req_cache_group", -1)

        # Consumer side: queue for remote KV loading
        if params.get("do_remote_prefill"):
            assert (
                not self.is_producer
            ), "Only the decode (consumer) side handles do_remote_prefill"
            self._reqs_need_recv[seq.id] = (seq, seq.block_table, slot_index)
            params["do_remote_prefill"] = False
            params["local_slot_index"] = slot_index
            logger.info(
                "[SCHEDULER-CONSUMER] Queued req %s for remote KV recv "
                "(%d blocks, slot=%d), transfer_id=%s, remote_host=%s, "
                "remote_handshake_port=%s",
                seq.id,
                len(seq.block_table),
                slot_index,
                params.get("transfer_id"),
                params.get("remote_host"),
                params.get("remote_handshake_port"),
            )

        # Producer side: queue block_ids for the write listener to look up
        if params.get("do_remote_decode"):
            assert self.is_producer, "Only the producer side handles do_remote_decode"
            self._reqs_need_save[seq.id] = (seq, seq.block_table, slot_index)
            logger.debug(
                "Queued req %s for KV save (%d blocks, slot=%d)",
                seq.id,
                len(seq.block_table),
                slot_index,
            )

    def request_finished(self, seq: Sequence) -> None:
        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": seq.block_table.copy(),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "local_slot_index": getattr(seq, "per_req_cache_group", -1),
        }

        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)


# ===================================================================
# MooncakeConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MooncakeConnector(KVConnectorBase):
    """Worker-side KV cache connector using Mooncake push-mode RDMA.

    Mooncake uses a push/WRITE model: the prefill (producer) node writes
    KV cache data directly into the decode (consumer) node's registered
    GPU memory via ``batch_transfer_sync_write``.
    """

    def __init__(self, config: Config) -> None:
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size

        kv_transfer_config = config.kv_transfer_config
        self.local_ip = get_ip()
        self._local_ping_port = get_open_port()

        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.is_consumer = not self.is_producer

        # Networking / service discovery config
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.proxy_ping_port = kv_transfer_config.get("proxy_ping_port", 36367)
        self.proxy_ip = kv_transfer_config.get("proxy_ip")
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.protocol = kv_transfer_config.get("protocol", MOONCAKE_DEFAULT_PROTOCOL)

        # Side channel port (ZMQ) — deterministic from config for proxy relay
        self.base_handshake_port = kv_transfer_config.get("handshake_port", 6301)
        self._side_channel_port = self.base_handshake_port + _port_offset(
            self.dp_rank, self.tp_rank, self.tp_size
        )

        # --- Mooncake TransferEngine initialization ---
        if not _MOONCAKE_AVAILABLE:
            raise RuntimeError(
                "Mooncake is not installed but kv_connector='mooncake' was requested. "
                "Install the mooncake package to use push-mode RDMA transfers."
            )

        # Determine which RDMA device this TP rank should use.
        # On AMD MI300X, each GPU has a paired RDMA NIC: GPU N → rdmaN.
        # Registering GPU memory with a non-local RDMA NIC fails with
        # EINVAL.  Pass the device name as a filter so Mooncake only
        # creates a context for the local NIC.
        ib_device = kv_transfer_config.get("ib_device", "")
        if not ib_device:
            ib_device = os.environ.get("ATOM_MOONCAKE_IB_DEVICE", "")
        if not ib_device:
            gpu_idx = torch.cuda.current_device()
            ib_device = f"rdma{gpu_idx}"
            logger.info(
                "Auto-selecting RDMA device %s for GPU %d (tp_rank=%d)",
                ib_device,
                gpu_idx,
                self.tp_rank,
            )

        self.transfer_engine = TransferEngine()
        ret = self.transfer_engine.initialize(
            self.local_ip,
            "P2PHANDSHAKE",
            self.protocol,
            ib_device,
        )
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine.initialize() failed (ret={ret}) "
                f"on ip={self.local_ip}, protocol={self.protocol}, "
                f"ib_device={ib_device}"
            )
        self.rpc_port = self.transfer_engine.get_rpc_port()
        self.engine_id = f"{self.local_ip}:{self.rpc_port}"
        logger.info(
            "Mooncake TransferEngine initialized: ip=%s, protocol=%s, "
            "ib_device=%s, rpc_port=%d",
            self.local_ip,
            self.protocol,
            ib_device,
            self.rpc_port,
        )

        # --- KV cache state (populated in register_kv_caches) ---
        self.kv_caches: dict[str, Any] | None = None
        self.kv_caches_base_addr: list[int] = []
        self._per_block_bytes_list: list[int] = []
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.block_len: int = config.kv_cache_block_size
        self.num_blocks: int = 0
        self._per_block_bytes: int = 0

        # --- V4 per-request state (populated in register_v4_kv_caches) ---
        self._has_slot_regions: bool = False
        self._v4_block_regions: list[tuple[int, int]] = (
            []
        )  # (base_addr, bytes_per_block)
        self._v4_slot_regions: list[tuple[int, int]] = []  # (base_addr, bytes_per_slot)
        self._gather_slot = None
        self._scatter_slot = None
        self._staging_base_addr: int = 0
        self._staging_slot_bytes: int = 0
        self._staging_pool_size: int = 0
        self._staging_free: list[int] = []
        self._staging_lock = threading.Lock()

        # --- Producer: completed prefill block_ids cache ---
        # Populated from ConnectorMetadata.reqs_to_save each step.
        # The write listener looks up block_ids here when consumer requests a write.
        self._completed_prefills: dict[ReqId, dict] = {}
        self._completed_prefills_lock = threading.Lock()
        self._completed_prefills_cv = threading.Condition(self._completed_prefills_lock)
        self._transfer_refcount: dict[ReqId, int] = {}
        self._transfer_refcount_lock = threading.Lock()

        # --- Consumer: pending receive tracking ---
        self._pending_recv: set[ReqId] = set()
        self._pending_recv_blocks: dict[ReqId, list[int]] = {}
        self._pending_recv_slots: dict[ReqId, int] = {}
        self._notification_port = get_open_port()

        # --- Completion tracking ---
        self.done_sending: set[str] = set()
        self.done_recving: set[str] = set()
        self._completion_lock = threading.Lock()

        # --- GPU memory fence: blocks pending coherence enforcement ---
        self._blocks_pending_fence: list[int] = []
        self._fence_lock = threading.Lock()

        # --- Transfer ID mapping (worker side) ---
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # --- Producer: thread pool for RDMA writes ---
        if self.is_producer:
            self._send_executor = ThreadPoolExecutor(
                max_workers=kv_transfer_config.get("num_worker_threads", 16),
                thread_name_prefix="mooncake-send-worker",
            )

        # --- ZMQ for metadata exchange ---
        self.zmq_context = zmq.Context()

        # --- Producer: persistent socket cache for write-done notifications ---
        self._notify_sockets: dict[str, zmq.Socket] = {}
        self._notify_sockets_lock = threading.Lock()

        # --- Msgspec encoder/decoder for bootstrap metadata ---
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

        # --- Service discovery ping (rank 0 only) ---
        if self.tp_rank == 0 and self.dp_rank == 0:
            self._ping_thread = threading.Thread(
                target=self._service_discovery_ping,
                args=(self.zmq_context,),
                daemon=True,
                name="mooncake-ping",
            )
            self._ping_thread.start()

    # -----------------------------------------------------------------
    # Service discovery
    # -----------------------------------------------------------------

    def _service_discovery_ping(self, zmq_context: zmq.Context) -> None:
        """Periodically register with the proxy (rank 0 only)."""
        grpc_endpoint = f"http://{self.request_address}/v1/completions"
        role_code = "P" if self.is_producer else "D"
        retry_count = 0
        msg_index = 1
        proxy_path = f"tcp://{self.proxy_ip}:{self.proxy_ping_port}"

        with zmq_context.socket(zmq.DEALER) as sock:
            sock.connect(proxy_path)

            while True:
                try:
                    registration_data = {
                        "type": "register",
                        "role": role_code,
                        "index": str(msg_index),
                        "request_address": grpc_endpoint,
                        "rpc_port": self.rpc_port,
                        "handshake_port": self.base_handshake_port,
                        "dp_size": self.dp_size,
                        "tp_size": self.tp_size,
                        "transfer_mode": "write",
                    }
                    sock.send(msgpack.dumps(registration_data))
                    logger.debug(
                        "Ping #%d sent to %s (role=%s)",
                        msg_index,
                        proxy_path,
                        role_code,
                    )
                    retry_count = 0

                except ConnectionRefusedError:
                    logger.info(
                        "Proxy connection refused: %s -> %s",
                        self.local_ip,
                        proxy_path,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error during ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected ping error: %s", e)
                    retry_count += 1
                    if retry_count >= MOONCAKE_MAX_PING_RETRIES:
                        logger.error(
                            "Ping failed after %d retries, aborting",
                            MOONCAKE_MAX_PING_RETRIES,
                        )
                        raise RuntimeError(
                            f"Service discovery ping failed after "
                            f"{retry_count} retries"
                        ) from e

                finally:
                    time.sleep(MOONCAKE_PING_INTERVAL_SECONDS)
                    msg_index += 1

    # -----------------------------------------------------------------
    # KVConnectorBase: register_kv_caches
    # -----------------------------------------------------------------
    _MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024

    def register_kv_caches(
        self, kv_caches: dict[str, Any], transfer_tensors: Any = None
    ) -> None:
        """Register KV cache tensors with the Mooncake TransferEngine."""
        self.kv_caches = kv_caches

        if transfer_tensors is None:
            logger.warning(
                "register_kv_caches called without transfer_tensors; "
                "RDMA transfers will not be available."
            )
            return

        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        tt: KVTransferTensors = transfer_tensors

        self._has_slot_regions = (
            len(tt.slot_regions) > 0 or tt.staging_region is not None
        )
        self.num_blocks = tt.num_blocks
        self._gather_slot = tt.gather_slot
        self._scatter_slot = tt.scatter_slot

        if tt.staging_region is not None:
            self._staging_base_addr = tt.staging_region.base_addr
            self._staging_slot_bytes = tt.staging_region.unit_bytes
            self._staging_pool_size = tt.staging_pool_size
            self._staging_free = list(range(tt.staging_pool_size))

        # Populate block/slot region lists for transfer offset computation
        self._v4_block_regions = [(r.base_addr, r.unit_bytes) for r in tt.block_regions]
        self._v4_slot_regions = [(r.base_addr, r.unit_bytes) for r in tt.slot_regions]

        self.kv_caches_base_addr = [r.base_addr for r in tt.block_regions]
        self._per_block_bytes_list = [r.unit_bytes for r in tt.block_regions]

        # Chunk all regions for RDMA memory registration
        reg_ptrs: list[int] = []
        reg_sizes: list[int] = []

        all_regions = list(tt.block_regions) + list(tt.slot_regions)
        if tt.staging_region is not None:
            all_regions.append(tt.staging_region)
        for r in all_regions:
            offset = 0
            while offset < r.total_bytes:
                chunk = min(self._MAX_RDMA_CHUNK_BYTES, r.total_bytes - offset)
                reg_ptrs.append(r.base_addr + offset)
                reg_sizes.append(chunk)
                offset += chunk

        logger.info(
            "Registering %d RDMA chunks (%d block regions, %d slot regions, "
            "max_chunk=%.2f GiB)",
            len(reg_ptrs),
            len(tt.block_regions),
            len(tt.slot_regions),
            self._MAX_RDMA_CHUNK_BYTES / (1024**3),
        )

        ret = self.transfer_engine.batch_register_memory(reg_ptrs, reg_sizes)
        if ret != 0:
            logger.error(
                "batch_register_memory FAILED (ret=%d), "
                "trying individual registration as fallback...",
                ret,
            )
            for ptr, sz_bytes in zip(reg_ptrs, reg_sizes):
                r = self.transfer_engine.register_memory(ptr, sz_bytes)
                if r != 0:
                    logger.error(
                        "  register_memory FAILED ptr=0x%x size=%d ret=%d",
                        ptr,
                        sz_bytes,
                        r,
                    )
        else:
            logger.info("batch_register_memory OK (%d chunks)", len(reg_ptrs))

        # Build metadata for bootstrap exchange
        if self._has_slot_regions:
            self._local_metadata = MooncakeAgentMetadata(
                engine_id=self.engine_id,
                rpc_port=self.rpc_port,
                num_blocks=tt.num_blocks,
                block_len=self.block_len,
                is_v4=True,
                v4_block_base_addrs=[b for b, _ in self._v4_block_regions],
                v4_block_bpb=[bpb for _, bpb in self._v4_block_regions],
                v4_slot_base_addrs=[b for b, _ in self._v4_slot_regions],
                v4_slot_bps=[bps for _, bps in self._v4_slot_regions],
                num_slots=tt.num_slots,
            )
        else:
            self._local_metadata = MooncakeAgentMetadata(
                engine_id=self.engine_id,
                rpc_port=self.rpc_port,
                kv_caches_base_addr=self.kv_caches_base_addr,
                num_blocks=tt.num_blocks,
                block_len=self.block_len,
            )

        logger.info(
            "Mooncake KV registration complete: role=%s, engine_id=%s, "
            "is_v4=%s, num_blocks=%d, block_regions=%d, slot_regions=%d",
            "PRODUCER" if self.is_producer else "CONSUMER",
            self.engine_id,
            self._has_slot_regions,
            tt.num_blocks,
            len(self._v4_block_regions),
            len(self._v4_slot_regions),
        )

        # Start side channel threads
        if self.is_producer:
            self._write_listener_thread = threading.Thread(
                target=self._write_listener,
                daemon=True,
                name="mooncake-write-listener",
            )
            self._write_listener_thread.start()
        else:
            self._notification_listener_thread = threading.Thread(
                target=self._notification_listener,
                daemon=True,
                name="mooncake-notify-listener",
            )
            self._notification_listener_thread.start()

    # -----------------------------------------------------------------
    # KVConnectorBase: start_load_kv
    # -----------------------------------------------------------------

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate KV transfers for pending requests.

        **Producer side**: Cache completed prefill block_ids from
        ``metadata.reqs_to_save`` so the write listener can look them up.

        **Consumer side**: For each pending recv request, connect to the
        producer's ZMQ side channel and send a write request with our
        memory addresses and block allocation.
        """
        if metadata is None:
            return

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        # Producer: cache block_ids + slot_index from completed prefills
        if self.is_producer:
            for req_id, meta in metadata.reqs_to_save.items():
                with self._completed_prefills_cv:
                    self._completed_prefills[req_id] = {
                        "block_ids": meta.local_block_ids,
                        "slot_index": meta.local_slot_index,
                    }
                    self._completed_prefills_cv.notify_all()
                logger.info(
                    "[PRODUCER] Cached %d prefill blocks (slot=%d) for req %s",
                    len(meta.local_block_ids),
                    meta.local_slot_index,
                    req_id,
                )
            return

        # Consumer: send write requests to producer
        if not metadata.reqs_to_recv:
            return

        logger.info(
            "[CONSUMER] start_load_kv: %d reqs_to_recv, id_map=%s",
            len(metadata.reqs_to_recv),
            metadata.request_id_to_transfer_id,
        )

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_tp_size = meta.tp_size
            if remote_tp_size != self.tp_size:
                remote_tp_rank = self.tp_rank % remote_tp_size
            else:
                remote_tp_rank = self.tp_rank
            remote_port = meta.remote_handshake_port + _port_offset(
                meta.remote_dp_rank, remote_tp_rank, remote_tp_size
            )
            remote_addr = make_zmq_path("tcp", meta.remote_host, remote_port)

            if self._has_slot_regions:
                # Acquire staging pool slot for this request's state RDMA
                consumer_staging_pool_idx = -1
                consumer_staging_addr = 0
                if self._staging_pool_size > 0:
                    consumer_staging_pool_idx = self._acquire_staging_slot()
                    consumer_staging_addr = (
                        self._staging_base_addr
                        + consumer_staging_pool_idx * self._staging_slot_bytes
                    )
                logger.info(
                    "[CONSUMER] Sending write_request (block+slot) for req %s "
                    "(transfer_id=%s, slot=%d, staging_pool=%d) to %s, "
                    "dst_block_ids=%s, %d block regions, %d slot regions",
                    req_id,
                    meta.transfer_id,
                    meta.local_slot_index,
                    consumer_staging_pool_idx,
                    remote_addr,
                    meta.local_block_ids[:10],
                    len(self._v4_block_regions),
                    len(self._v4_slot_regions),
                )
                write_request = msgpack.dumps(
                    {
                        "request_id": req_id,
                        "transfer_id": meta.transfer_id,
                        "consumer_host": self.local_ip,
                        "consumer_rpc_port": self.rpc_port,
                        "dst_block_ids": meta.local_block_ids,
                        "notify_host": self.local_ip,
                        "notify_port": self._notification_port,
                        "consumer_tp_size": self.tp_size,
                        "is_v4": True,
                        "dst_slot_index": meta.local_slot_index,
                        "consumer_v4_block_base_addrs": [
                            b for b, _ in self._v4_block_regions
                        ],
                        "consumer_v4_block_bpb": [
                            bpb for _, bpb in self._v4_block_regions
                        ],
                        "consumer_v4_slot_base_addrs": [
                            b for b, _ in self._v4_slot_regions
                        ],
                        "consumer_v4_slot_bps": [
                            bps for _, bps in self._v4_slot_regions
                        ],
                        "consumer_staging_addr": consumer_staging_addr,
                        "consumer_staging_bytes": self._staging_slot_bytes,
                    }
                )
            else:
                unique_bpb = sorted(set(self._per_block_bytes_list))
                logger.info(
                    "[CONSUMER] Sending write_request for req %s (transfer_id=%s) "
                    "to %s (handshake_port=%d, dp_rank=%d, "
                    "local_tp=%d, remote_tp=%d/%d), "
                    "dst_block_ids=%s, num_regions=%d, "
                    "bytes/block=%s, num_blocks=%d",
                    req_id,
                    meta.transfer_id,
                    remote_addr,
                    meta.remote_handshake_port,
                    meta.remote_dp_rank,
                    self.tp_rank,
                    remote_tp_rank,
                    remote_tp_size,
                    meta.local_block_ids[:10],
                    len(self.kv_caches_base_addr),
                    unique_bpb,
                    self.num_blocks,
                )
                write_request = msgpack.dumps(
                    {
                        "request_id": req_id,
                        "transfer_id": meta.transfer_id,
                        "consumer_host": self.local_ip,
                        "consumer_rpc_port": self.rpc_port,
                        "consumer_base_addrs": self.kv_caches_base_addr,
                        "dst_block_ids": meta.local_block_ids,
                        "notify_host": self.local_ip,
                        "notify_port": self._notification_port,
                        "consumer_tp_size": self.tp_size,
                    }
                )

            with self._notify_sockets_lock:
                sock = self._notify_sockets.get(remote_addr)
                if sock is None:
                    sock = self.zmq_context.socket(zmq.DEALER)
                    sock.setsockopt(zmq.LINGER, 5000)
                    sock.setsockopt(zmq.SNDHWM, 0)
                    sock.connect(remote_addr)
                    self._notify_sockets[remote_addr] = sock
                sock.send_multipart([MSG_WRITE_REQUEST, write_request])

            self._pending_recv.add(req_id)
            self._pending_recv_blocks[req_id] = list(meta.local_block_ids)
            if meta.local_slot_index >= 0:
                self._pending_recv_slots[req_id] = (
                    meta.local_slot_index,
                    consumer_staging_pool_idx,
                )
            logger.info(
                "[CONSUMER] write_request sent for req %s to %s",
                req_id,
                remote_addr,
            )

    # -----------------------------------------------------------------
    # Staging pool management
    # -----------------------------------------------------------------

    def _acquire_staging_slot(self) -> int:
        with self._staging_lock:
            if self._staging_free:
                return self._staging_free.pop()
        logger.warning(
            "Staging pool exhausted (size=%d), blocking until a slot is freed. "
            "Increase ATOM_PD_STAGING_POOL if this happens frequently.",
            self._staging_pool_size,
        )
        while True:
            time.sleep(0.001)
            with self._staging_lock:
                if self._staging_free:
                    return self._staging_free.pop()

    def _release_staging_slot(self, idx: int) -> None:
        with self._staging_lock:
            self._staging_free.append(idx)

    # -----------------------------------------------------------------
    # KVConnectorBase: get_finished
    # -----------------------------------------------------------------

    def get_finished(self) -> tuple[set, set]:
        """Return ``(done_sending, done_recving)`` and clear internal sets."""
        with self._completion_lock:
            ds = self.done_sending.copy()
            dr = self.done_recving.copy()
            self.done_sending.clear()
            self.done_recving.clear()
        if ds or dr:
            logger.info(
                "[%s] get_finished: sending=%s, recving=%s",
                "PRODUCER" if self.is_producer else "CONSUMER",
                ds,
                dr,
            )
        return ds, dr

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed RDMA receives."""
        with self._fence_lock:
            blocks = self._blocks_pending_fence
            self._blocks_pending_fence = []
        return blocks

    # -----------------------------------------------------------------
    # Producer: write listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _write_listener(self) -> None:
        """Accept write requests from consumers and dispatch RDMA writes."""
        path = make_zmq_path("tcp", "*", self._side_channel_port)
        logger.info("Mooncake write listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                identity, msg_type = parts[0], parts[1]

                if msg_type == MSG_GET_META:
                    encoded = self._encoder.encode(self._local_metadata)
                    sock.send_multipart([identity, b"", encoded])
                    logger.debug("Sent metadata to peer")

                elif msg_type == MSG_WRITE_REQUEST:
                    request_data = msgpack.loads(parts[2])
                    logger.info(
                        "[PRODUCER] Received write_request for req %s "
                        "(transfer_id=%s, consumer=%s:%s)",
                        request_data["request_id"],
                        request_data.get("transfer_id"),
                        request_data.get("consumer_host"),
                        request_data.get("consumer_rpc_port"),
                    )
                    self._send_executor.submit(self._execute_transfer, request_data)

                else:
                    logger.error("Unknown message type: %s", msg_type)

    # -----------------------------------------------------------------
    # Producer: execute RDMA write
    # -----------------------------------------------------------------

    def _execute_transfer(self, request_data: dict) -> None:
        """Compute offsets and perform RDMA write for a single request."""
        req_id = request_data["request_id"]
        transfer_id = request_data.get("transfer_id", req_id)
        consumer_host = request_data["consumer_host"]
        consumer_rpc_port = request_data["consumer_rpc_port"]
        dst_block_ids = request_data["dst_block_ids"]
        notify_host = request_data["notify_host"]
        notify_port = request_data["notify_port"]
        consumer_tp_size = request_data.get("consumer_tp_size", self.tp_size)
        consumers_per_rank = max(1, consumer_tp_size // self.tp_size)
        has_slot_data = request_data.get("is_v4", False)

        logger.info(
            "[PRODUCER] _execute_transfer: req_id=%s, transfer_id=%s, "
            "consumer=%s:%s, dst_blocks=%d, has_slot_data=%s",
            req_id,
            transfer_id,
            consumer_host,
            consumer_rpc_port,
            len(dst_block_ids),
            has_slot_data,
        )

        prefill_data = self._wait_for_prefill_data(transfer_id)
        if prefill_data is None:
            logger.error(
                "[PRODUCER] Timed out waiting for prefill data for "
                "transfer_id=%s (req_id=%s). Available keys: %s",
                transfer_id,
                req_id,
                list(self._completed_prefills.keys()),
            )
            return

        src_block_ids = prefill_data["block_ids"]
        target = f"{consumer_host}:{consumer_rpc_port}"

        if hasattr(self.transfer_engine, "get_first_buffer_address"):
            remote_buf = self.transfer_engine.get_first_buffer_address(target)
            if remote_buf == 0:
                logger.error(
                    "[PRODUCER] Consumer %s has NO registered buffers.",
                    target,
                )

        if has_slot_data:
            self._execute_block_slot_transfer(
                request_data,
                target,
                src_block_ids,
                dst_block_ids,
                prefill_data,
                req_id,
            )
        else:
            self._execute_block_transfer(
                request_data,
                target,
                src_block_ids,
                dst_block_ids,
                req_id,
            )

        # Notify consumer — all data (blocks + state for V4) is written.
        self._send_write_done(notify_host, notify_port, req_id)

        # Track refcount for multi-consumer TP fan-out.
        all_done = False
        with self._transfer_refcount_lock:
            if transfer_id not in self._transfer_refcount:
                self._transfer_refcount[transfer_id] = consumers_per_rank
            self._transfer_refcount[transfer_id] -= 1
            if self._transfer_refcount[transfer_id] <= 0:
                self._transfer_refcount.pop(transfer_id)
                all_done = True

        if all_done:
            with self._completion_lock:
                self.done_sending.add(transfer_id)
            with self._completed_prefills_lock:
                self._completed_prefills.pop(transfer_id, None)
            logger.info(
                "[PRODUCER] All %d consumers served for transfer_id=%s",
                consumers_per_rank,
                transfer_id,
            )

    def _execute_block_transfer(
        self,
        request_data: dict,
        target: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        req_id: str,
    ) -> None:
        """Block-only RDMA transfer (MHA, MLA, and other block-indexed backends)."""
        consumer_base_addrs = request_data["consumer_base_addrs"]

        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        sizes: list[int] = []

        num_regions = len(self.kv_caches_base_addr)
        for region_idx in range(num_regions):
            src_base = self.kv_caches_base_addr[region_idx]
            dst_base = consumer_base_addrs[region_idx]
            bpb = self._per_block_bytes_list[region_idx]
            for sb, db in zip(src_block_ids, dst_block_ids):
                src_addrs.append(src_base + sb * bpb)
                dst_addrs.append(dst_base + db * bpb)
                sizes.append(bpb)

        logger.info(
            "[PRODUCER] block RDMA write: req=%s, %d regions × %d blocks, "
            "total_bytes=%d",
            req_id,
            num_regions,
            len(src_block_ids),
            sum(sizes),
        )

        if not self._rdma_write_with_retry(
            target, src_addrs, dst_addrs, sizes, req_id, "block"
        ):
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)

    def _execute_block_slot_transfer(
        self,
        request_data: dict,
        target: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        prefill_data: dict,
        req_id: str,
    ) -> None:
        """Two-phase RDMA for backends with per-request state: block regions first, then slot regions."""
        consumer_block_addrs = request_data["consumer_v4_block_base_addrs"]
        consumer_block_bpb = request_data["consumer_v4_block_bpb"]
        consumer_slot_addrs = request_data["consumer_v4_slot_base_addrs"]
        consumer_slot_bps = request_data["consumer_v4_slot_bps"]
        dst_slot = request_data["dst_slot_index"]
        src_slot = prefill_data["slot_index"]

        # ---- Phase 1: Block transfer ----
        block_src: list[int] = []
        block_dst: list[int] = []
        block_sizes: list[int] = []

        for region_idx, (src_base, bpb) in enumerate(self._v4_block_regions):
            dst_base = consumer_block_addrs[region_idx]
            for sb, db in zip(src_block_ids, dst_block_ids):
                block_src.append(src_base + sb * bpb)
                block_dst.append(dst_base + db * consumer_block_bpb[region_idx])
                block_sizes.append(bpb)

        logger.info(
            "[PRODUCER] block RDMA: req=%s, %d regions × %d blocks, " "total_bytes=%d",
            req_id,
            len(self._v4_block_regions),
            len(src_block_ids),
            sum(block_sizes),
        )

        if not self._rdma_write_with_retry(
            target, block_src, block_dst, block_sizes, req_id, "block"
        ):
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)
            return

        # ---- Phase 2: Slot transfer ----
        if src_slot < 0 or dst_slot < 0:
            logger.info(
                "[PRODUCER] slot transfer skipped (src_slot=%d, dst_slot=%d)",
                src_slot,
                dst_slot,
            )
            return

        slot_src: list[int] = []
        slot_dst: list[int] = []
        slot_sizes: list[int] = []

        # Phase 2a: SWA slot regions (direct, no staging)
        for region_idx, (src_base, bps) in enumerate(self._v4_slot_regions):
            dst_base = consumer_slot_addrs[region_idx]
            slot_src.append(src_base + src_slot * bps)
            slot_dst.append(dst_base + dst_slot * consumer_slot_bps[region_idx])
            slot_sizes.append(bps)

        # Phase 2b: compressor states via staging buffer (182 → 1)
        producer_pool_idx = -1
        consumer_staging_addr = request_data.get("consumer_staging_addr", 0)
        if self._gather_slot is not None and consumer_staging_addr:
            producer_pool_idx = self._acquire_staging_slot()
            self._gather_slot(src_slot, producer_pool_idx)
            slot_src.append(
                self._staging_base_addr + producer_pool_idx * self._staging_slot_bytes
            )
            slot_dst.append(consumer_staging_addr)
            slot_sizes.append(self._staging_slot_bytes)

        logger.info(
            "[PRODUCER] slot RDMA: req=%s, %d entries, "
            "src_slot=%d → dst_slot=%d, total_bytes=%d",
            req_id,
            len(slot_src),
            src_slot,
            dst_slot,
            sum(slot_sizes),
        )

        if not self._rdma_write_with_retry(
            target, slot_src, slot_dst, slot_sizes, req_id, "slot"
        ):
            logger.error("[PRODUCER] slot transfer failed for req %s", req_id)

        if producer_pool_idx >= 0:
            self._release_staging_slot(producer_pool_idx)

    def _wait_for_prefill_data(self, req_id: str) -> dict | None:
        """Wait until prefill data is available for this request.

        Returns dict with "block_ids" and "slot_index" keys, or None on timeout.
        """
        with self._completed_prefills_cv:
            ready = self._completed_prefills_cv.wait_for(
                lambda: req_id in self._completed_prefills,
                timeout=PREFILL_LOOKUP_TIMEOUT,
            )
            if ready:
                return self._completed_prefills[req_id]
            return None

    def _rdma_write_with_retry(
        self,
        target: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        sizes: list[int],
        req_id: str,
        label: str,
    ) -> bool:
        """Chunked RDMA write with retry. Returns True on success."""
        max_entries_per_batch = 4096
        total_entries = len(src_addrs)
        max_retries = 3

        for chunk_start in range(0, total_entries, max_entries_per_batch):
            chunk_end = min(chunk_start + max_entries_per_batch, total_entries)
            chunk_src = src_addrs[chunk_start:chunk_end]
            chunk_dst = dst_addrs[chunk_start:chunk_end]
            chunk_sizes = sizes[chunk_start:chunk_end]

            retry_delay = 2.0
            for attempt in range(max_retries):
                try:
                    ret = self.transfer_engine.batch_transfer_sync_write(
                        target, chunk_src, chunk_dst, chunk_sizes
                    )
                    if ret == 0:
                        break
                    logger.error(
                        "[PRODUCER] %s RDMA chunk error %d for req %s → %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        label,
                        ret,
                        req_id,
                        target,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                except Exception:
                    logger.exception(
                        "[PRODUCER] %s RDMA chunk FAILED for req %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        label,
                        req_id,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                    ret = -1
                if ret == 0:
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return False
        return True

    def _send_write_done(self, host: str, port: int, req_id: str) -> None:
        """Send write-done notification to consumer via persistent socket.

        Sends the notification multiple times for reliability — the consumer
        uses a set so duplicates are harmless.
        """
        path = make_zmq_path("tcp", host, port)
        notification = msgpack.dumps({"request_id": req_id})
        with self._notify_sockets_lock:
            sock = self._notify_sockets.get(path)
            if sock is None:
                sock = self.zmq_context.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 5000)
                sock.setsockopt(zmq.SNDHWM, 0)
                sock.connect(path)
                self._notify_sockets[path] = sock
            for _ in range(3):
                sock.send_multipart([MSG_WRITE_DONE, notification])
        logger.info("[PRODUCER] write-done sent for req %s", req_id)

    # -----------------------------------------------------------------
    # Consumer: notification listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _notification_listener(self) -> None:
        """Receive write-done notifications from producers."""
        path = make_zmq_path("tcp", "*", self._notification_port)
        logger.info("Mooncake notification listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                msg_type = parts[1]

                if msg_type == MSG_WRITE_DONE:
                    data = msgpack.loads(parts[2])
                    req_id = data["request_id"]
                    slot_info = self._pending_recv_slots.pop(req_id, None)
                    if slot_info is not None and self._scatter_slot is not None:
                        compute_slot, pool_idx = slot_info
                        if pool_idx >= 0:
                            self._scatter_slot(compute_slot, pool_idx)
                            self._release_staging_slot(pool_idx)
                    dst_blocks = self._pending_recv_blocks.pop(req_id, None)
                    if dst_blocks:
                        with self._fence_lock:
                            self._blocks_pending_fence.extend(dst_blocks)
                    with self._completion_lock:
                        self.done_recving.add(req_id)
                        self._pending_recv.discard(req_id)
                    logger.info(
                        "[CONSUMER] Write-done received for req %s, "
                        "done_recving now: %s",
                        req_id,
                        self.done_recving,
                    )
                else:
                    logger.error("Unknown notification type: %s", msg_type)
