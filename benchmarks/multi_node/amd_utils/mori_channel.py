# SPDX-License-Identifier: Apache-2.0
# MoRI-IO transfer channel for LMCache (ROCm/AMD).
#
# Mirrors nixl_channel.py but uses ROCm MoRI-IO (https://github.com/rocm/mori)
# as the P2P RDMA transport instead of NIXL/UCX. This lets a single
# LMCacheConnectorV1 do BOTH disaggregated-prefill KV transfer AND prefix-cache
# reuse over MoRI — the same "unified connector" pattern NVIDIA uses with
# LMCache-over-NIXL — avoiding the MultiConnector[MoRIIOConnector, LMCacheMP]
# block-ownership conflict that corrupts KV on AMD.
#
# API mapping (nixl_agent -> mori.io.IOEngine):
#   get_agent_metadata()           -> get_engine_desc()            (+ .pack())
#   add_remote_agent(meta)         -> register_remote_engine(EngineDesc.unpack)
#   register_memory(reg_descs)     -> register_memory(ptr,len,gpu_id,GPU)
#   get_serialized_descs/deser     -> MemoryDesc.pack()/unpack()
#   make_prepped_xfer+transfer     -> batch_write/batch_read(+allocate_transfer_uid)
#   check_xfer_state(handle)       -> TransferStatus.Succeeded()/Wait()/Code()/Message()
#   deregister_memory              -> deregister_memory / deregister_remote_engine
#
# Addressing: like NixlChannel, the local LMCache staging buffer (buffer_ptr,
# buffer_size) is registered as one region and logically paged by `align_bytes`.
# A MemoryObj's page slot (mem_obj.meta.address) is the page index; the byte
# offset into the registered region is page_index * page_size.

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import asyncio
import os
import threading
import time
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

if TYPE_CHECKING:
    from mori.io import IOEngine, MemoryDesc

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Wire messages for the 2-stage handshake (mirrors NixlChannel's messages).
# Stage 1: exchange engine descriptors (register_remote_engine).
# Stage 2: exchange memory descriptors (the registered staging region).
# ---------------------------------------------------------------------------
class MoriMsgBase(msgspec.Struct, tag=True):
    pass


class MoriInitRequest(MoriMsgBase):
    local_engine_desc_bytes: bytes


class MoriInitResponse(MoriMsgBase):
    remote_engine_desc_bytes: bytes


class MoriMemRegRequest(MoriMsgBase):
    local_id: str
    local_mem_desc_bytes: bytes


class MoriMemRegResponse(MoriMsgBase):
    remote_mem_desc_bytes: bytes


MoriMsg = Union[
    MoriInitRequest, MoriInitResponse, MoriMemRegRequest, MoriMemRegResponse
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class MoriChannel(BaseTransferChannel):
    def __init__(
        self,
        async_mode: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        self.role = kwargs["role"]
        self.page_size = int(kwargs["align_bytes"])
        self.buffer_ptr = int(kwargs["buffer_ptr"])
        self.buffer_size = int(kwargs["buffer_size"])
        self.tp_rank = int(kwargs["tp_rank"])

        # gpu_id for mori memory registration. Defaults to tp_rank but can be
        # overridden if HIP_VISIBLE_DEVICES remaps the local device ordinal.
        gpu_id = kwargs.get("mori_gpu_id", None)
        if gpu_id is None:
            gpu_id = _env_int("LMCACHE_MORI_GPU_ID", self.tp_rank)
        self.gpu_id = int(gpu_id)

        # local_ip for the IOEngine bind (engine-to-engine RDMA connection).
        local_ip = kwargs.get("local_ip", None) or os.environ.get(
            "VLLM_HOST_IP", os.environ.get("LMCACHE_MORI_HOST_IP", "0.0.0.0")
        )
        self.local_ip = local_ip

        self.mori_wrapper = MoriEngineWrapper(
            buffer_ptr=self.buffer_ptr,
            buffer_size=self.buffer_size,
            tp_rank=self.tp_rank,
            gpu_id=self.gpu_id,
            local_ip=self.local_ip,
        )
        self.engine: "IOEngine" = self.mori_wrapper.engine
        self.local_mem_desc: "MemoryDesc" = self.mori_wrapper.mem_desc

        # Used for P2P side message (mirrors NixlChannel.peer_lookup_url).
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        # peer_id -> remote MemoryDesc (target of write / source of read)
        self.remote_mem_descs_dict: dict[str, "MemoryDesc"] = {}
        # peer_id set of engines we have already register_remote_engine'd
        self._registered_engines: set[str] = set()

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        self.zmq_context = get_zmq_context(use_asyncio=async_mode)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self.transfer_timeout = float(
            os.environ.get("LMCACHE_MORI_TRANSFER_TIMEOUT", "120")
        )

        self._init_side_channels()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _register_remote_engine_once(self, key: str, engine_desc_bytes: bytes):
        # mori is import-time-gated to ROCm hosts; import lazily.
        from mori.io import EngineDesc

        if key in self._registered_engines:
            return
        self.engine.register_remote_engine(EngineDesc.unpack(engine_desc_bytes))
        self._registered_engines.add(key)

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )

        # Stage 1: exchange engine descriptors.
        init_req = MoriInitRequest(
            local_engine_desc_bytes=self.engine.get_engine_desc().pack()
        )
        init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        init_resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=MoriMsg)
        self._register_remote_engine_once(peer_id, init_resp.remote_engine_desc_bytes)

        # Stage 2: exchange memory descriptors.
        mem_req = MoriMemRegRequest(
            local_id=local_id,
            local_mem_desc_bytes=self.local_mem_desc.pack(),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(mem_req))
        mem_resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=MoriMsg)
        self._store_remote_mem_desc(peer_id, mem_resp.remote_mem_desc_bytes)

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(init_tmp_socket, init_side_msg)

        init_tmp_socket.close()
        return init_ret_msg

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )

        init_req = MoriInitRequest(
            local_engine_desc_bytes=self.engine.get_engine_desc().pack()
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        init_resp = msgspec.msgpack.decode(await init_tmp_socket.recv(), type=MoriMsg)
        self._register_remote_engine_once(peer_id, init_resp.remote_engine_desc_bytes)

        mem_req = MoriMemRegRequest(
            local_id=local_id,
            local_mem_desc_bytes=self.local_mem_desc.pack(),
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(mem_req))
        mem_resp = msgspec.msgpack.decode(await init_tmp_socket.recv(), type=MoriMsg)
        self._store_remote_mem_desc(peer_id, mem_resp.remote_mem_desc_bytes)

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket, init_side_msg
            )

        init_tmp_socket.close()
        return init_ret_msg

    def _store_remote_mem_desc(self, peer_id: str, mem_desc_bytes: bytes):
        from mori.io import MemoryDesc

        self.remote_mem_descs_dict[peer_id] = MemoryDesc.unpack(mem_desc_bytes)

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_mem_descs_dict

    def _init_side_channels(self):
        if self.peer_init_url is None:
            return
        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[MoriMsg, InitSideMsgBase]
    ) -> Union[MoriMsg, InitSideRetMsgBase]:
        resp: Union[MoriMsg, InitSideRetMsgBase]
        if isinstance(req, MoriInitRequest):
            # Register the initiator's engine, reply with ours.
            self._register_remote_engine_once(
                "peer:" + str(uuid.uuid4()), req.local_engine_desc_bytes
            )
            resp = MoriInitResponse(
                remote_engine_desc_bytes=self.engine.get_engine_desc().pack()
            )
            logger.info("MoRI: replied init (engine desc) response")
        elif isinstance(req, MoriMemRegRequest):
            self._store_remote_mem_desc(req.local_id, req.local_mem_desc_bytes)
            resp = MoriMemRegResponse(
                remote_mem_desc_bytes=self.local_mem_desc.pack()
            )
            logger.info("MoRI: replied mem-register response")
        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("MoRI: replied P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")
        return resp

    def _init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.init_side_channel)
        while self.running:
            try:
                req = msgspec.msgpack.decode(
                    self.init_side_channel.recv(), type=Union[MoriMsg, SideMsg]
                )
                resp = self._handle_init_msg(req)
                self.init_side_channel.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("MoRI init loop error: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.init_side_channel)
        logger.info("MoRI: starting async init loop")
        while self.running:
            try:
                req = msgspec.msgpack.decode(
                    await self.init_side_channel.recv(), type=Union[MoriMsg, SideMsg]
                )
                resp = self._handle_init_msg(req)
                await self.init_side_channel.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("MoRI async init loop error: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in MoRI channel"
            )
        return local_indices

    def _offsets(self, page_indices: list[int]) -> list[int]:
        ps = self.page_size
        return [int(i) * ps for i in page_indices]

    def _wait_statuses(self, statuses: list) -> None:
        """Block until all mori TransferStatus complete; raise on failure."""
        if not statuses:
            return
        start = time.monotonic()
        remaining = list(statuses)
        while remaining:
            still = []
            for st in remaining:
                if st.Succeeded():
                    continue
                still.append(st)
            remaining = still
            if not remaining:
                break
            if time.monotonic() - start > self.transfer_timeout:
                raise RuntimeError(
                    f"MoRI transfer timeout after {self.transfer_timeout}s, "
                    f"{len(remaining)}/{len(statuses)} pending"
                )
            time.sleep(0.001)

    async def _async_wait_statuses(self, statuses: list) -> None:
        if not statuses:
            return
        start = time.monotonic()
        remaining = list(statuses)
        while remaining:
            remaining = [st for st in remaining if not st.Succeeded()]
            if not remaining:
                break
            if time.monotonic() - start > self.transfer_timeout:
                raise RuntimeError(
                    f"MoRI async transfer timeout after {self.transfer_timeout}s, "
                    f"{len(remaining)}/{len(statuses)} pending"
                )
            await asyncio.sleep(0.001)

    # ------------------------------------------------------------------ #
    # Send/Recv (two-sided) — not used by LMCache PD path (write/read only)
    # ------------------------------------------------------------------ #
    def batched_send(self, objects, transfer_spec=None) -> int:
        raise NotImplementedError

    def batched_recv(self, buffers, transfer_spec=None) -> int:
        raise NotImplementedError

    async def async_batched_send(self, objects, transfer_spec=None) -> int:
        raise NotImplementedError

    async def async_batched_recv(self, buffers, transfer_spec=None) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Read/Write (one-sided RDMA) — the core KV transfer
    # ------------------------------------------------------------------ #
    def _batch_op(self, op: str, mem_objs, transfer_spec):
        assert transfer_spec is not None
        peer_key = "receiver_id" if op == "write" else "sender_id"
        peer_id = transfer_spec[peer_key]
        remote_desc = self.remote_mem_descs_dict[peer_id]

        local_offsets = self._offsets(self.get_local_mem_indices(mem_objs))
        remote_offsets = self._offsets(transfer_spec["remote_indexes"])
        sizes = [self.page_size] * len(local_offsets)
        uid = self.engine.allocate_transfer_uid()

        if op == "write":
            # local region -> remote region
            statuses = self.engine.batch_write(
                [self.local_mem_desc], [local_offsets],
                [remote_desc], [remote_offsets],
                [sizes], [uid],
            )
        else:
            # remote region -> local region
            statuses = self.engine.batch_read(
                [self.local_mem_desc], [local_offsets],
                [remote_desc], [remote_offsets],
                [sizes], [uid],
            )
        return statuses

    def batched_write(self, objects, transfer_spec=None) -> int:
        statuses = self._batch_op("write", objects, transfer_spec)
        self._wait_statuses(statuses)
        return len(objects)

    def batched_read(self, buffers, transfer_spec=None) -> int:
        statuses = self._batch_op("read", buffers, transfer_spec)
        self._wait_statuses(statuses)
        return len(buffers)

    async def async_batched_write(self, objects, transfer_spec=None) -> int:
        statuses = self._batch_op("write", objects, transfer_spec)
        await self._async_wait_statuses(statuses)
        return len(objects)

    async def async_batched_read(self, buffers, transfer_spec=None) -> int:
        statuses = self._batch_op("read", buffers, transfer_spec)
        await self._async_wait_statuses(statuses)
        return len(buffers)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join(timeout=2.0)
        try:
            self.zmq_context.term()
        except Exception:
            pass
        try:
            self.engine.deregister_memory(self.local_mem_desc)
        except Exception as e:
            logger.warning("MoRI deregister_memory failed: %s", e)


@dataclass
class MoriEngineWrapper:
    engine: Any
    mem_desc: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        tp_rank: int,
        gpu_id: int,
        local_ip: str = "0.0.0.0",
    ):
        try:
            from mori.io import (
                BackendType,
                IOEngine,
                IOEngineConfig,
                MemoryLocationType,
                PollCqMode,
                RdmaBackendConfig,
            )
        except ImportError as err:
            raise RuntimeError("MoRI-IO (mori) is not available") from err

        engine_key = f"lmcache-mori-tp{tp_rank}-{uuid.uuid4().hex[:8]}"
        # port=0 -> auto-bind; the actual host:port is carried in EngineDesc and
        # exchanged via the zmq handshake, then register_remote_engine connects.
        config = IOEngineConfig(host=local_ip, port=0)
        engine = IOEngine(engine_key, config)

        qp_per_transfer = _env_int("LMCACHE_MORI_QP_PER_TRANSFER", 4)
        post_batch_size = _env_int("LMCACHE_MORI_POST_BATCH_SIZE", -1)
        num_worker_threads = _env_int("LMCACHE_MORI_NUM_WORKERS", 4)
        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            PollCqMode.POLLING,
            False,
        )
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        actual_port = engine.get_engine_desc().port
        assert actual_port > 0, f"MoRI engine {engine_key} failed to bind a port"
        logger.info(
            "MoRI IOEngine %s bound %s:%s (qp=%s workers=%s)",
            engine_key, local_ip, actual_port, qp_per_transfer, num_worker_threads,
        )

        # Register the LMCache contiguous staging buffer (GPU) as one region.
        mem_desc = engine.register_memory(
            buffer_ptr, buffer_size, gpu_id, MemoryLocationType.GPU
        )

        self.engine = engine
        self.mem_desc = mem_desc
