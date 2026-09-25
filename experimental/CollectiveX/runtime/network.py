"""Network environment selection shared by launchers, preparation, and ranks."""
from __future__ import annotations

from pathlib import Path
import os
import re


DEVICE_LIST = r"[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*"
INTERFACE = r"[A-Za-z][A-Za-z0-9_.-]{0,31}"
SCRUB = """NCCL_NET NCCL_NET_PLUGIN NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
NCCL_IB_GID_INDEX NCCL_IB_SL NCCL_IB_MERGE_NICS NCCL_CROSS_NIC
NVSHMEM_ENABLE_NIC_PE_MAPPING NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER EP_NIC_NAME EP_OVERRIDE_RDMA_SL
MORI_RDMA_DEVICES MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
UCCL_SOCKET_IFNAME UCCL_IB_HCA UCCL_IB_GID_INDEX UCCL_IB_SL UCCL_IB_TC
UCCL_IB_MAX_INFLIGHT_BYTES UCCL_IB_MAX_INFLIGHT_NORMAL UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC""".split()


def gid_environment(env: dict[str, str], layer: str) -> dict[str, str]:
    """Only RoCE consumes a GID index; UCCL needs its own variable, not NCCL's."""
    result = dict(env)
    names = ("NVSHMEM_IB_GID_INDEX", "NCCL_IB_GID_INDEX", "UCCL_IB_GID_INDEX")
    for name in names:
        result.pop(name, None)
    index = env.get("COLLX_IB_GID_INDEX", "")
    if index:
        if layer not in ("roce", "infiniband", "efa"):
            raise ValueError("unsupported RDMA link layer")
        if layer == "roce":
            result.update(dict.fromkeys(names, index))
    return result


def network_environment(env: dict[str, str], nodes: int, transport: str) -> dict[str, str]:
    """Apply the existing platform selectors without inheriting another transport's state."""
    if nodes < 1:
        raise ValueError("invalid network placement")
    result = {key: value for key, value in env.items() if key not in SCRUB}
    single = env.get("COLLX_SINGLE_NODE_RDMA_DEVICES", "")
    # Single-node and MNNVL runs need only the scrub. Single-node low-latency kernels use
    # NVLink/XGMI; /dev/gdrdrv is absent on H200. A pinned HCA list can still steer the legacy
    # Buffer's self-enabled IBGDA initialization away from B300 rails that reject AH creation.
    if nodes == 1 and single:
        if not re.fullmatch(DEVICE_LIST, single):
            raise ValueError("invalid private single-node RDMA device selector")
        result["NVSHMEM_HCA_LIST"] = single
    if nodes == 1 or transport == "mnnvl":
        return result

    devices = env.get("COLLX_RDMA_DEVICES", "")
    if not devices:
        raise ValueError("RDMA execution requires a private device selector")
    if not re.fullmatch(DEVICE_LIST, devices):
        raise ValueError("invalid private RDMA device selector")
    fabric = env.get("COLLX_RDMA_FABRIC", "")
    if fabric not in ("", "efa"):
        raise ValueError("invalid private RDMA fabric")
    interface = env.get("COLLX_SOCKET_IFNAME", "")
    if interface:
        if not re.fullmatch(INTERFACE, interface):
            raise ValueError("invalid private socket interface selector")
        result.update(NCCL_SOCKET_IFNAME=interface, GLOO_SOCKET_IFNAME=interface)
    if fabric == "efa":
        # EFA has no verbs GID/SL/TC selectors. The cluster's enroot hook mounts aws-ofi-nccl;
        # its libfabric plugin enumerates the rails and exports ncclGinPlugin for GIN.
        result.update(NCCL_NET_PLUGIN="ofi", FI_PROVIDER="efa", FI_EFA_FORK_SAFE="1",
                      NVSHMEM_REMOTE_TRANSPORT="libfabric", NVSHMEM_LIBFABRIC_PROVIDER="efa")
        return result

    names = [selector.split(":")[0] for selector in devices.split(",")]
    result.update(NVSHMEM_HCA_LIST=devices, NVSHMEM_ENABLE_NIC_PE_MAPPING="1",
                  NCCL_IB_HCA=f"={devices}", MORI_RDMA_DEVICES=",".join(names),
                  EP_NIC_NAME=names[0], UCCL_IB_HCA=f"={devices}", UCCL_SOCKET_IFNAME=interface)
    # RCCL selects its own plugin; forcing NCCL_NET=IB breaks AMD. UCCL honors the exact
    # '=' selector and ':port'; dropping them would prefix-match mlx5_1 against mlx5_10..19.
    if env.get("COLLX_VENDOR", "nvidia") == "amd":
        result.update(UCCL_IB_MAX_INFLIGHT_BYTES="2097152", UCCL_IB_MAX_INFLIGHT_NORMAL="1",
                      UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC="1")
    else:
        result["NCCL_NET"] = "IB"
    # NCCL's dual-port fusion disables GIN. Rail-isolated fabrics also prohibit cross-NIC
    # pairs: peers on separate per-port subnets black-hole at QP RTR.
    result["NCCL_IB_MERGE_NICS"] = "0"
    isolated = env.get("COLLX_RAIL_ISOLATED", "")
    if isolated not in ("", "0", "1"):
        raise ValueError("invalid private rail isolation flag")
    if isolated == "1":
        result["NCCL_CROSS_NIC"] = "0"
    for field, limit, label, targets in (
        ("COLLX_IB_GID_INDEX", 255, "IB GID index", ()),
        ("COLLX_RDMA_SERVICE_LEVEL", 15, "RDMA service level",
         ("NVSHMEM_IB_SL", "NCCL_IB_SL", "EP_OVERRIDE_RDMA_SL", "MORI_RDMA_SL", "MORI_IO_SL", "UCCL_IB_SL")),
        ("COLLX_RDMA_TRAFFIC_CLASS", 255, "RDMA traffic class", ("MORI_RDMA_TC", "MORI_IO_TC", "UCCL_IB_TC")),
    ):
        value = env.get(field, "")
        if value:
            if not re.fullmatch(r"[0-9]+", value) or int(value) > limit:
                raise ValueError(f"invalid private {label}")
            result.update(dict.fromkeys(targets, value))
    result.update(NVSHMEM_IB_ENABLE_IBGDA="1", NVSHMEM_IBGDA_NIC_HANDLER="gpu")
    layer = env.get("COLLX_RDMA_LINK_LAYER", "")
    if layer:
        if layer not in ("roce", "infiniband", "efa"):
            raise ValueError("invalid validated RDMA link layer")
        result = gid_environment(result, layer)
    return result


def validated_selectors(output: str, nodes: int, env: dict[str, str]) -> dict[str, str]:
    """Resolve the existing private probe markers, including different per-node interfaces."""
    interfaces = re.findall(rf"^\[collectivex-private\] socket-interface-selected=({INTERFACE})$", output, re.M)
    count = len(re.findall(r"^\[collectivex-private\] socket-interface-selected=", output, re.M))
    if not interfaces or count != nodes:
        raise RuntimeError(f"network-profile-socket-markers={count}/{nodes} unique={len(set(interfaces))}")
    layers = re.findall(r"^\[collectivex-private\] rdma-link-layer=(roce|infiniband|efa)$", output, re.M)
    if len(layers) != nodes or len(set(layers)) != 1:
        raise RuntimeError("network-profile link-layer markers disagree")
    result = dict(env)
    if len(set(interfaces)) == 1:
        result["COLLX_SOCKET_IFNAME"] = interfaces[0]
    else:
        result.pop("COLLX_SOCKET_IFNAME", None)
    result["COLLX_RDMA_LINK_LAYER"] = layers[0]
    return gid_environment(result, layers[0])


def validate_container_network(env: dict[str, str], sys_root: Path = Path("/sys")) -> None:
    """Recheck the selected host interfaces inside the actual container."""
    if int(env.get("COLLX_NODES", "1")) <= 1 or env.get("COLLX_TRANSPORT") == "mnnvl":
        return
    devices = env.get("COLLX_RDMA_DEVICES", "")
    if not re.fullmatch(DEVICE_LIST, devices):
        raise RuntimeError("invalid scale-out RDMA selector")
    if env.get("COLLX_RDMA_FABRIC") == "efa":
        if not os.access("/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so", os.R_OK):
            raise RuntimeError("aws-ofi-nccl plugin is absent inside the container")
    if not env.get("GLOO_SOCKET_IFNAME") or not devices:
        raise RuntimeError("scale-out network selectors are unavailable")
    for interface in env["GLOO_SOCKET_IFNAME"].split(","):
        if not (sys_root / "class/net" / interface).is_dir():
            raise RuntimeError("configured scale-out socket interface is absent")
    for device in devices.split(","):
        if not (sys_root / "class/infiniband" / device.split(":")[0]).is_dir():
            raise RuntimeError("configured scale-out RDMA device is absent")
