#!/usr/bin/env python3
"""Allocation and network checks used by CollectiveX launchers."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


def default_route_interface(route_path: Path = Path("/proc/net/route")) -> str:
    for line in route_path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 1:
            return fields[0]
    return ""


def prepare_cache(parent_path: str) -> str:
    path = Path(parent_path).resolve() / f".collectivex-backend-cache-{os.getuid()}"
    # parents=True: this runs before the first container import, so a fresh pool's squash_dir may
    # not exist yet (b200-nscale run 31092445934). 0o700 applies to the cache dir, not its parents.
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return str(path)


DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# Every manifest media type a tag can point at; the registry answers with the digest of
# whichever it holds. For a multi-arch tag that is the index digest, which changes whenever
# any platform updates — a slightly over-eager cache key, never a stale one.
MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def registry_reference(image: str) -> tuple[str, str, str]:
    """Split host/repository:tag with Docker Hub's implied-registry rules."""
    name, _, tag = image.rpartition(":")
    first, _, rest = name.partition("/")
    if rest and ("." in first or first == "localhost"):
        return first, rest, tag
    return "registry-1.docker.io", name if "/" in name else f"library/{name}", tag


def bearer_challenge(header: str) -> dict[str, str]:
    scheme, _, params = header.partition(" ")
    if scheme.lower() != "bearer":
        return {}
    return {key.lower(): value
            for key, value in re.findall(r'([A-Za-z_]+)="([^"]*)"', params)}


def resolve_image_digest(image: str, timeout: float = 10.0, opener=None) -> str:
    """Best-effort manifest digest for a tag; empty string on any failure.

    Anonymous registry-v2 token dance (docker.io/ghcr.io/nvcr.io all speak it for
    public images). Only HEAD requests — nothing is pulled.
    """
    host, repository, tag = registry_reference(image)
    if not tag:
        return ""
    opener = opener or urllib.request.build_opener()
    url = f"https://{host}/v2/{repository}/manifests/{tag}"

    def head(token: str = ""):
        request = urllib.request.Request(
            url, method="HEAD", headers={"Accept": MANIFEST_ACCEPT})
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        return opener.open(request, timeout=timeout)

    try:
        try:
            response = head()
        except urllib.error.HTTPError as error:
            if error.code != 401:
                return ""
            challenge = bearer_challenge(error.headers.get("WWW-Authenticate", ""))
            realm = challenge.get("realm", "")
            if not realm.startswith("https://"):
                return ""
            query = {"scope": f"repository:{repository}:pull"}
            if challenge.get("service"):
                query["service"] = challenge["service"]
            with opener.open(f"{realm}?{urllib.parse.urlencode(query)}",
                             timeout=timeout) as grant:
                body = json.loads(grant.read().decode())
            token = body.get("token") or body.get("access_token") or ""
            if not token:
                return ""
            response = head(token)
        with response:
            digest = response.headers.get("Docker-Content-Digest", "") or ""
    except (OSError, ValueError):
        return ""
    return digest if DIGEST_PATTERN.fullmatch(digest) else ""


def validate_cuda_context(expected: int) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != expected:
        raise SystemExit(1)


_GPU_HEALTH_FIELDS = ("index", "clocks_event_reasons.sw_thermal_slowdown",
                      "clocks_event_reasons.hw_thermal_slowdown", "temperature.gpu")


def gpu_health_faults(output: str, max_temperature_c: int = 90) -> list[str]:
    """Throttled or overheating GPUs in an `nvidia-smi --format=csv,noheader` block.

    Split out from the I/O so parsing is testable without hardware; see
    tests/test_runtime.py::GpuHealthProbe. Returns [] for anything unreadable -- the caller treats
    an unreadable probe as healthy rather than blocking a leg on it.
    """
    faults = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(_GPU_HEALTH_FIELDS):
            continue
        index, software, hardware, temperature = cells
        # "Not Active" is the healthy reading, so compare exactly -- a substring test for
        # "Active" passes the fault straight through.
        throttled = "Active" in (software, hardware)
        try:
            too_hot = int(temperature.split()[0]) > max_temperature_c
        except (IndexError, ValueError):
            too_hot = False
        if throttled or too_hot:
            faults.append(
                f"gpu {index}: sw_thermal={software} hw_thermal={hardware} temp={temperature}"
            )
    return faults


def gpu_temperature_spread(output: str) -> tuple[int, int, int] | None:
    """`(hottest, median, spread)` GPU temperature, or None if unreadable.

    Reported, not gated on: the absolute threshold in `gpu_health_faults` can be unreachable (an
    H100 clamps at ~86-87 C, under the 90 C gate), and in the one measured fault the only
    pre-flight signal was relative -- the sick GPU idled at 55 C against ~30 C for its siblings.
    Healthy references: 50-66 C under load on h100, 34-39 C on b200.
    """
    temperatures = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(_GPU_HEALTH_FIELDS):
            continue
        try:
            temperatures.append(int(cells[3].split()[0]))
        except (IndexError, ValueError):
            continue
    if not temperatures:
        return None
    temperatures.sort()
    median = temperatures[len(temperatures) // 2]
    return temperatures[-1], median, temperatures[-1] - median


def validate_gpu_health(max_temperature_c: int = 90) -> None:
    """Reject an allocation holding a thermally throttled GPU.

    Every collective is a barrier, so one clamped device paces every rank: a B200 with GPU 7 at
    120 MHz ran a case 17x slower and was killed twice by the wall-clock guard. Gate on the throttle
    flag, not the clock -- an idle B200 also reads 120 MHz -- with temperature as an independent
    second signal. Fails open on anything unreadable: no `nvidia-smi`, non-zero exit, bad output.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_HEALTH_FIELDS)}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    faults = gpu_health_faults(output, max_temperature_c)
    for fault in faults:
        _emit(f"gpu-health-fault {fault}")
    if faults:
        raise SystemExit(1)
    # Positive control: without it a blind gate -- no visible devices, or a driver spelling these
    # fields `clocks_throttle_reasons.*` -- is indistinguishable from a healthy pass.
    spread = gpu_temperature_spread(output)
    detail = "" if spread is None else f" hottest={spread[0]}C median={spread[1]}C spread={spread[2]}C"
    _emit(
        f"gpu-health-checked gpus={sum(1 for line in output.splitlines() if line.strip())}{detail}"
    )


def _emit(marker: str) -> None:
    # collx_validate_network_profile_on_job (runtime/common.sh) greps these exact strings
    # out of the per-node probe log to derive COLLX_SOCKET_IFNAME / COLLX_RDMA_LINK_LAYER and to
    # diagnose failures. The marker vocabulary is a string contract with that function —
    # keep the two halves in lockstep (see tests/test_runtime.py::NetworkProfileContract).
    print(f"[collectivex-private] {marker}")


def _check_port(port_path: Path, ordinal: int, gid_index: str, profile: str):
    # Return the port's link layer ("roce"/"infiniband") when it is active, carries a
    # non-empty GID at the pinned index, and agrees with any already-seen link layer;
    # otherwise emit the matching rdma-port-<ordinal>=<reason> marker and return None.
    if not port_path.is_dir():
        _emit(f"rdma-port-{ordinal}=missing"); return None
    state = port_path / "state"
    if not state.is_file() or state.read_text().split()[:1] != ["4:"]:
        _emit(f"rdma-port-{ordinal}=inactive"); return None
    if gid_index:
        gid = port_path / "gids" / gid_index
        if not gid.is_file():
            _emit(f"rdma-port-{ordinal}=gid-missing"); return None
        if not "".join(c for c in gid.read_text() if c not in ":0" and not c.isspace()):
            _emit(f"rdma-port-{ordinal}=gid-empty"); return None
    link = port_path / "link_layer"
    if not link.is_file():
        _emit(f"rdma-port-{ordinal}=link-layer-missing"); return None
    layer = {"Ethernet": "roce", "InfiniBand": "infiniband"}.get(link.read_text().strip())
    if layer is None:
        _emit(f"rdma-port-{ordinal}=link-layer-invalid"); return None
    if profile and profile != layer:
        _emit(f"rdma-port-{ordinal}=link-layer-mixed"); return None
    return layer


def validate_network_profile(socket_names: str, rdma_devices: str, gid_index: str,
                             sys_root: Path = Path("/sys"),
                             route_path: Path = Path("/proc/net/route")) -> None:
    # Prove the operator-pinned scale-out fabric on this node: resolve the cross-node socket
    # interface (operator selector, else this node's default route), confirm it is live, and
    # confirm every pinned RDMA port is active with a consistent link layer. On success emit
    # the socket-interface-selected and rdma-link-layer markers the launcher consumes; on any
    # failure emit the diagnostic marker and exit non-zero. The seam carries exactly ONE socket
    # interface: the launcher's marker-extraction regex has no comma, so a multi-interface
    # selector could never survive past this probe — fail it loudly here instead.
    interface = socket_names or default_route_interface(route_path)
    if not interface:
        _emit("socket-interface-1=default-route-missing")
        raise SystemExit(1)
    _emit(f"socket-interface-selected={interface}")
    net = sys_root / "class" / "net" / interface
    if not net.is_dir():
        _emit("socket-interface-1=missing"); raise SystemExit(1)
    operstate = net / "operstate"
    state = operstate.read_text().strip() if operstate.is_file() else ""
    if state not in ("up", "unknown"):
        _emit("socket-interface-1=down"); raise SystemExit(1)
    profile = ""
    for ordinal, selector in enumerate((s for s in rdma_devices.split(",") if s), start=1):
        device, _, configured_port = selector.partition(":")
        ports = sys_root / "class" / "infiniband" / device / "ports"
        if not ports.is_dir():
            _emit(f"rdma-device-{ordinal}=missing"); raise SystemExit(1)
        if configured_port:
            layer = _check_port(ports / configured_port, ordinal, gid_index, profile)
            if layer is None: raise SystemExit(1)
            profile = layer
        else:
            active = False
            for port_path in sorted(p for p in ports.iterdir() if p.is_dir()):
                layer = _check_port(port_path, ordinal, gid_index, profile)
                if layer is not None:
                    profile, active = layer, True
            if not active: raise SystemExit(1)
    if not profile: raise SystemExit(1)
    _emit(f"rdma-link-layer={profile}")


def main() -> None:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("default-route-interface")
    command = commands.add_parser("prepare-cache"); command.add_argument("parent")
    command = commands.add_parser("cuda-context"); command.add_argument("expected", type=int)
    command = commands.add_parser("image-digest"); command.add_argument("image")
    commands.add_parser("gpu-health")
    command = commands.add_parser("network-profile"); command.add_argument("socket_names"); command.add_argument("rdma_devices"); command.add_argument("gid_index")
    args = parser.parse_args()
    if args.command == "default-route-interface": print(default_route_interface(), end="")
    elif args.command == "prepare-cache": print(prepare_cache(args.parent), end="")
    elif args.command == "cuda-context": validate_cuda_context(args.expected)
    elif args.command == "image-digest": print(resolve_image_digest(args.image), end="")
    elif args.command == "gpu-health": validate_gpu_health()
    else: validate_network_profile(args.socket_names, args.rdma_devices, args.gid_index)


if __name__ == "__main__": main()
