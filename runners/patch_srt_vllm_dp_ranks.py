#!/usr/bin/env python3
"""Patch srt-slurm v1.0.53 to allocate one process per vLLM DP rank."""

from __future__ import annotations

import sys
from pathlib import Path


OLD_BLOCK = '''            else:
                # DP+EP mode: one process per GPU
                # Each process gets a single GPU and a unique dp_rank
                dp_rank = 0
                # Allocate a unique DP RPC port for this endpoint's leader node
                dp_rpc_port = port_allocator.next_dp_rpc_port(endpoint.leader_node)
                # Allocate a single NIXL base port for this endpoint.
                # vLLM internally computes: actual_port = base + data_parallel_rank
                # so all DP ranks in the endpoint share the same base port.
                dp_size = self._get_dp_size(endpoint.mode) or len(endpoint.gpu_indices)
                nixl_base_port = port_allocator.next_nixl_port_block(dp_size)
                for _node_rank, node in enumerate(endpoint.nodes):
                    for gpu_idx in sorted(endpoint.gpu_indices):
                        is_leader = dp_rank == 0
                        http_port = port_allocator.next_http_port(node) if is_leader else 0
                        bootstrap_port = (
                            port_allocator.next_bootstrap_port(node)
                            if endpoint.mode == "prefill" and is_leader
                            else None
                        )
                        kv_events_port = port_allocator.next_kv_events_port()
                        nixl_port = nixl_base_port

                        processes.append(
                            Process(
                                node=node,
                                gpu_indices=frozenset([gpu_idx]),  # Single GPU per process
                                sys_port=current_sys_port,
                                http_port=http_port,
                                endpoint_mode=endpoint.mode,
                                endpoint_index=endpoint.index,
                                node_rank=dp_rank,  # dp_rank stored in node_rank for now
                                bootstrap_port=bootstrap_port,
                                kv_events_port=kv_events_port,
                                nixl_port=nixl_port,
                                dp_rpc_port=dp_rpc_port,
                            )
                        )
                        current_sys_port += 1
                        dp_rank += 1
'''

NEW_BLOCK = '''            else:
                # External DP mode: one process per DP rank. A rank may own
                # multiple GPUs when tensor or pipeline parallelism is enabled.
                dp_rank = 0
                dp_rpc_port = port_allocator.next_dp_rpc_port(endpoint.leader_node)
                config = self.get_config_for_mode(endpoint.mode)
                dp_size = self._get_dp_size(endpoint.mode) or endpoint.total_gpus
                tp_size = config.get("tensor-parallel-size") or config.get("tensor_parallel_size") or 1
                pp_size = config.get("pipeline-parallel-size") or config.get("pipeline_parallel_size") or 1
                gpus_per_dp_rank = tp_size * pp_size
                expected_gpus = dp_size * gpus_per_dp_rank
                if endpoint.total_gpus != expected_gpus:
                    raise ValueError(
                        f"{endpoint.mode} DP={dp_size}, TP={tp_size}, PP={pp_size} requires "
                        f"{expected_gpus} GPUs, but the endpoint allocated {endpoint.total_gpus}"
                    )

                nixl_base_port = port_allocator.next_nixl_port_block(dp_size)
                for node in endpoint.nodes:
                    local_gpus = sorted(endpoint.gpu_indices)
                    if len(local_gpus) % gpus_per_dp_rank != 0:
                        raise ValueError(
                            f"{endpoint.mode} TP={tp_size}, PP={pp_size} requires "
                            f"{gpus_per_dp_rank} GPUs per DP rank, but node {node} has "
                            f"{len(local_gpus)} allocated GPUs"
                        )
                    for offset in range(0, len(local_gpus), gpus_per_dp_rank):
                        rank_gpus = frozenset(local_gpus[offset : offset + gpus_per_dp_rank])
                        is_leader = dp_rank == 0
                        http_port = port_allocator.next_http_port(node) if is_leader else 0
                        bootstrap_port = (
                            port_allocator.next_bootstrap_port(node)
                            if endpoint.mode == "prefill" and is_leader
                            else None
                        )

                        processes.append(
                            Process(
                                node=node,
                                gpu_indices=rank_gpus,
                                sys_port=current_sys_port,
                                http_port=http_port,
                                endpoint_mode=endpoint.mode,
                                endpoint_index=endpoint.index,
                                node_rank=dp_rank,
                                bootstrap_port=bootstrap_port,
                                kv_events_port=port_allocator.next_kv_events_port(),
                                nixl_port=nixl_base_port,
                                dp_rpc_port=dp_rpc_port,
                            )
                        )
                        current_sys_port += 1
                        dp_rank += 1

                if dp_rank != dp_size:
                    raise ValueError(
                        f"{endpoint.mode} allocated {dp_rank} DP ranks, expected {dp_size}"
                    )
'''


def patch_backend(root: Path) -> bool:
    """Apply the rank allocator patch and return whether the file changed."""
    backend = root / "src/srtctl/backends/vllm.py"
    source = backend.read_text()

    if NEW_BLOCK in source:
        return False
    if source.count(OLD_BLOCK) != 1:
        raise RuntimeError(
            f"unsupported srt-slurm vLLM backend at {backend}: expected allocation block not found exactly once"
        )

    backend.write_text(source.replace(OLD_BLOCK, NEW_BLOCK))
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} SRT_SLURM_CHECKOUT", file=sys.stderr)
        return 2

    try:
        changed = patch_backend(Path(argv[1]).resolve())
    except (OSError, RuntimeError) as error:
        print(f"ERROR: failed to patch srt-slurm vLLM DP ranks: {error}", file=sys.stderr)
        return 1

    state = "Patched" if changed else "Already patched"
    print(f"{state} srt-slurm vLLM DP rank allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
