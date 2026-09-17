#!/usr/bin/env python3
"""Apply the PR 52921 worker sizing changes to the c56best vLLM base."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/simple_kv_offload/worker.py"
)
source = path.read_text()

replacements = (
    (
        """from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
""",
        """from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.sizing import (
    local_num_offload_blocks,
    sync_num_offload_blocks_across_workers,
)
""",
    ),
    (
        """        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        # Use lowest priority so KV cache I/O yields to compute streams.
""",
        """        # Use lowest priority so KV cache I/O yields to compute streams.
""",
    ),
    (
        """        num_disk_slots = max(1, self.disk_capacity_bytes // total_bytes_per_block)
        self.num_cpu_blocks = num_disk_slots

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            num_disk_slots,
            (num_disk_slots * total_bytes_per_block) / (1024**3),
""",
        """        local_num_disk_slots = local_num_offload_blocks(
            self.disk_capacity_bytes, total_bytes_per_block
        )
        self.num_cpu_blocks = sync_num_offload_blocks_across_workers(
            local_num_disk_slots
        )

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
""",
    ),
    (
        """            rank_path,
            num_disk_slots,
            total_bytes_per_block,
""",
        """            rank_path,
            self.num_cpu_blocks,
            total_bytes_per_block,
""",
    ),
    (
        """    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
""",
        """    ) -> None:
        local_num_cpu_blocks = local_num_offload_blocks(
            self.cpu_capacity_bytes, total_bytes_per_block
        )
        self.num_cpu_blocks = sync_num_offload_blocks_across_workers(
            local_num_cpu_blocks
        )

        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
""",
    ),
)

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"PR 52921 worker compatibility anchor missing: {old[:80]!r}")
    source = source.replace(old, new, 1)

path.write_text(source)
print("Applied PR 52921 worker compatibility overlay")
