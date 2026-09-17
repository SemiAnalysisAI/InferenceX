#!/usr/bin/env python3
"""Backport vLLM PR 56621 to the c56best Simple CPU offload worker."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/simple_kv_offload/worker.py"
)
source = path.read_text()

replacements = (
    (
        """        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
""",
        """        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None
        self._store_submitted = False

        # Compute-done event recorded before each store; reused across steps
""",
    ),
    (
        """    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
""",
        """    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        self._store_submitted = False
        if metadata.load_event >= 0:
""",
    ),
    (
        """    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None
""",
        """    def clear_connector_metadata(self) -> None:
        # No-forward steps skip the normal wait_for_save hook.
        self.wait_for_save()
        self._connector_metadata = None
""",
    ),
    (
        """        metadata = self._connector_metadata
        if metadata is not None and metadata.store_gpu_blocks:
            backend = self._backend
""",
        """        metadata = self._connector_metadata
        if (
            metadata is not None
            and metadata.store_gpu_blocks
            and not self._store_submitted
        ):
            backend = self._backend
""",
    ),
    (
        """                events_list=self._store_events,
                wait_event=self._store_compute_done,
            )

    def get_finished(
""",
        """                events_list=self._store_events,
                wait_event=self._store_compute_done,
            )
            self._store_submitted = True

    def get_finished(
""",
    ),
)

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"PR 56621 compatibility anchor missing: {old[:80]!r}")
    source = source.replace(old, new, 1)

path.write_text(source)
print("Applied PR 56621 no-forward CPU store backport")
