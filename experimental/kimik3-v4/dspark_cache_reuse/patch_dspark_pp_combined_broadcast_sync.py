#!/usr/bin/env python3
"""Add a correctness-first synchronous mode for the combined PP packet."""

from pathlib import Path
import py_compile


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu/pp_utils.py"
)
source = path.read_text()

replacements = (
    (
        """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
""",
        """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            if os.environ.get("K3_PP_DRAFT_SYNC", "0") == "1":
                # K3 PP synchronous combined packet: prove device completion
                # before publishing the deferred receive slot.
                self.broadcast_stream.synchronize()
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
""",
    ),
    (
        """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
""",
        """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            if os.environ.get("K3_PP_DRAFT_SYNC", "0") == "1":
                # K3 PP synchronous combined packet: wait for both token and
                # count payloads before returning from sample_tokens.
                self.broadcast_stream.synchronize()
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
""",
    ),
)

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"PP combined sync patch anchor missing: {old[:100]!r}")
    source = source.replace(old, new, 1)

path.write_text(source)
py_compile.compile(str(path), doraise=True)
print("K3 PP combined packet synchronous mode ready")
