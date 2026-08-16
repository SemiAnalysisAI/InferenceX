#!/usr/bin/env python3
"""Backport vllm-project/vllm#50183 into the pinned nightly."""

from pathlib import Path
import os


dist = Path(os.environ.get("DIST", "/usr/local/lib/python3.12/dist-packages"))
path = dist / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
text = path.read_text()

target_old = '''    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
'''
target_new = '''    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    # See _insert_resampled_kernel: NaN breaks tl.argmax index bounds.
    local_max = tl.where(local_max != local_max, float("-inf"), local_max)
    max_block_idx = tl.argmax(local_max, axis=0)
'''

resample_old = '''    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
'''
resample_new = '''    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
'''

if "NaN breaks tl.argmax index bounds" not in text:
    if text.count(target_old) != 1 or text.count(resample_old) != 1:
        raise SystemExit(f"ERROR: vLLM #50183 anchors not found exactly once in {path}")
    text = text.replace(target_old, target_new).replace(resample_old, resample_new)
    path.write_text(text)

if text.count("NaN breaks tl.argmax index bounds") != 1:
    raise SystemExit("ERROR: target argmax NaN guard missing or duplicated")
if text.count("NaN max values (from NaN target logits)") != 1:
    raise SystemExit("ERROR: resampled argmax NaN guard missing or duplicated")
print(f"PATCH(vLLM #50183) OK: {path}")
