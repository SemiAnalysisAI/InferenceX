#!/usr/bin/env python3
"""CPU test for the causal row map in vllm_pr50619_native_mtp.py.

The fallback path is the part that is not a straight copy of
vllm-project/vllm#50619 (the PR raises on mixed query lengths; this image has
no uniform-padding machinery, so it flattens causally instead). Check it
against a literal reference: request r's verify token t must attend exactly the
tokens ``[0, seq_len_r - qo_len_r + t]`` of request r's own KV slice.

Runs on CPU, no GPU and no vllm import: the helper source is lifted out of the
patcher and exec'd with torch alone.
"""

import sys
import types

import torch

import vllm_pr50619_native_mtp as patcher


def _load_helper():
    ns: dict = {"torch": torch}
    exec(patcher.HELPER, ns)
    return ns["_pr50619_causal_flatten"]


def _fake_decode(seq_lens, qo_lens):
    """Metadata shaped like AiterMLADecodeMetadata's token-level paged KV.

    paged_kv_indptr is cumsum(seq_lens) and paged_kv_indices holds token-level
    entries, matching _expand_page_indices_kernel's flat view. Give each
    request a recognizable index range so a wrong row maps to a wrong request.
    """
    indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.tensor(seq_lens).cumsum(0).int()]
    )
    indices = torch.cat(
        [
            torch.arange(s, dtype=torch.int32) + 1000 * (r + 1)
            for r, s in enumerate(seq_lens)
        ]
        or [torch.zeros(0, dtype=torch.int32)]
    )
    d = types.SimpleNamespace()
    d.paged_kv_indptr = indptr
    d.paged_kv_indices = indices
    d.mtp_qo_len = torch.tensor(qo_lens, dtype=torch.int64)
    return d


def check(name, seq_lens, qo_lens):
    flatten = _load_helper()
    d = _fake_decode(seq_lens, qo_lens)
    num_rows = int(sum(qo_lens))
    new_indices, new_indptr, min_len = flatten(d, num_rows)

    assert min_len == 1, f"{name}: expected single-split, got {min_len}"
    assert new_indptr.numel() == num_rows + 1, f"{name}: indptr length"

    row = 0
    for r, (s, q) in enumerate(zip(seq_lens, qo_lens)):
        for t in range(q):
            lo, hi = int(new_indptr[row]), int(new_indptr[row + 1])
            got = new_indices[lo:hi].tolist()
            # Causal: context_r = s - q, so token t sees context_r + 1 + t.
            want_len = max(0, s - q + 1 + t)
            want = [1000 * (r + 1) + i for i in range(want_len)]
            assert got == want, (
                f"{name}: request {r} token {t}\n  want {want[:4]}..{want[-2:]} "
                f"(len {len(want)})\n  got  {got[:4]}..{got[-2:]} (len {len(got)})"
            )
            # Independent of `want`: every index must belong to request r, and
            # none may reach the request's own future draft tokens.
            if got:
                positions = [i - 1000 * (r + 1) for i in got]
                assert min(positions) >= 0 and max(positions) < s, (
                    f"{name}: row {row} escaped request {r}'s KV slice"
                )
                assert max(positions) < s - q + 1 + t, (
                    f"{name}: row {row} reaches position {max(positions)}, "
                    f"past its causal limit {s - q + t}"
                )
            row += 1
    print(f"ok  {name}: {num_rows} rows, {len(seq_lens)} reqs")


def check_leak_contrast():
    """The image's row map is non-causal; show the test would catch it."""
    seq_lens, qo_lens = [40], [8]
    d = _fake_decode(seq_lens, qo_lens)
    flatten = _load_helper()
    new_indices, new_indptr, _ = flatten(d, 8)
    lens = [int(new_indptr[i + 1] - new_indptr[i]) for i in range(8)]
    assert lens == [33, 34, 35, 36, 37, 38, 39, 40], lens
    # Non-causal (image) would be [40]*8: every row seeing all 8 draft tokens.
    assert lens != [40] * 8
    print(f"ok  causal staircase: {lens} (image emits {[40] * 8})")


if __name__ == "__main__":
    check("uniform k=7 verify, 4 reqs", [4096, 300, 8, 65536], [8, 8, 8, 8])
    check("mixed: fresh request joins at 1 token", [4096, 12, 300], [8, 1, 8])
    check("padding slots with seq_len 0", [4096, 0, 0], [8, 8, 8])
    check("qlen == seq_len (no context yet)", [8, 8], [8, 8])
    check("single request, conc 1", [327680], [8])
    check_leak_contrast()
    print("\nall causal row-map checks passed")
    sys.exit(0)
