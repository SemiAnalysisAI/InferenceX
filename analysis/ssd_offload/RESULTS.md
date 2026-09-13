# Can vLLM offload KV to SSD on B200?

One-off feasibility probe, conc 19 on the throwaway analysis config. Three
runs, each answering one question and exposing the next.

## The hardware is there and it is fast

The B200 node (`im-b200-c001`) carries 8x 3.5 TB NVMe in a 27.9 TB RAID0
(`md0`), xfs, writable from inside the container at `/raid`, 16 TB free.

| path | device | fstype | dd write | dd read |
|---|---|---|---:|---:|
| `/raid` | `/dev/md0` | xfs | 3.1 GB/s | 4.6 GB/s |
| `/ix` | 172.16.0.16:... | nfs4 | 260 MB/s | 484 MB/s |

The first run measured the NFS row by accident: the scratch picker ranked
candidates by free space and had no entry on the node's own storage, so the
3 TB network mount beat everything. `/tmp` would have been worse -- it is
1 TB of tmpfs, i.e. RAM. Any future recipe must pin `/raid` explicitly;
picking "the biggest writable filesystem" picks wrong on this node.

## The blocker is the model, not the disk

`LMCacheConnectorV1` does not subclass `SupportsHMA`, so vLLM disables the
hybrid KV cache manager whenever the connector is attached:

> Turning off hybrid kv cache manager because `--kv-transfer-config` selects
> a KV connector that does not support it. [...] To add HMA support to a KV
> connector, subclass `SupportsHMA`.

DeepSeek-V4.1-Flash has a heterogeneous KV layout, so with HMA off vLLM
tries to collapse the specs into one type during cudagraph memory profiling
and cannot:

    vllm/v1/core/kv_cache_utils.py:1838 in _promote_local_kv_cache_specs
    ValueError: Failed to promote local KV cache specs to one unified type.

All four TP ranks fail identically at `determine_available_memory`, before
any KV is allocated. This is not a flag that can be tuned around: the
connector's lack of HMA support and the model's hybrid KV are incompatible
by construction.

## Status

Unanswered: whether KV actually lands on disk and comes back. The probe has
never reached its measurement phase.

- SSD offload on B200 with **DeepSeek-V4.1-Flash via LMCache**: blocked
  upstream. Needs `SupportsHMA` on the connector.
- SSD offload on B200 **in general**: untested. A model with a uniform KV
  layout would isolate this, since the failure above is specific to hybrid
  specs.

## Two false starts, both mine

Neither was a property of the hardware, recorded so the next attempt does
not repeat them:

1. `lmcache.__version__` does not exist -- killed run 1 before the server
   started. Use `importlib.metadata.version`.
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, copied in from the
   engram driver, is rejected outright alongside the connector: the VMM
   allocator can remap KV virtual addresses to different physical pages and
   invalidate pinned connector memory.

Runs: 34738868544 (NFS), 34739092558 (expandable_segments), 34739280279 (HMA).
