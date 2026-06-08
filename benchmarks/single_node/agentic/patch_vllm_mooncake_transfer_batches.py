#!/usr/bin/env python3
"""Temporarily bound MooncakeStoreConnector transfer batches.

Mooncake's TCP connection pool grows without a concurrency ceiling. Large
DeepSeek-V4 requests therefore create enough simultaneous per-layer transfers
to exhaust the node's TCP ports. This patch preserves the same keys and buffer
lists but submits them in smaller sequential batches.
"""

import argparse
from pathlib import Path


HELPER_ANCHOR = '''def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]
'''

HELPER = '''

_INFERENCEX_MOONCAKE_BATCH_PATCH = True


def _run_mooncake_transfer_batches(fn, keys, addrs, sizes, *args):
    max_keys = int(os.getenv("INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS", "0"))
    if max_keys <= 0 or len(keys) <= max_keys:
        return fn(keys, addrs, sizes, *args)

    results = []
    for start in range(0, len(keys), max_keys):
        end = start + max_keys
        results.extend(fn(keys[start:end], addrs[start:end], sizes[start:end], *args))
    return results
'''

PUT_CALL = '''res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )'''

PATCHED_PUT_CALL = '''res = _run_mooncake_transfer_batches(
                    self.store.batch_put_from_multi_buffers,
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )'''

GET_CALL = '''res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )'''

PATCHED_GET_CALL = '''res = _run_mooncake_transfer_batches(
                    self.store.batch_get_into_multi_buffers,
                    batch_keys,
                    batch_addrs,
                    batch_sizes,
                )'''


def patch_worker(worker_path: Path) -> None:
    source = worker_path.read_text()
    if "_INFERENCEX_MOONCAKE_BATCH_PATCH = True" in source:
        print(f"Mooncake transfer batching already patched: {worker_path}")
        return

    replacements = (
        (HELPER_ANCHOR, HELPER_ANCHOR + HELPER),
        (PUT_CALL, PATCHED_PUT_CALL),
        (GET_CALL, PATCHED_GET_CALL),
    )
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one patch target in {worker_path}, found {count}: "
                f"{old.splitlines()[0]}"
            )
        source = source.replace(old, new, 1)

    worker_path.write_text(source)
    print(f"Patched Mooncake transfer batching: {worker_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-path", type=Path)
    args = parser.parse_args()

    worker_path = args.worker_path
    if worker_path is None:
        import vllm

        worker_path = Path(vllm.__file__).parent / (
            "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
        )
    patch_worker(worker_path)


if __name__ == "__main__":
    main()
