#!/usr/bin/env python3
"""Cap MooncakeStoreConnector BatchGet/BatchPut key counts per RPC.

Reads INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS (e.g. 32) to split large
transfer bursts that exhaust TCP ephemeral ports under high concurrency.

Upstream vLLM only splits GET batches for disk-offload staging budgets; memory
tier loads use a single sub-batch with all keys (~1000+ at agentic conc=32).
"""
from __future__ import annotations

import os
from pathlib import Path

PATCH_MARKER = "INFERENCEX_MOONCAKE_TRANSFER_BATCH"

HELPER_CODE = f'''
def _inferencex_max_transfer_batch_keys() -> int | None:  # {PATCH_MARKER}
    raw = os.environ.get("INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS", "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _split_transfer_load_batches_by_key_count(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    block_ids: list[int],
    max_keys: int,
) -> list[tuple[list[str], list[list[int]], list[list[int]], list[int]]]:
    if max_keys <= 0 or len(keys) <= max_keys:
        return [(keys, addrs, sizes, block_ids)]
    batches: list[
        tuple[list[str], list[list[int]], list[list[int]], list[int]]
    ] = []
    for start in range(0, len(keys), max_keys):
        end = start + max_keys
        batches.append(
            (keys[start:end], addrs[start:end], sizes[start:end], block_ids[start:end])
        )
    return batches


def _split_transfer_put_batches_by_key_count(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    max_keys: int,
) -> list[tuple[list[str], list[list[int]], list[list[int]]]]:
    if max_keys <= 0 or len(keys) <= max_keys:
        return [(keys, addrs, sizes)]
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    for start in range(0, len(keys), max_keys):
        end = start + max_keys
        batches.append((keys[start:end], addrs[start:end], sizes[start:end]))
    return batches

'''

RECV_ANCHOR = (
    "load_batches = [(key_list_c, addr_list_c, size_list_c, block_id_list_c)]"
)
RECV_INSERT = f"""
        _ix_max_keys = _inferencex_max_transfer_batch_keys()  # {PATCH_MARKER}
        if (
            _ix_max_keys is not None
            and self.usable_disk_offload_buffer_budget_bytes is None
            and len(key_list_c) > _ix_max_keys
        ):
            load_batches = _split_transfer_load_batches_by_key_count(
                key_list_c,
                addr_list_c,
                size_list_c,
                block_id_list_c,
                _ix_max_keys,
            )"""

PUT_OLD_V0231 = """            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                failed = [i for i, v in enumerate(res) if v < 0]
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    failed_codes = set(res[i] for i in failed)
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d, num_keys=%d), "
                        "first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        len(keys),
                        keys[0] if keys else "N/A",
                    )
                    if (
                        MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                        and not self._mark_request_skipped_for_pressure(req_id)
                    ):
                        logger.warning(
                            "Detected Mooncake CPU/disk offloading pressure "
                            "(NO_AVAILABLE_HANDLE); skipping future store "
                            "batches for request %s until a later store "
                            "batch succeeds",
                            req_id,
                        )
                else:
                    self._record_saved(req_id, token_len)
                    if self._clear_store_pressure():
                        logger.info(
                            "Mooncake CPU/disk offloading pressure cleared "
                            "after a successful store batch"
                        )
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.error("Failed to put key %s, error: %s", keys, e)"""

PUT_OLD_LEGACY = """            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                failed = [i for i, v in enumerate(res) if v < 0]
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    failed_codes = set(res[i] for i in failed)
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d, num_keys=%d), "
                        "first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        len(keys),
                        keys[0] if keys else "N/A",
                    )
                    if (
                        MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                        and not self._mark_request_skipped_for_pressure(req_id)
                    ):
                        logger.warning(
                            "Detected Mooncake CPU/disk offloading pressure "
                            "(NO_AVAILABLE_HANDLE); skipping future store "
                            "batches for request %s until a later store "
                            "batch succeeds",
                            req_id,
                        )
                elif self._clear_store_pressure():
                    logger.info(
                        "Mooncake CPU/disk offloading pressure cleared after a "
                        "successful store batch"
                    )
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.error("Failed to put key %s, error: %s", keys, e)"""

PUT_LOOP_BODY = f"""                batch_bytes = _sum_batch_bytes(sizes)
                put_start = time.perf_counter()
                try:
                    res = self.store.batch_put_from_multi_buffers(
                        keys,
                        addrs,
                        sizes,
                        self.replicate_config,
                    )
                    failed = [i for i, v in enumerate(res) if v < 0]
                    self._record_operation(
                        "save_put",
                        put_start,
                        len(keys),
                        num_bytes=batch_bytes,
                        status="partial_failure" if failed else "ok",
                        num_failed_keys=len(failed),
                    )
                    if failed:
                        failed_codes = set(res[i] for i in failed)
                        logger.warning(
                            "batch_put failed: %d/%d keys failed "
                            "(codes=%s, batch_bytes=%d, num_keys=%d), "
                            "first_key=%s",
                            len(failed),
                            len(keys),
                            failed_codes,
                            batch_bytes,
                            len(keys),
                            keys[0] if keys else "N/A",
                        )
                        if (
                            MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                            and not self._mark_request_skipped_for_pressure(req_id)
                        ):
                            logger.warning(
                                "Detected Mooncake CPU/disk offloading pressure "
                                "(NO_AVAILABLE_HANDLE); skipping future store "
                                "batches for request %s until a later store "
                                "batch succeeds",
                                req_id,
                            )
                        _ix_put_failed = True
                        break
                except Exception as e:
                    self._record_operation(
                        "save_put",
                        put_start,
                        len(keys),
                        num_bytes=batch_bytes,
                        status="error",
                        num_failed_keys=len(keys),
                    )
                    logger.error("Failed to put key %s, error: %s", keys, e)
                    _ix_put_failed = True
                    break"""

PUT_NEW_V0231 = f"""            _ix_put_max_keys = _inferencex_max_transfer_batch_keys()  # {PATCH_MARKER}
            if _ix_put_max_keys is not None:
                put_batches = _split_transfer_put_batches_by_key_count(
                    keys, addrs, sizes, _ix_put_max_keys
                )
            else:
                put_batches = [(keys, addrs, sizes)]

            _ix_put_failed = False
            for keys, addrs, sizes in put_batches:
{PUT_LOOP_BODY}
            if not _ix_put_failed:
                self._record_saved(req_id, token_len)
                if self._clear_store_pressure():
                    logger.info(
                        "Mooncake CPU/disk offloading pressure cleared "
                        "after a successful store batch"
                    )"""

PUT_NEW_LEGACY = f"""            _ix_put_max_keys = _inferencex_max_transfer_batch_keys()  # {PATCH_MARKER}
            if _ix_put_max_keys is not None:
                put_batches = _split_transfer_put_batches_by_key_count(
                    keys, addrs, sizes, _ix_put_max_keys
                )
            else:
                put_batches = [(keys, addrs, sizes)]

            _ix_put_failed = False
            for keys, addrs, sizes in put_batches:
{PUT_LOOP_BODY}
            if not _ix_put_failed and self._clear_store_pressure():
                logger.info(
                    "Mooncake CPU/disk offloading pressure cleared after a "
                    "successful store batch"
                )"""

PUT_PATCH_VARIANTS: list[tuple[str, str, str]] = [
    ("v0.23.1", PUT_OLD_V0231, PUT_NEW_V0231),
    ("legacy", PUT_OLD_LEGACY, PUT_NEW_LEGACY),
]

HELPER_ANCHOR = "def _sum_batch_bytes(sizes: list[list[int]]) -> int:"


def find_worker_py() -> Path:
    import vllm

    return (
        Path(vllm.__file__).resolve().parent
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "mooncake"
        / "store"
        / "worker.py"
    )


def _select_put_patch(text: str) -> tuple[str, str, str] | None:
    for label, put_old, put_new in PUT_PATCH_VARIANTS:
        if put_old in text:
            return label, put_old, put_new
    return None


def apply_patch(path: Path) -> None:
    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"[patch-mooncake-batch] Already applied: {path}")
        return

    if HELPER_ANCHOR not in text:
        raise SystemExit(f"[patch-mooncake-batch] helper anchor not found in {path}")
    if RECV_ANCHOR not in text:
        raise SystemExit(f"[patch-mooncake-batch] recv anchor not found in {path}")

    put_patch = _select_put_patch(text)
    if put_patch is None:
        raise SystemExit(f"[patch-mooncake-batch] put block not found in {path}")

    put_label, put_old, put_new = put_patch

    max_keys = os.environ.get("INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS", "").strip()
    if not max_keys:
        raise SystemExit(
            "[patch-mooncake-batch] set INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS"
        )

    text = text.replace(HELPER_ANCHOR, HELPER_CODE + HELPER_ANCHOR, 1)
    text = text.replace(RECV_ANCHOR, RECV_ANCHOR + RECV_INSERT, 1)
    text = text.replace(put_old, put_new, 1)
    if PATCH_MARKER not in text:
        raise SystemExit(f"[patch-mooncake-batch] patch application failed for {path}")

    path.write_text(text)
    print(
        f"[patch-mooncake-batch] Patched {path} "
        f"(variant={put_label}, "
        f"INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS={max_keys})"
    )


def main() -> None:
    path = find_worker_py()
    if not path.is_file():
        raise SystemExit(f"[patch-mooncake-batch] worker.py not found: {path}")
    apply_patch(path)


if __name__ == "__main__":
    main()
