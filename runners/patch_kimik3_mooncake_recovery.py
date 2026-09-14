#!/usr/bin/env python3
"""Backport vLLM #55297 to the pinned Kimi-K3 B300 Mooncake image.

Upstream: f3a831c2015d9eb6f7e600dbd2ef565166d64437
Base: 3696c772aae308f2420a8f307b0971e6986c4818
Only two hook insertion contexts differ from upstream; recovery logic is unchanged.
See docs/waiver/3088.md. Unknown or partially patched sources stop the launch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

# Relative package path, pristine SHA256, patched SHA256, exact edits.
PATCHES = (
    ('distributed/kv_transfer/kv_connector/v1/base.py',
     'bc1965431087676876f58360cd9cc07ab6c06febe6d747695f10b051fd85c412',
     'cbce0b160ca2d14c477103cf3b7f3433b2533eaa631439a693d1ffd06501342f', (
        ('''        """
        return

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.
''', '''        """
        return

    def on_load_failure(self, request_ids: set[str]) -> None:
        """Notify the connector before failed external KV loads are looked up again.

        Connectors may use this callback to make a failed external cache hit a
        request-local miss on the next scheduling attempt. The default is a
        no-op because not every connector needs special handling before the
        affected tokens are recomputed.
        """
        return

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.
'''),
    )),
    ('distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py',
     'd18e207bfce93ab53b2156902c8bdf7b223e1441c5070eddb07b4db2cb7b54b9',
     '44ea4cda5ef3dbc8c7a32b824dc43280e264f4f63e956a5197d716c2fbaf66a3', (
        ('''    def take_events(self) -> Iterable[KVCacheEvent]:
''', '''    def on_load_failure(self, request_ids: set[str]) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.on_load_failure(request_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
'''),
    )),
    ('distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py',
     'c75fc95a585ee4390963b01618c5ece1b52b30f8da95ff1a00a850948a943544',
     '3fbd7807ea1b1d55558b604c208e6158d932e57778e3d8a30705edae22bdf7bb', (
        ('''
        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
''', '''
        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        # A failed load can rewind a request to token zero. Bypass lookups
        # until local allocation succeeds so stale metadata cannot cause a livelock.
        self._load_failure_bypass_req_ids: set[str] = set()
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
'''),
        ('''        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        if not self.enable_lookup:
            return 0, False

''', '''        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        if request.request_id in self._load_failure_bypass_req_ids:
            self.load_specs.pop(request.request_id, None)
            logger.info(
                "Skipping Mooncake lookup for request %s after KV load failure",
                request.request_id,
            )
            return 0, False

        if not self.enable_lookup:
            return 0, False

'''),
        ('''
        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

        if request.request_id not in self.load_specs:
            return
''', '''
        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        self._load_failure_bypass_req_ids.discard(request.request_id)

        if request.request_id not in self.load_specs:
            return
'''),
        ('''
        for finished_req_id in scheduler_output.finished_req_ids:
            self.client.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
''', '''
        for finished_req_id in scheduler_output.finished_req_ids:
            self.client.discard(finished_req_id)
            self._load_failure_bypass_req_ids.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
'''),
        ('''    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
''', '''    def on_load_failure(self, request_ids: set[str]) -> None:
        """Skip external lookups until requests are allocated for recompute."""
        self._load_failure_bypass_req_ids.update(request_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
'''),
    )),
    ('distributed/kv_transfer/kv_connector/v1/multi_connector.py',
     'aafafbf4b0e3a43e7c864270fe56ffaa4b0f3dc343e77a0871db25f075d984dd',
     '6f4fad0450ef91a75717b9631d668379d905d07fd876325da8d72eae17cca17e', (
        ('''        for c in self._connectors:
            c.on_new_request(request)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata:
''', '''        for c in self._connectors:
            c.on_new_request(request)

    def on_load_failure(self, request_ids: set[str]) -> None:
        for c in self._connectors:
            c.on_load_failure(request_ids)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata:
'''),
    )),
    ('v1/core/sched/scheduler.py',
     '6ba2a83c7fb6078e4d1c2af7a9a2bf820f83b0570f9e1e1908294a981bf1bace',
     '9266f66969dbc0a6474bf53b4b3835c77a961ad4ffb9398ebcb368e1cb2eafc7', (
        ('''            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
''', '''            total_failed_tokens,
        )

        # Only async requests rewound to zero return to the waiting queue and
        # run the connector lookup again. Notify the connector so a persistent
        # external hit cannot make that request repeat the same failed load.
        async_relookup_req_ids = {
            req_id
            for req_id in async_failed_req_ids
            if self.requests[req_id].num_computed_tokens == 0
        }
        if self.connector is not None and async_relookup_req_ids:
            self.connector.on_load_failure(async_relookup_req_ids)

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
'''),
    )),
)


def patch_mooncake(package_root: Path) -> bool:
    """Preflight every source before writing; return False for the complete patch."""
    pending: list[tuple[Path, str]] = []
    already_patched = 0
    for relative, pristine_sha, patched_sha, edits in PATCHES:
        path = package_root / relative
        source = path.read_text()
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == patched_sha:
            already_patched += 1
            continue
        if digest != pristine_sha:
            raise RuntimeError(f"unsupported vLLM source: {path} (SHA256 {digest})")
        patched = source
        for old, new in edits:
            if patched.count(old) != 1:
                raise RuntimeError(f"unexpected patch context: {path}")
            patched = patched.replace(old, new)
        if hashlib.sha256(patched.encode()).hexdigest() != patched_sha:
            raise RuntimeError(f"unexpected patched source: {path}")
        pending.append((path, patched))
    if already_patched and pending:
        raise RuntimeError("partially patched vLLM Mooncake recovery; refusing to serve")
    for path, patched in pending:
        path.write_text(patched)
    return bool(pending)


def main() -> int:
    try:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("vllm package is not installed")
        root = Path(next(iter(spec.submodule_search_locations)))
        changed = patch_mooncake(root)
    except (OSError, RuntimeError) as error:
        print(f"ERROR: Kimi B300 Mooncake recovery patch: {error}", file=sys.stderr)
        return 1
    print("Applied" if changed else "Already applied", "vLLM #55297 Mooncake recovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
