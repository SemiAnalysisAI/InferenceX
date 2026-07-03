#!/usr/bin/env python3
"""Fail-closed privacy check for CollectiveX public result documents."""
from __future__ import annotations

import argparse
import json
import os
import re


SENSITIVE_FIELDS = frozenset({
    "environment", "env", "host", "hostname", "uuid", "gpu_uuid", "device_uuid",
    "pci_bus_id", "ip_address", "ip_addresses", "master_addr", "ssh", "ssh_target",
    "nodelist", "node_list", "nic_guid", "ib_guid", "topology_matrix", "rdma_devices",
})
SENSITIVE_VALUE_PATTERNS = (
    ("private-path", re.compile(r"(?:^|[\s=:])/(?:home|mnt|workspace|root|Users|tmp)/")),
    ("ipv4-address", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("ipv6-or-pci-address", re.compile(
        r"(?:\b[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){2,}\b|(?<![\w:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f]{0,4}(?![\w:]))",
        re.I,
    )),
    ("uuid", re.compile(
        r"\b(?:GPU-|MIG-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.I,
    )),
    ("ssh-target", re.compile(r"(?:ssh://|\b[^\s/@]+@[^\s/]+)")),
    ("host-identifier", re.compile(r"\b(?:host|hostname|master_addr|nodelist)=[^\s]+", re.I)),
    ("secret-token", re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{16,}|AKIA[0-9A-Z]{16})"
    )),
)


def _normalized_field(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _sensitive_value_rule(value: str) -> str | None:
    return next((name for name, pattern in SENSITIVE_VALUE_PATTERNS if pattern.search(value)), None)


def assert_publication_safe(docs: list[dict]) -> None:
    """Reject private infrastructure fields and value shapes."""
    def walk(value, doc_index: int) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                field = _normalized_field(key)
                if field in SENSITIVE_FIELDS:
                    raise SystemExit(
                        f"artifact safety: doc[{doc_index}] contains forbidden field {field!r}"
                    )
                walk(child, doc_index)
        elif isinstance(value, list):
            for child in value:
                walk(child, doc_index)
        elif isinstance(value, str):
            rule = _sensitive_value_rule(value)
            if rule:
                raise SystemExit(
                    f"artifact safety: doc[{doc_index}] contains forbidden {rule} value"
                )

    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise SystemExit(f"artifact safety: doc[{index}] is not a JSON object")
        walk(doc, index)


def load_documents(paths: list[str]) -> list[dict]:
    docs: list[dict] = []
    for path in paths:
        if os.path.basename(path).startswith("env_"):
            continue
        if not os.path.isfile(path):
            raise SystemExit(f"artifact safety: result file not found: {path}")
        with open(path) as fh:
            if path.endswith(".ndjson"):
                for line_number, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"artifact safety: malformed NDJSON at {path}:{line_number}: {exc}"
                        ) from exc
            else:
                try:
                    docs.append(json.load(fh))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"artifact safety: malformed JSON at {path}: {exc}") from exc
    if not docs:
        raise SystemExit("artifact safety: no public result documents found")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CollectiveX result artifacts for private data")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    docs = load_documents(args.paths)
    assert_publication_safe(docs)
    print(f"artifact safety: {len(docs)} public document(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
