#!/usr/bin/env python3
"""Fail-closed privacy check for CollectiveX public result documents."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import stat


SENSITIVE_FIELDS = frozenset({
    "environment", "env", "host", "hostname", "uuid", "gpu_uuid", "device_uuid",
    "pci_bus_id", "ip_address", "ip_addresses", "master_addr", "ssh", "ssh_target",
    "nodelist", "node_list", "nic_guid", "ib_guid", "topology_matrix", "rdma_devices",
    "user", "username", "password", "passwd", "secret", "token", "access_token",
    "api_token", "auth_token", "api_key", "private_key", "credential", "credentials",
    "address", "addresses", "ip", "ips",
})
SENSITIVE_FIELDS_COMPACT = frozenset(item.replace("_", "") for item in SENSITIVE_FIELDS)
SENSITIVE_FIELD_SUFFIXES = (
    "_host", "_hostname", "_address", "_addresses", "_path", "_paths", "_ip", "_ips",
    "_password", "_passwd", "_secret", "_token", "_credential", "_credentials",
    "_uuid", "_guid", "_bus_id",
)
SENSITIVE_VALUE_PATTERNS = (
    ("private-path", re.compile(
        r"(?<![A-Za-z0-9_.-])/(?:home|mnt|workspace|root|users|tmp|data|it-share|lustre|raid|nvme_home|scratch|gpfs|fsx)(?:/|$)",
        re.I,
    )),
    ("ipv4-address", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("pci-address", re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b", re.I)),
    ("hardware-address", re.compile(
        r"\b(?:[0-9a-f]{2}[:-]){5}(?:[0-9a-f]{2})\b|"
        r"\b(?:[0-9a-f]{2}:){7}(?:[0-9a-f]{2})\b|\b0x[0-9a-f]{16}\b",
        re.I,
    )),
    ("uuid", re.compile(
        r"\b(?:GPU-|MIG-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.I,
    )),
    ("ssh-target", re.compile(r"(?:ssh://|\bssh\s+[^\s/@]+@[^\s/]+)", re.I)),
    ("host-identifier", re.compile(
        r"\b(?:host(?:name)?|master[_-]?(?:addr|address)|node[_-]?list)\s*(?:=|:)\s*[^\s,;]+",
        re.I,
    )),
    ("private-hostname", re.compile(
        r"\b(?:[a-z0-9-]+\.)+(?:cluster|corp|internal|lan|local)\b|"
        r"\b(?:compute|gpu|head|login|node|worker)[-_]?[0-9][a-z0-9_.-]*\b|"
        r"\bdgx-[a-z0-9-]+-[0-9]+\b|\bip-(?:[0-9]{1,3}-){3}[0-9]{1,3}\b",
        re.I,
    )),
    ("secret-token", re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
        r"(?:AKIA|ASIA)[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
        r"(?:sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|"
        r"sk_(?:live|test)_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})|"
        r"npm_[A-Za-z0-9]{20,}|"
        r"pypi-[A-Za-z0-9_-]{20,}|dckr_pat_[A-Za-z0-9_-]{20,}|"
        r"Bearer\s+[A-Za-z0-9._~+/-]{16,}|Basic\s+[A-Za-z0-9+/=]{16,}|"
        r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|"
        r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----)",
        re.I,
    )),
    ("secret-assignment", re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd|secret|accountkey)\s*(?:=|:)\s*[\"']?"
        r"[A-Za-z0-9+/_=.~-]{8,}",
        re.I,
    )),
)
IPV6_CANDIDATE = re.compile(
    r"(?<![0-9A-Za-z])\[?([0-9A-Fa-f:]{2,}(?:%[0-9A-Za-z_.-]+)?)\]?"
)
CONTEXTUAL_VALUE_RULES = frozenset({"ssh-target", "host-identifier", "private-hostname"})
MAX_INPUT_BYTES = 64 * 1024 * 1024


class ArtifactSafetyError(ValueError):
    """A document contains data that cannot cross the public boundary."""


def _normalized_field(value: object) -> str:
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(value).strip())
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return normalized.lower().replace("-", "_")


def _sensitive_value_rule(value: str, *, contextual: bool = True) -> str | None:
    matched = next(
        (
            name for name, pattern in SENSITIVE_VALUE_PATTERNS
            if (contextual or name not in CONTEXTUAL_VALUE_RULES) and pattern.search(value)
        ),
        None,
    )
    if matched:
        return matched
    for candidate in IPV6_CANDIDATE.findall(value):
        try:
            address = candidate.split("%", 1)[0]
            if ipaddress.ip_address(address).version == 6:
                return "ipv6-address"
        except ValueError:
            continue
    return None


def assert_publication_safe(docs: list[dict]) -> None:
    """Reject private infrastructure fields and value shapes."""
    def walk(value, doc_index: int, parent_field: str | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                field = _normalized_field(key)
                compact = field.replace("_", "")
                if (
                    field in SENSITIVE_FIELDS
                    or compact in SENSITIVE_FIELDS_COMPACT
                    or field.endswith(SENSITIVE_FIELD_SUFFIXES)
                ):
                    raise ArtifactSafetyError(
                        f"artifact safety: doc[{doc_index}] contains forbidden private field"
                    )
                key_rule = _sensitive_value_rule(str(key))
                if key_rule:
                    raise ArtifactSafetyError(
                        f"artifact safety: doc[{doc_index}] contains forbidden {key_rule} key"
                    )
                walk(child, doc_index, field)
        elif isinstance(value, list):
            for child in value:
                walk(child, doc_index, parent_field)
        elif isinstance(value, str):
            rule = _sensitive_value_rule(value, contextual=parent_field != "ref")
            if rule:
                raise ArtifactSafetyError(
                    f"artifact safety: doc[{doc_index}] contains forbidden {rule} value"
                )

    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ArtifactSafetyError(f"artifact safety: doc[{index}] is not a JSON object")
        walk(doc, index)


def load_documents(paths: list[str]) -> list[dict]:
    docs: list[dict] = []
    for path in paths:
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise ArtifactSafetyError("artifact safety: result file is unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 0
            or metadata.st_size > MAX_INPUT_BYTES
        ):
            raise ArtifactSafetyError("artifact safety: result file is unavailable")
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_size)
                != (metadata.st_dev, metadata.st_ino, metadata.st_size)
            ):
                raise ArtifactSafetyError("artifact safety: result file changed during open")
            with os.fdopen(descriptor, encoding="utf-8") as fh:
                descriptor = -1
                if path.endswith(".ndjson"):
                    for line_number, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        try:
                            docs.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ArtifactSafetyError(
                                f"artifact safety: malformed NDJSON at input line {line_number}"
                            ) from exc
                else:
                    docs.append(json.load(fh))
        except json.JSONDecodeError as exc:
            raise ArtifactSafetyError("artifact safety: malformed JSON input") from exc
        except (OSError, UnicodeError) as exc:
            raise ArtifactSafetyError("artifact safety: result file is unreadable") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    if not docs:
        raise ArtifactSafetyError("artifact safety: no public result documents found")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CollectiveX result artifacts for private data")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    try:
        docs = load_documents(args.paths)
        assert_publication_safe(docs)
    except ArtifactSafetyError as exc:
        parser.error(str(exc))
    print(f"artifact safety: {len(docs)} public document(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
