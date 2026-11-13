#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Iterable
import re
import sys


_GROUP = r"[0-9A-Fa-f]{1,4}"
# A dotted quad that is not part of a longer dotted number or word (v1.2.3.4, 1.2.3.4.5) and not
# a pinned package version (name==1.2.3.4).
_IPV4 = re.compile(r"(?<![0-9A-Za-z.])(?<!==)(?:\d{1,3}\.){3}\d{1,3}(?![0-9A-Za-z]|\.\d)")
# All eight groups, or a compressed address whose first group has three or four digits, as every
# routable, link-local and unique-local prefix does. A shorter first group would also match Python
# slices (x[1::2]); the delimiters keep C++ scopes (std::runtime_error) and clock times out.
_IPV6 = re.compile(
    rf"(?<![0-9A-Za-z_:.])(?:(?:{_GROUP}:){{7}}{_GROUP}"
    rf"|[0-9A-Fa-f]{{3,4}}(?::{_GROUP}){{0,6}}::(?:{_GROUP}(?::{_GROUP}){{0,6}})?)"
    r"(?![0-9A-Za-z_:])"
)
_MAC = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])")


def _hosts(names: Iterable[str]) -> re.Pattern[str] | None:
    stems = set()
    for name in names:
        label = name.strip().split(".", 1)[0]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) or label.isdigit():
            continue
        stem = label.rstrip("0123456789")
        # Slurm prints a node set as stem[ranges], and the pool's other nodes share the stem.
        stems.add(re.escape(stem) + r"(?:\d+|\[[0-9,-]+\])" if stem != label else re.escape(label))
    if not stems:
        return None
    # The tail takes a domain or an interface alias (node01.cluster.example, node01-ib0) with it.
    return re.compile(
        r"(?<![A-Za-z0-9_.-])(?:" + "|".join(sorted(stems, key=len, reverse=True))
        + r")(?:[.-][A-Za-z0-9]+)*(?![A-Za-z0-9_])"
    )


def redact(text: str, hosts: Iterable[str] = ()) -> str:
    """Mask the host identifiers in log text that is about to be shown publicly: IPv4, IPv6 and
    MAC literals, and the given host names with the names Slurm derives from them."""
    text = _IPV4.sub("<ipv4>", text)
    text = _IPV6.sub("<ipv6>", text)
    text = _MAC.sub("<mac>", text)
    pattern = _hosts(hosts)
    return pattern.sub("<host>", text) if pattern else text


def main() -> None:
    hosts = [name for argument in sys.argv[1:] for name in argument.split(",")]
    # surrogateescape passes bytes that are not UTF-8 through unchanged.
    text = sys.stdin.buffer.read().decode("utf-8", "surrogateescape")
    sys.stdout.buffer.write(redact(text, hosts).encode("utf-8", "surrogateescape"))


if __name__ == "__main__": main()
