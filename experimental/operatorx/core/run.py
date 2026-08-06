from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunInfo:
    id: str
    started_at: str
    finished_at: str
    cluster: str | None
    operatorx_version: str
    operatorx_git_sha: str | None
    hostname: str
    instance_type: str | None
    software: dict[str, str]
    container_image: str | None
    env: dict[str, str]


def to_dict(r: RunInfo) -> dict:
    return asdict(r)


def from_dict(d: dict) -> RunInfo:
    return RunInfo(**d)
