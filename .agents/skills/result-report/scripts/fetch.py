"""Find benchmark configs in a GitHub Actions run or a local folder, and read their files.

A config is one benchmark job: the small aggregate JSON (artifact ``bmk_agentic_<RF>``)
plus the raw results tree (artifact ``agentic_<RF>``). Remote artifact zips are read
member by member over HTTP Range requests, so only the needed bytes are downloaded.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "vngcloud/InferenceX"
AGG_PREFIX = "bmk_agentic_"
KNOWN_OTHER = ("power_audit_", "gpu_metrics_", "server_logs_", "results_bmk", "run-stats")  # look-alike artifacts, ignored silently
RAW_PREFIX = "agentic_"
RUN_URL_RE = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/actions/runs/(\d+)")
FETCHED = {"bytes": 0}  # downloaded bytes of this process, reported by the CLI
FAILED_JOBS: list[dict] = []  # full details of failed jobs (notes only carry a short tail)


def progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Source:
    kind: str  # "run" or "dir"
    repo: str | None = None
    run_id: str | None = None
    path: Path | None = None

    @property
    def link(self) -> str:
        if self.kind == "run":
            return f"https://github.com/{self.repo}/actions/runs/{self.run_id}"
        return str(self.path)


def parse_source(arg: str, default_repo: str = DEFAULT_REPO) -> Source:
    match = RUN_URL_RE.search(arg)
    if match:
        return Source("run", repo=match.group(1), run_id=match.group(2))
    if arg.isdigit():
        return Source("run", repo=default_repo, run_id=arg)
    path = Path(arg).expanduser()
    if path.is_dir():
        return Source("dir", path=path.resolve())
    raise ValueError(f"not a run URL, run id or directory: {arg}")


class DirReader:
    """Read files under one local results root."""

    size_kind = "on disk"

    def __init__(self, root: Path):
        self.root = root

    def names(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def read(self, rel: str) -> bytes | None:
        path = self.root / rel
        return path.read_bytes() if path.is_file() else None

    def size(self, rel: str) -> int | None:
        """Bytes the member costs to read (file size); None if absent."""
        path = self.root / rel
        return path.stat().st_size if path.is_file() else None


@dataclass
class Config:
    rf: str  # pairing key: artifact suffix <RF>, or a relative path for hand-named folders
    source: Source
    agg: dict | None  # aggregate JSON
    files: object | None  # DirReader / ZipReader over the results root


def _is_aggregate(doc: object) -> bool:
    return isinstance(doc, dict) and "infmax_model_prefix" in doc and ("request_metrics" in doc or "state" in doc)


def _key(rel: Path, prefix: str) -> str:
    """GH artifact folders pair on their <RF> suffix; anything else keys on its relative path."""
    for part in reversed(rel.parts):
        if part.startswith(prefix):
            return part[len(prefix):]
    return rel.as_posix()


def local_configs(src: Source, notes: list[str] | None = None) -> list[Config]:
    root = src.path
    aggs: dict[str, dict] = {}
    ignored: list[str] = []
    used_tops: set[str] = set()  # top-level folders that hold a result
    single = False  # the root itself is one run folder
    # multi-artifact download: only bmk_agentic_*/ holds aggregates (power_audit_*, results_bmk, run-stats look alike)
    multi = any(p.is_dir() and p.name.startswith(AGG_PREFIX) for p in root.iterdir())
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root)
        outside = multi and not rel.parts[0].startswith(AGG_PREFIX)
        if any(p.startswith("search_iter_") or p in ("aiperf_artifacts", "aggregate") for p in rel.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if _is_aggregate(doc):
            if outside:
                if not rel.parts[0].startswith(KNOWN_OTHER):
                    ignored.append(rel.parts[0])
            else:
                aggs[_key(rel, AGG_PREFIX)] = doc
                if len(rel.parts) == 1:
                    single = True
                else:
                    used_tops.add(rel.parts[0])
    roots: dict[str, Path] = {}
    for name in ("profile_export_aiperf.json", "search_history.json"):
        for path in sorted(root.rglob(name)):
            rel = path.relative_to(root)
            if any(p.startswith("search_iter_") or p == "aggregate" for p in rel.parts):
                continue
            base = path.parent.parent if path.parent.name == "aiperf_artifacts" else path.parent
            roots.setdefault(_key(base.relative_to(root), RAW_PREFIX), base)
            if base == root:
                single = True
            else:
                used_tops.add(base.relative_to(root).parts[0])
    if ignored and notes is not None:
        notes.append(f"ignored {len(ignored)} aggregate-looking file(s) outside {AGG_PREFIX}*: " + ", ".join(sorted(set(ignored))))
    if notes is not None and not single and (aggs or roots):
        empty = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
                       and p.name not in used_tops and p.name not in set(ignored)
                       and not p.name.startswith(KNOWN_OTHER))
        if empty:
            notes.append(f"skipped {len(empty)} folder(s) with no results: " + ", ".join(empty))
    lone_aggs = [k for k in aggs if k not in roots]
    lone_roots = [k for k in roots if k not in aggs]
    if len(lone_aggs) == 1 and len(lone_roots) == 1:  # e.g. hand-named ag/ + bmk/ copies
        roots[lone_aggs[0]] = roots.pop(lone_roots[0])
    return [
        Config(rf=k, source=src, agg=aggs.get(k), files=DirReader(roots[k]) if k in roots else None)
        for k in sorted(set(aggs) | set(roots))
    ]


MIN_FETCH = 256 * 1024  # bytes per Range request; zipfile asks for small chunks
FAILED_CONCLUSIONS = {"failure", "cancelled", "timed_out"}


class TooLarge(Exception):
    """Range reads failed and the full artifact zip is over the --max-full-zip-mb budget."""


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests: enough for zipfile.ZipFile."""

    def __init__(self, url: str):
        self.url, self.pos, self.fetched = url, 0, 0
        self._start, self._buf = 0, b""
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as resp:
            if resp.status != 206:
                raise OSError(f"server ignored Range (HTTP {resp.status})")
            self.size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self._start <= self.pos and self.pos + n <= self._start + len(self._buf)):
            end = min(self.pos + max(n, MIN_FETCH), self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            with urllib.request.urlopen(req) as resp:
                if resp.status != 206:  # a 200 carries the whole body, not our slice
                    raise OSError(f"server ignored Range (HTTP {resp.status})")
                self._buf = resp.read()
            self._start = self.pos
            self.fetched += len(self._buf)
            FETCHED["bytes"] += len(self._buf)
        offset = self.pos - self._start
        data = self._buf[offset:offset + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _gh_pages(path: str, key: str) -> list[dict]:
    items, page = [], 1
    while True:
        doc = json.loads(_gh("api", f"{path}?per_page=100&page={page}"))
        items += doc[key]
        if not doc[key] or len(items) >= doc["total_count"]:
            return items
        page += 1


def blob_url(repo: str, artifact_id: int) -> str:
    """The short-lived blob URL the artifact zip endpoint redirects to (it supports Range)."""
    token = _gh("auth", "token").strip()
    request = urllib.request.Request(
        f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        urllib.request.build_opener(_NoRedirect).open(request)
    except urllib.error.HTTPError as err:
        if err.code in (301, 302, 303, 307, 308):
            err.close()
            return err.headers["Location"]
        raise
    raise OSError(f"artifact {artifact_id}: expected a redirect to the zip blob")


def _atomic_write(path: Path, data: bytes | str) -> None:
    """Write via <name>.tmp + os.replace so a killed run never leaves a truncated cache entry."""
    tmp = path.with_name(path.name + ".tmp")
    (tmp.write_bytes if isinstance(data, bytes) else tmp.write_text)(data)
    os.replace(tmp, path)


class ZipReader:
    """Members of one remote artifact zip, each cached under cache_dir/<artifact id>/."""

    def __init__(self, repo: str, artifact: dict, cache_dir: Path, max_full_zip_mb: float, url_for=None):
        self.repo, self.artifact, self.max_mb = repo, artifact, max_full_zip_mb
        self.cache = cache_dir / str(artifact["id"])
        self._url_for = url_for
        self._zip: zipfile.ZipFile | None = None
        self._dir: DirReader | None = None

    def _backend(self):
        if self._zip is None and self._dir is None:
            try:
                url = (self._url_for or blob_url)(self.repo, self.artifact["id"])
                self._zip = zipfile.ZipFile(RangeFile(url))
            except (OSError, zipfile.BadZipFile) as err:
                self._dir = DirReader(self._download_full(err))
        return self._zip if self._zip is not None else self._dir

    def _download_full(self, reason: Exception) -> Path:
        size_mb = self.artifact["size_in_bytes"] / 1e6
        if size_mb > self.max_mb:
            raise TooLarge(
                f"{self.artifact['name']}: Range read failed ({reason}); the full zip is "
                f"{size_mb:.0f} MB > --max-full-zip-mb {self.max_mb:g}"
            )
        dest = self.cache / "full"
        if not dest.is_dir():
            _gh("run", "download", str(self.artifact["workflow_run"]["id"]), "-R", self.repo,
                "-n", self.artifact["name"], "-D", str(dest))
            FETCHED["bytes"] += self.artifact["size_in_bytes"]
        return dest

    def _listing(self) -> dict[str, int]:
        """member name -> compressed size, cached in names.json (read from the zip directory, no payload)."""
        listing = self.cache / "names.json"
        if listing.is_file():
            try:
                doc = json.loads(listing.read_text())
            except json.JSONDecodeError:
                doc = None
            if isinstance(doc, dict):
                return doc  # an older list-of-names cache has no sizes: fall through and rebuild it
        backend = self._backend()
        if isinstance(backend, zipfile.ZipFile):
            sizes = {i.filename: i.compress_size for i in backend.infolist() if not i.is_dir()}
        else:
            sizes = {n: backend.size(n) for n in backend.names()}
        listing.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(listing, json.dumps(sizes))
        return sizes

    def names(self) -> list[str]:
        return list(self._listing())

    def size(self, rel: str) -> int | None:
        """Compressed bytes the member costs to download; None if absent or unknown."""
        return self._listing().get(rel)

    def read(self, rel: str) -> bytes | None:
        if Path(rel).is_absolute() or ".." in Path(rel).parts:  # member names come from a remote zip
            return None
        cached = self.cache / "files" / rel
        if cached.is_file():
            return cached.read_bytes()
        if rel not in self.names():
            return None
        size = self.size(rel)
        progress(f"fetching {self.artifact['name']}/{rel}" + (f" ({size / 1e6:.1f} MB)" if size else "") + " ...")
        try:
            data = self._backend().read(rel)
        except OSError:  # the short-lived blob URL expired: re-resolve once and retry
            self.close()
            try:
                data = self._backend().read(rel)
            except OSError as err:  # Range really does not work: whole zip, if within budget
                self.close()
                self._dir = DirReader(self._download_full(err))
                data = self._dir.read(rel)
        cached.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(cached, data)
        return data

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
        self._zip = self._dir = None


def job_tail(name: str, limit: int = 60) -> str:
    """Readable end of a (GitHub-truncated) job name: the part after the last ' / ', whitespace collapsed."""
    tail = " ".join(name.rsplit(" / ", 1)[-1].split())
    return tail if len(tail) <= limit else "…" + tail[-limit:]


def remote_configs(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    notes: list[str] = []
    progress(f"listing jobs and artifacts of {src.link} ...")
    jobs = _gh_pages(f"repos/{src.repo}/actions/runs/{src.run_id}/jobs", "jobs")
    for job in jobs:
        if job.get("conclusion") in FAILED_CONCLUSIONS:
            FAILED_JOBS.append({"id": job.get("id"), "name": job["name"], "conclusion": job["conclusion"],
                                "url": job.get("html_url")})
            ref = job.get("html_url") or (f"job {job['id']}" if job.get("id") else "")
            notes.append(f"{src.link}: job failed: {job_tail(job['name'])} ({job['conclusion']})"
                         + (f" {ref}" if ref else ""))
    parts: dict[str, dict[str, dict]] = {}
    for art in _gh_pages(f"repos/{src.repo}/actions/runs/{src.run_id}/artifacts", "artifacts"):
        for prefix, kind in ((AGG_PREFIX, "agg"), (RAW_PREFIX, "raw")):
            if not art["name"].startswith(prefix):
                continue
            if art.get("expired"):
                notes.append(f"{src.link}: artifact {art['name']} expired, skipped")
            else:
                parts.setdefault(art["name"][len(prefix):], {})[kind] = art
    configs: list[Config] = []
    for rf, found in sorted(parts.items()):
        raw = ZipReader(src.repo, found["raw"], cache_dir, max_full_zip_mb) if "raw" in found else None
        aggs: list[dict | None] = [None]
        if "agg" in found:
            progress(f"reading aggregate of {rf} ...")
            reader = ZipReader(src.repo, found["agg"], cache_dir, max_full_zip_mb)
            aggs = [json.loads(reader.read(n)) for n in reader.names() if n.endswith(".json")] or [None]
        configs += [Config(rf=rf, source=src, agg=agg, files=raw) for agg in aggs]
    if not configs:
        notes.append(f"{src.link}: no bmk_agentic_* / agentic_* artifacts found")
    return configs, notes


def discover(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    if src.kind == "run":
        return remote_configs(src, cache_dir, max_full_zip_mb)
    notes: list[str] = []
    configs = local_configs(src, notes)
    if not configs:
        notes.append(f"{src.link}: no aggregate JSON or aiperf export found")
    return configs, notes
