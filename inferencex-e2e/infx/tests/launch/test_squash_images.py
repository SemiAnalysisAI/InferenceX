"""Squash image staging of the Slurm backend against fake enroot/unsquashfs/srun/flock binaries."""

import fcntl
import json
import stat
import sys
from pathlib import Path

import pytest

from infx.clusters import load_inventory
from infx.clusters.slurm import SquashCache, SquashPolicy
from infx.launch.backends.base import BackendError, Image, Job
from infx.launch.backends.slurm import SlurmBackend
from infx.launch.backends.slurm import squash as containers
from infx.launch.backends.slurm.squash import (
    ImageError,
    enroot_uri,
    ensure_image,
    squash_path,
)
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest

VALID = "hsqs-fixture"
IMAGE = "lmsysorg/sglang:v0.5.9@sha256:" + "a" * 64
SQUASH_NAME = "lmsysorg_sglang_v0.5.9_sha256_" + "a" * 64 + ".sqsh"
REGISTRY_IMAGE = "registry-1.docker.io#lmsysorg/sglang:sha256:" + "a" * 64
VALID_SHA256 = "b3f1af678620ca28d0ae61cdffe6c9017f923171b40fb7e11fc716d4be8d7468"


def recorder(log: Path) -> str:
    """Bash snippet appending argv and the enroot env as a JSON line to ``log``."""
    return (
        f"{sys.executable} -c 'import json,os,sys; print(json.dumps({{\"argv\": sys.argv[1:], "
        f"\"env\": {{k: v for k, v in os.environ.items() if k.startswith(\"ENROOT_\")}}}}))' "
        f"\"$@\" >> {log}"
    )


def calls(log: Path) -> list[dict]:
    """Recorded invocations."""
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """Fake image tools: unsquashfs accepts files starting with the squash magic."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")
    monkeypatch.setattr(containers, "RETRY_DELAY_S", 0)
    logs = {name: tmp_path / f"{name}.log" for name in ("enroot", "srun", "unsquashfs")}

    def install(name: str, body: str) -> None:
        binary = binaries / name
        binary.write_text(f"#!/bin/bash\n{body}\n")
        binary.chmod(0o755)

    install("unsquashfs", f'{recorder(logs["unsquashfs"])}\n'
            f'[ "$1" = -l ] && [ -r "$2" ] && [ "$(head -c 4 "$2")" = "{VALID[:4]}" ]')
    install("flock", "exit 0")
    install("enroot", f'{recorder(logs["enroot"])}\n[ "$1" = import ] && [ "$2" = -o ]\n'
            f'umask 077\nprintf "{VALID}" > "$3"')
    install("srun", f'{recorder(logs["srun"])}\n'
            'while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"')
    install("hostname", "echo node1")
    return install, logs


def policy(tmp_path, mode, visibility="shared"):
    return SquashPolicy(dir=tmp_path / "squash", import_mode=mode, visibility=visibility, lock_timeout_s=5)


def test_valid_cache_is_reused_without_import(tools, tmp_path):
    _, logs = tools
    squash = tmp_path / "squash" / SQUASH_NAME
    squash.parent.mkdir()
    squash.write_text(VALID)
    assert ensure_image(IMAGE, policy(tmp_path, "submit-host"), job=None) == str(squash)
    assert calls(logs["enroot"]) == []


def test_invalid_cache_is_atomically_reimported(tools, tmp_path):
    _, logs = tools
    squash = tmp_path / "squash" / SQUASH_NAME
    squash.parent.mkdir()
    squash.write_text("truncated")
    stale = squash.with_name(squash.name + ".tmp.deadhost.1")
    stale.write_text("partial")
    foreign = squash.with_name(squash.name + ".tmp.4242")
    foreign.write_text("in progress")

    assert ensure_image(IMAGE, policy(tmp_path, "submit-host"), job=None) == str(squash)

    [call] = calls(logs["enroot"])
    output, uri = call["argv"][2:]
    assert output != str(squash) and Path(output).parent == squash.parent
    assert uri == "docker://registry-1.docker.io#lmsysorg/sglang:sha256:" + "a" * 64
    assert set(call["env"]) >= {"ENROOT_TEMP_PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH",
                                "ENROOT_RUNTIME_PATH"}
    assert squash.read_text() == VALID
    assert squash.stat().st_mode & stat.S_IROTH
    assert sorted(p.name for p in squash.parent.iterdir()) == [SQUASH_NAME, f"{SQUASH_NAME}.lock", foreign.name]
    assert not any(Path(path).exists() for path in call["env"].values())


@pytest.mark.parametrize(("lock_file", "lock_name"), [
    ("beside-squash", f"{SQUASH_NAME}.lock"),
    ("locks-dir", f".locks/{SQUASH_NAME.removesuffix('.sqsh')}.lock"),
])  # fmt: skip
def test_import_waits_while_another_importer_holds_its_lock(tools, tmp_path, lock_file, lock_name):
    _, logs = tools
    lock = tmp_path / "squash" / lock_name
    lock.parent.mkdir(parents=True)
    images = SquashPolicy(dir=tmp_path / "squash", import_mode="submit-host", lock_timeout_s=1, lock_file=lock_file)
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(ImageError, match="timed out"):
            ensure_image(IMAGE, images, job=None)
    assert calls(logs["enroot"]) == []


@pytest.mark.parametrize("mode", ["submit-host", "compute"])
def test_lock_file_this_account_cannot_write_is_opened_read_only(tools, tmp_path, mode):
    lock = tmp_path / "squash" / f"{SQUASH_NAME}.lock"
    lock.parent.mkdir()
    lock.write_text("")
    lock.chmod(0o444)
    squash = ensure_image(IMAGE, policy(tmp_path, mode), job=Job("7") if mode == "compute" else None)
    assert Path(squash).read_text() == VALID
    assert stat.S_IMODE(lock.stat().st_mode) == 0o444


def test_import_producing_invalid_squash_fails_without_replacing(tools, tmp_path):
    install, logs = tools
    install("enroot", f'{recorder(logs["enroot"])}\nprintf garbage > "$3"')
    with pytest.raises(ImageError, match="after 3 attempts"):
        ensure_image(IMAGE, policy(tmp_path, "submit-host"), job=None)
    assert len(calls(logs["enroot"])) == 3
    assert sorted(p.name for p in (tmp_path / "squash").iterdir()) == [f"{SQUASH_NAME}.lock"]


@pytest.mark.parametrize(("mode", "visibility", "job", "message", "steps"), [
    ("pre-staged", "shared", None, rf"pre-staged image .* missing or invalid at .*{SQUASH_NAME}", 0),
    ("pre-staged", "node-local", Job("7"), "stage it there", 1),
    ("compute", "node-local", None, "requires a job allocation", 0),
])  # fmt: skip
def test_an_image_that_cannot_be_staged_fails_at_once(tools, tmp_path, mode, visibility, job, message, steps):
    _, logs = tools
    with pytest.raises(ImageError, match=message):
        ensure_image(IMAGE, policy(tmp_path, mode, visibility), job=job)
    assert len(calls(logs["srun"])) == steps
    assert calls(logs["enroot"]) == []


def test_all_nodes_imports_on_the_jobs_node(tools, tmp_path):
    _, logs = tools
    squash = ensure_image(IMAGE, policy(tmp_path, "all-nodes", "node-local"), job=Job("7"))
    [step] = calls(logs["srun"])
    options = step["argv"][:step["argv"].index("bash")]
    assert {"--jobid=7", "--nodes=1", "--ntasks-per-node=1"} <= set(options)
    assert Path(squash).read_text() == VALID
    [imported] = calls(logs["enroot"])
    assert imported["argv"][2].startswith(squash + ".tmp.node1.")
    assert sorted(p.name for p in Path(squash).parent.iterdir()) == [SQUASH_NAME, f"{SQUASH_NAME}.lock"]


def test_node_import_retries_transient_step_failures(tools, tmp_path):
    install, logs = tools
    counter = tmp_path / "attempts"
    install("srun", f'n=$(cat {counter} 2>/dev/null || echo 0); echo $((n+1)) > {counter}\n'
            '[ "$n" -ge 2 ] || { echo "mkdir: cannot create directory: Protocol family not supported" >&2; exit 1; }\n'
            'while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do shift; done\nexec "$@"')
    squash = ensure_image(IMAGE, policy(tmp_path, "compute"), job=Job("7"))
    assert counter.read_text().strip() == "3"
    assert Path(squash).read_text() == VALID


@pytest.mark.parametrize("image,uri", [
    ("nginx:1.27.4", "docker://nginx:1.27.4"),
    ("nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9", "docker://nvcr.io#nvidia/ai-dynamo/sglang-runtime:0.9"),
    ("nvcr.io#nvidia/k8s/dcgm-exporter:4.6.0", "docker://nvcr.io#nvidia/k8s/dcgm-exporter:4.6.0"),
    ("localhost:5000/team/img:dev", "docker://localhost:5000#team/img:dev"),
    ("nginx@sha256:" + "b" * 64, "docker://registry-1.docker.io#library/nginx:sha256:" + "b" * 64),
    ("nginx:1.27@sha256:" + "b" * 64, "docker://registry-1.docker.io#library/nginx:sha256:" + "b" * 64),
    ("lmsysorg/sglang:v0.5@sha256:" + "c" * 64, "docker://registry-1.docker.io#lmsysorg/sglang:sha256:" + "c" * 64),
    ("nvcr.io/nvidia/sglang:25.01@sha256:" + "d" * 64, "docker://nvcr.io#nvidia/sglang:sha256:" + "d" * 64),
])
def test_enroot_uri_normalization(image, uri):
    assert enroot_uri(image) == uri


def test_squash_locations_override_the_cache_field_by_field():
    squash = SquashCache.model_validate({
        "dir": "/cache", "import": "compute",
        "framework-dirs": {
            "sglang": {"key-style": "plus", "import": "unchecked", "model-prefixes": {
                "dsv4": {"key-style": "plus"}, "glm5.2": {"dir": "/other"},
            }},
            "trt": {"dir": "/trt", "key-style": "plus-strip-nvcr"},
        },
        "helper-dirs": {"nginx": {"import": "unchecked"}},
    })  # fmt: skip
    image = "nvcr.io/nvidia/sglang:0.9"

    def located(policy: SquashPolicy) -> tuple[str, str]:
        return str(squash_path(image, policy)), policy.import_mode

    assert located(squash.policy("sglang", "dsr1")) == ("/cache/nvcr.io+nvidia+sglang+0.9.sqsh", "unchecked")
    assert located(squash.policy("sglang", "dsv4")) == ("/cache/nvcr.io+nvidia+sglang+0.9.sqsh", "compute")
    assert located(squash.policy("sglang", "glm5.2")) == ("/other/nvcr.io_nvidia_sglang_0.9.sqsh", "compute")
    assert located(squash.policy("trt", "dsr1")) == ("/trt/nvidia+sglang+0.9.sqsh", "compute")
    assert located(squash.helper_policy("nginx")) == ("/cache/nvcr.io_nvidia_sglang_0.9.sqsh", "unchecked")
    for policy in (squash.policy(), squash.policy("vllm"), squash.helper_policy("dcgm-exporter")):
        assert located(policy) == ("/cache/nvcr.io_nvidia_sglang_0.9.sqsh", "compute")


def srtctl_backend(tmp_path, **squash) -> SlurmBackend:
    """The Slurm backend of a cluster whose squash cache imports on this host."""
    record = {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "slurm": {
        "partition": "p", "exclusive": True,
        "squash": {"dir": str(tmp_path / "squash"), "import": "submit-host", "lock-timeout-s": 5, **squash},
    }}  # fmt: skip
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    return SlurmBackend(cluster, LaunchRequest.from_env({"RUNNER_NAME": "c_0"}), Lifecycle())


def test_srtctl_multi_node_jobs_import_first_unless_the_cluster_opts_out(tools, tmp_path):
    _, logs = tools
    squash = tmp_path / "squash" / SQUASH_NAME

    reusing = srtctl_backend(tmp_path, **{"multi-node-import": False})
    assert reusing.stage_image(IMAGE).reference == REGISTRY_IMAGE
    assert calls(logs["enroot"]) == []
    importing = srtctl_backend(tmp_path)
    assert importing.stage_image(IMAGE, single_node=True).reference == REGISTRY_IMAGE
    assert importing.stage_image(IMAGE).reference == str(squash)
    assert len(calls(logs["enroot"])) == 1
    assert reusing.stage_image(IMAGE).reference == str(squash)


def test_unchecked_images_are_handed_to_jobs_without_validation_or_import(tools, tmp_path):
    _, logs = tools
    backend = srtctl_backend(tmp_path, **{
        "framework-dirs": {"sglang": {"key-style": "plus", "import": "unchecked", "model-prefixes": {
            "dsv4": {"key-style": "plus"},
        }}},
        "helper-dirs": {"nginx": {"import": "unchecked"}},
    })  # fmt: skip
    squash = tmp_path / "squash" / ("lmsysorg+sglang+v0.5.9+sha256+" + "a" * 64 + ".sqsh")
    squash.parent.mkdir()
    squash.write_text("truncated")

    assert backend.stage_image(IMAGE, framework="sglang", model_prefix="dsr1").reference == str(squash)
    nginx = backend.stage_image("nginx:1.27.4", helper="nginx")
    assert nginx.reference == str(tmp_path / "squash" / "nginx_1.27.4.sqsh")
    assert calls(logs["unsquashfs"]) == calls(logs["enroot"]) == calls(logs["srun"]) == []
    assert squash.read_text() == "truncated"
    assert backend.stage_image(IMAGE, framework="sglang", model_prefix="dsv4").reference == str(squash)
    assert len(calls(logs["enroot"])) == 1
    assert squash.read_text() == VALID


def test_image_provenance_is_the_squash_digest_or_the_registry_reference(tools, tmp_path):
    backend = srtctl_backend(tmp_path)
    staged = backend.stage_image(IMAGE)
    squash = Path(staged.reference)

    assert backend.image_provenance(staged) == f"{VALID_SHA256}  {squash}"
    registry = Image("nginx:1.27.4", "nginx:1.27.4")
    assert backend.image_provenance(registry) == "nginx:1.27.4"
    squash.unlink()
    with pytest.raises(BackendError, match="not readable"):
        backend.image_provenance(staged)
