"""Exercise cluster-profile rendering through the launchers' shared entrypoint."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.cluster_config import render_cluster_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("power", ["0", "1", "missing-exporter"])
def test_launcher_writes_job_local_cluster_config(tmp_path: Path, power: str) -> None:
    runner_dir = tmp_path / "runners"
    profiles = runner_dir / "srt-slurm"
    profiles.mkdir(parents=True)
    (profiles / "fixture.yaml").write_text(
        """default_account: ${SLURM_ACCOUNT}
default_partition: ${SLURM_PARTITION}
srtctl_root: ${SRTCTL_ROOT}
gpus_per_node: 4
use_segment_sbatch_directive: false
containers:
  '${IMAGE}': ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
default_mounts:
  /cache: /old-cache
model_paths:
  retained: /models/retained
default_sbatch_directives:
  exclude: ['${EXCLUDED_NODE}']
"""
    )
    output = tmp_path / "job config.yaml"
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT),
        "SLURM_ACCOUNT": "benchmark",
        "SLURM_PARTITION": "batch",
        "SRTCTL_ROOT": '/shared/job "quoted": 1',
        "SQUASH_FILE": '/images/engine "quoted".sqsh',
        "NGINX_SQUASH_FILE": "/images/nginx.sqsh",
        "IMAGE": "registry/engine:tag",
    }
    env.pop("DCGM_EXPORTER_SQSH", None)
    if power == "1":
        env["DCGM_EXPORTER_SQSH"] = "/images/exporter.sqsh"
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; INFERENCEX_SLURM_UTILS_DIR="$2"; '
            'write_srt_cluster_config fixture "$3" "$4" '
            '--var EXCLUDED_NODE gpu-b --model selected "/models/price$" '
            '--mount /cache /new-cache --container prefill /images/prefill.sqsh',
            "bash", str(ROOT / "runners/slurm_utils.sh"), str(runner_dir), str(output),
            "1" if power == "missing-exporter" else power,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if power == "missing-exporter":
        assert result.returncode != 0
        assert "DCGM_EXPORTER_SQSH" in result.stdout
        assert not output.exists()
        return

    assert result.returncode == 0, result.stderr
    rendered = yaml.safe_load(output.read_text())
    assert rendered == {
        "default_account": "benchmark",
        "default_partition": "batch",
        "srtctl_root": '/shared/job "quoted": 1',
        "gpus_per_node": 4,
        "use_segment_sbatch_directive": False,
        "containers": {
            "registry/engine:tag": '/images/engine "quoted".sqsh',
            "nginx-sqsh": "/images/nginx.sqsh",
            "prefill": "/images/prefill.sqsh",
            **({"dcgm-exporter": "/images/exporter.sqsh"} if power == "1" else {}),
        },
        "default_mounts": {"/cache": "/new-cache"},
        "model_paths": {"retained": "/models/retained", "selected": "/models/price$"},
        "default_sbatch_directives": {"exclude": ["gpu-b"]},
    }


@pytest.mark.parametrize(
    ("profile", "error"),
    [
        ("srtctl_root: ${MISSING_ROOT}\n", "MISSING_ROOT"),
        ("- not-a-mapping\n", "Cluster profile must be a YAML mapping"),
    ],
)
def test_invalid_profile_does_not_overwrite_config(tmp_path: Path, profile: str, error: str) -> None:
    source = tmp_path / "profile.yaml"
    source.write_text(profile)
    output = tmp_path / "srtslurm.yaml"
    output.write_text("default_partition: existing\n")
    result = subprocess.run(
        [sys.executable, "-m", "infx.srt_slurm.cluster_config", str(source), str(output)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert error in result.stderr
    assert output.read_text() == "default_partition: existing\n"


def test_runtime_mounts_create_a_mapping_without_profile_mounts() -> None:
    config = render_cluster_config(
        {"use_exclusive_sbatch_directive": True},
        {},
        {"default_mounts": [["/job", "/infmax-workspace"], ["/weights", "/weights"]]},
    )
    assert config == {
        "use_exclusive_sbatch_directive": True,
        "default_mounts": {"/job": "/infmax-workspace", "/weights": "/weights"},
    }
