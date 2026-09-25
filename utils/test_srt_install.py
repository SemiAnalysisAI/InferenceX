"""Run the login installer and rendered compute setup without GPUs or Slurm."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(0o755)


def test_installer_resolves_version_off_shared_disk_and_preserves_edits(tmp_path):
    checkout = tmp_path / "shared-checkout"
    checkout.mkdir()
    (checkout / "tracked").write_text("original\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "add", "tracked"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=checkout, check=True,
    )
    subprocess.run(["git", "tag", "v2.7.0"], cwd=checkout, check=True)
    (checkout / "tracked").write_text("patched\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable(binaries / "git", '''
if [[ "$PWD" == "$SHARED_CHECKOUT" && "$1" == describe ]]; then
    echo 'git describe timed out on shared storage' >&2
    exit 124
fi
exec "$REAL_GIT" "$@"
''')
    executable(binaries / "uv", '''
case "$1 $2" in
  'tool run')
    git describe --dirty --tags --long > "$GITHUB_WORKSPACE/describe"
    rc=$?
    (( rc == 0 )) || exit "$rc"
    [[ "$(cat tracked)" == patched ]] || exit 40
    printf '%s' "$PWD" > "$GITHUB_WORKSPACE/version-source"
    echo '2.7.0.post0+dirty'
    ;;
  'venv --quiet')
    mkdir -p "${@: -1}/bin"
    printf '%s\n' '# activated fixture' > "${@: -1}/bin/activate"
    ;;
  'pip install')
    printf '%s\n' "$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SRTCTL" > "$GITHUB_WORKSPACE/installed-version"
    ;;
  *) exit 41 ;;
esac
''')
    env = {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(tmp_path), "SHARED_CHECKOUT": str(checkout),
        "REAL_GIT": shutil.which("git"),
    }
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; install_srt_slurm "$2"', "bash",
         str(ROOT / "runners/slurm_utils.sh"), str(tmp_path / "venv")],
        cwd=checkout, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "installed-version").read_text() == "2.7.0.post0+dirty\n"
    assert (checkout / ".infx-srt-version").read_text() == "2.7.0.post0+dirty\n"
    assert (tmp_path / "srt-slurm-version.txt").read_text() == "2.7.0.post0+dirty\n"
    assert (tmp_path / "describe").read_text().rstrip().endswith("-dirty")
    assert not Path((tmp_path / "version-source").read_text()).exists()
    assert (checkout / "tracked").read_text() == "patched\n"


@pytest.mark.parametrize("failure", ["version", "empty-version", "venv", "install"])
def test_install_failure_stops_before_submission(tmp_path, failure):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable(binaries / "uv", '''
case "$1 $2" in
  'tool run')
    [[ "$FAILURE" != version ]] || exit 42
    [[ "$FAILURE" != empty-version ]] || exit 0
    echo 2.7.0
    ;;
  'venv --quiet')
    [[ "$FAILURE" != venv ]] || exit 43
    mkdir -p "${@: -1}/bin"
    echo '# fixture' > "${@: -1}/bin/activate"
    ;;
  'pip install') exit 44 ;;
  *) exit 45 ;;
esac
''')
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; install_srt_slurm "$2" || exit $?; touch "$3"', "bash",
         str(ROOT / "runners/slurm_utils.sh"), str(tmp_path / "venv"), str(tmp_path / "submitted")],
        cwd=checkout, env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
                           "GITHUB_WORKSPACE": str(tmp_path), "FAILURE": failure},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "submitted").exists()
    assert (checkout / ".infx-srt-version").exists() == (failure in {"venv", "install"})


def compute_script(tmp_path):
    """Apply the shipped patch and render the actual upstream job template."""
    template_root = tmp_path / "src/srtctl/templates"
    template_root.mkdir(parents=True)
    shutil.copy(ROOT / "utils/srt-slurm/src/srtctl/templates/job_script_minimal.j2", template_root)
    shutil.copy(ROOT / "utils/srt-slurm/pyproject.toml", tmp_path)
    subprocess.run(
        ["git", "apply", str(ROOT / "runners/srt-slurm/patches/local-version-compute-setup.patch")],
        cwd=tmp_path, check=True, capture_output=True,
    )
    rendered = Environment(loader=FileSystemLoader(template_root)).get_template("job_script_minimal.j2").render(
        srtctl_source=str(tmp_path), output_base=str(tmp_path / "outputs"),
        config_environment={}, sbatch_directives={}, runtime_config_filename="runtime.yaml",
    )
    script = tmp_path / "job.sh"
    script.write_text(rendered)
    return script


@pytest.mark.parametrize("sync_status", [0, 38])
def test_compute_uses_staged_version_and_propagates_sync_failure(tmp_path, sync_status):
    script = compute_script(tmp_path)
    (tmp_path / ".infx-srt-version").write_text("2.7.0.post1+g123abcd\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable(binaries / "flock", 'shift 2\nexec "$@"\n')
    executable(binaries / "uv", '''
case "$1" in
  --version) echo fixture ;;
  sync)
    echo "$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SRTCTL" > "$SRTCTL_SOURCE_DIR/sync-version"
    exit "$SYNC_STATUS"
    ;;
  run) touch "$SRTCTL_SOURCE_DIR/ran-orchestrator" ;;
  *) exit 46 ;;
esac
''')
    env = {**os.environ, "SLURM_JOB_ID": "42", "SYNC_STATUS": str(sync_status)}
    env.pop("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SRTCTL", None)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == sync_status, result.stdout
    assert (tmp_path / "sync-version").read_text() == "2.7.0.post1+g123abcd\n"
    assert (tmp_path / "ran-orchestrator").exists() == (sync_status == 0)
