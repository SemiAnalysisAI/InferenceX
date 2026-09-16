import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = REPO_ROOT / "runners" / "nscale_sbatch_adapter.sh"


@pytest.mark.parametrize(
    ("gpu_directive", "expected_arg"),
    [
        ("#SBATCH --gpus-per-node=8", "--gpus-per-node=8"),
        ("#SBATCH --gres=gpu:8", "--gres=gpu:8"),
    ],
)
def test_adapter_forwards_rendered_gpu_directive(tmp_path, gpu_directive, expected_arg):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_salloc = fake_bin / "salloc"
    fake_salloc.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >&2\nexit 42\n")
    fake_salloc.chmod(0o755)

    job_script = tmp_path / "job.slurm"
    job_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --ntasks-per-node=1",
                "#SBATCH --job-name=adapter-test",
                f"#SBATCH --output={tmp_path}/%j.log",
                "#SBATCH --time=00:10:00",
                "#SBATCH --account=benchmark",
                "#SBATCH --partition=batch_1",
                gpu_directive,
            ]
        )
        + "\n"
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["NSCALE_SALLOC_STATE_DIR"] = str(tmp_path / "state")
    result = subprocess.run(
        [str(ADAPTER), str(job_script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 42
    assert expected_arg in result.stderr
