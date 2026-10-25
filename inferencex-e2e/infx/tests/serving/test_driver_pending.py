from pathlib import Path

from infx.launch.drivers.serving import wait_started


def fake_squeue(tmp_path: Path, monkeypatch, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"
    squeue.write_text(f"#!/bin/bash\n{body}\n")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")


def test_pending_job_says_why_until_its_log_appears(tmp_path, monkeypatch, capsys):
    log = tmp_path / "job.log"
    calls = tmp_path / "calls"
    # Pending twice, then the job starts and writes its log.
    fake_squeue(tmp_path, monkeypatch, f'echo x >> {calls}; [[ $(wc -l < {calls}) -ge 2 ]] && touch {log}; '
                                       'echo "PENDING Resources (null)"')
    wait_started("7", log, poll_s=0, note_s=0)
    out = capsys.readouterr().out
    assert "Slurm job 7 PENDING Resources (null) (0m queued)" in out
    assert out.count("PENDING") == 2


def test_job_gone_from_queue_returns_without_log(tmp_path, monkeypatch, capsys):
    fake_squeue(tmp_path, monkeypatch, "exit 0")
    wait_started("7", tmp_path / "never.log", poll_s=0, note_s=0)
    assert capsys.readouterr().out == ""


def test_local_squash_preflight_fails_fast_with_the_reason(tmp_path, monkeypatch):
    import pytest

    from infx.launch.context import LaunchError
    from infx.launch.drivers.serving import check_local_squash

    image = tmp_path / "engine.sqsh"
    with pytest.raises(LaunchError, match="not a file on this node"):
        check_local_squash("engine", str(image))
    image.write_text("not squashfs")
    image.chmod(0)
    with pytest.raises(LaunchError, match="not readable"):
        check_local_squash("engine", str(image))
    image.chmod(0o644)
    fake_squeue(tmp_path, monkeypatch, "")  # puts tmp_path/bin first on PATH
    unsquashfs = tmp_path / "bin" / "unsquashfs"
    unsquashfs.write_text("#!/bin/bash\nexit 1\n")
    unsquashfs.chmod(0o755)
    with pytest.raises(LaunchError, match="not a valid squashfs"):
        check_local_squash("engine", str(image))
    unsquashfs.write_text("#!/bin/bash\nexit 0\n")
    check_local_squash("engine", str(image))
