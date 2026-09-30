import io
import tarfile

import pytest

from research.dsv41.export_profiles import export_profiles


def archive(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))


def test_research_files_become_direct_artifact_members(tmp_path):
    source = tmp_path / "server.tar.gz"
    archive(
        source,
        [
            ("./logs/research/serving/rank0.trace.json.gz", b"trace"),
            ("./logs/research/operators/results.json", b"results"),
            ("./logs/server.log", b"server log"),
        ],
    )
    output = tmp_path / "profiles"
    assert export_profiles([source], output) == 2
    assert (output / "serving/rank0.trace.json.gz").read_bytes() == b"trace"
    assert (output / "operators/results.json").read_bytes() == b"results"
    assert not (output / "server.log").exists()


def test_archive_traversal_is_rejected(tmp_path):
    source = tmp_path / "server.tar.gz"
    archive(source, [("./logs/research/../../outside", b"bad")])
    with pytest.raises(ValueError, match="Unsafe archive"):
        export_profiles([source], tmp_path / "profiles")
    assert not (tmp_path / "outside").exists()
