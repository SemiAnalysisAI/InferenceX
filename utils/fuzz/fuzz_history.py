import subprocess
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from infx.matrix.plan import generation_inputs_at_ref
from infx.workflows.prepare_perf_changelog_merge import resolve_conflict_bytes
from infx.workflows.validate_perf_changelog import ChangelogValidationError
from changelog_gate_tests.test_prepare_perf_changelog_merge import block


@given(fresh=st.integers(0, 6), shared=st.integers(0, 6), padding=st.integers(0, 5))
def test_changelog_merge_preserves_main_and_only_adds_remaining_contributions(fresh, shared, padding):
    repo = 'SemiAnalysisAI/InferenceX'
    link = f'https://github.com/{repo}/pull/42'
    base = block('historical', f'https://github.com/{repo}/pull/1').replace(b'\n', b' ' * padding + b'\n')
    shared_keys = [f'shared-{i}' for i in range(shared)]
    fresh_keys = [f'fresh-{i}' for i in range(fresh)]
    pr = base + b'\n' + b'\n'.join(block(key, 'XXX') for key in shared_keys + fresh_keys)
    if not shared_keys and not fresh_keys:
        pr = base
    main = base + b'\n' + b'\n'.join(block(key, f'https://github.com/{repo}/pull/41')
                                     for key in ['main-only', *shared_keys])
    if not fresh_keys:
        with pytest.raises(ChangelogValidationError, match='no .*contribution|no appended entry'):
            resolve_conflict_bytes(base, pr, main, 42, repo)
    else:
        result = resolve_conflict_bytes(base, pr, main, 42, repo)
        assert result == main + b'\n' + b'\n'.join(block(key, link) for key in fresh_keys)


@given(blob=st.binary(max_size=16_384), second=st.binary(max_size=128), missing=st.booleans(), symlink=st.booleans(),
       name=st.sampled_from(['two words.bin', '中文.bin', 'line\nbreak.bin', 'tab\tname.bin', 'carriage\rreturn.bin']))
def test_historical_input_snapshot_preserves_exact_git_blobs(blob, second, missing, symlink, name):
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        files = {"infx/data.bin": blob, "infx/second.bin": second, "infx/__init__.py": b"",
                 f"infx/assets/{name}": second,
                 "utils/matrix_logic/generate_sweep_configs.py": b"print('historical')\n",
                 "configs/amd-master.yaml": b"{}\n", "configs/nvidia-master.yaml": b"{}\n",
                 "configs/runners.yaml": b"labels: {}\n"}
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if symlink:
            (root / "infx/link").symlink_to("data.bin")
            files["infx/link"] = b"data.bin"
        if missing:
            (root / "configs/runners.yaml").unlink()
        (root / ".gitattributes").write_text("infx/data.bin export-ignore\ninfx/second.bin export-subst\n")
        def git(*args):
            return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                                   "-c", "core.autocrlf=false", *args], cwd=root,
                                  check=True, capture_output=True, timeout=10).stdout
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
        (root / "infx/data.bin").write_bytes(b"uncommitted")
        patch.chdir(root)
        if missing:
            with pytest.raises(ValueError, match="missing generation inputs"):
                with generation_inputs_at_ref("HEAD"):
                    pytest.fail("An incomplete revision must not be exposed")
        else:
            with generation_inputs_at_ref("HEAD") as inputs:
                snapshot = Path(inputs.generator_script).parents[2]
                assert {str(path.relative_to(snapshot)): path.read_bytes()
                        for path in snapshot.rglob("*") if path.is_file()} == files
                assert not (snapshot / "infx/link").is_symlink()
            assert not snapshot.exists()
