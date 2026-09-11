import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mooncake_backport", Path(__file__).with_name("patch_vllm_mooncake_block_state.py")
)
assert spec and spec.loader
backport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backport)


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/a.py").write_text("value = 1\n")
    (tmp_path / "vllm/b.py").write_text("name = 'old'\n")
    patch = tmp_path / "change.patch"
    patch.write_text("""diff --git a/vllm/a.py b/vllm/a.py
--- a/vllm/a.py
+++ b/vllm/a.py
@@ -1 +1 @@
-value = 1
+value = 2
diff --git a/vllm/b.py b/vllm/b.py
--- a/vllm/b.py
+++ b/vllm/b.py
@@ -1 +1 @@
-name = 'old'
+name = 'new'
""")
    monkeypatch.setattr(backport, "PATCH", patch)
    monkeypatch.setenv("PATH", "")
    return tmp_path


def test_patch_works_without_git_and_is_idempotent(source_tree: Path) -> None:
    assert backport.apply_patch(source_tree)
    assert (source_tree / "vllm/a.py").read_text() == "value = 2\n"
    assert (source_tree / "vllm/b.py").read_text() == "name = 'new'\n"
    assert not backport.apply_patch(source_tree)


def test_unsupported_later_file_leaves_all_files_unchanged(source_tree: Path) -> None:
    (source_tree / "vllm/b.py").write_text("unrecognized = True\n")
    with pytest.raises(RuntimeError, match="refusing partial backport"):
        backport.apply_patch(source_tree)
    assert (source_tree / "vllm/a.py").read_text() == "value = 1\n"
    assert (source_tree / "vllm/b.py").read_text() == "unrecognized = True\n"


def test_invalid_python_leaves_all_files_unchanged(source_tree: Path) -> None:
    patch = backport.PATCH
    patch.write_text(patch.read_text().replace("+name = 'new'", "+name = ("))
    with pytest.raises(SyntaxError):
        backport.apply_patch(source_tree)
    assert (source_tree / "vllm/a.py").read_text() == "value = 1\n"
    assert (source_tree / "vllm/b.py").read_text() == "name = 'old'\n"
