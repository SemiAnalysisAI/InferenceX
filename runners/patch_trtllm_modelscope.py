#!/usr/bin/env python3
"""Backport ModelScope loading to the pinned TensorRT-LLM runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HF_IMPORT = "from huggingface_hub import snapshot_download\n"
ALIASED_HF_IMPORT = (
    "from huggingface_hub import snapshot_download as hf_snapshot_download\n"
)

OLD_DOWNLOAD_BLOCK = '''def download_hf_model(model: str, revision: Optional[str] = None) -> Path:
    ignore_patterns = ["original/**/*"]
    logger.info(f"Downloading model {model} from HuggingFace")
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_patterns=ignore_patterns,
            revision=revision,
            tqdm_class=DisabledTqdm)
    logger.info(f"Finished downloading model {model} from HuggingFace")
    return Path(hf_folder)


def download_hf_partial(model: str,
                        allow_patterns: List[str],
                        revision: Optional[str] = None) -> Path:
    """Download a partial model from HuggingFace.

    Args:
        model: The model name or path.
        revision: The revision to use for the model.
        allow_patterns: The patterns to allow for the model.

    Returns:
        The path to the downloaded model.
    """
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            revision=revision,
            allow_patterns=allow_patterns,
            tqdm_class=DisabledTqdm)
    return Path(hf_folder)


'''

NEW_DOWNLOAD_BLOCK = '''def download_hf_model(model: str, revision: Optional[str] = None) -> Path:
    ignore_patterns = ["original/**/*"]
    hub_name = "ModelScope" if use_modelscope() else "Hugging Face"
    logger.info(f"Downloading model {model} from {hub_name}")
    with get_file_lock(model):
        model_folder = _snapshot_download(model,
                                          ignore_patterns=ignore_patterns,
                                          revision=revision)
    logger.info(f"Finished downloading model {model} from {hub_name}")
    return Path(model_folder)


def download_hf_partial(model: str,
                        allow_patterns: List[str],
                        revision: Optional[str] = None) -> Path:
    """Download selected model files from the configured model hub.

    Args:
        model: The model name or path.
        revision: The revision to use for the model.
        allow_patterns: The patterns to allow for the model.

    Returns:
        The path to the downloaded model.
    """
    with get_file_lock(model):
        model_folder = _snapshot_download(model,
                                          revision=revision,
                                          allow_patterns=allow_patterns)
    return Path(model_folder)


def use_modelscope() -> bool:
    """Return whether remote model IDs should resolve through ModelScope."""
    return os.environ.get("TRTLLM_USE_MODELSCOPE", "false").strip().lower(
    ) in ("1", "true")


def _snapshot_download(model: str,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       allow_patterns: Optional[List[str]] = None) -> str:
    """Download a snapshot from ModelScope or Hugging Face.

    ModelScope uses different names for its file filters. Keep the optional
    import in this boundary so standard TensorRT-LLM installations do not need
    the ``modelscope`` package.
    """
    local_files_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if use_modelscope():
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as error:
            raise ImportError(
                "TRTLLM_USE_MODELSCOPE is enabled, but ModelScope is not "
                "installed. Install it with `pip install modelscope`.") from error

        kwargs = {
            "model_id": model,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        if ignore_patterns:
            kwargs["ignore_file_pattern"] = ignore_patterns
        if allow_patterns:
            kwargs["allow_file_pattern"] = allow_patterns
        return snapshot_download(**kwargs)

    return hf_snapshot_download(
        model,
        local_files_only=local_files_only,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        revision=revision,
        tqdm_class=DisabledTqdm)


'''


def installed_package_root() -> Path:
    """Return the installed TensorRT-LLM package root."""
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("tensorrt_llm package is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    """Replace one pinned source fragment or fail on an unknown runtime."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"unsupported TensorRT-LLM source at {path}: "
            f"expected one pinned fragment, found {count}"
        )
    return source.replace(old, new, 1)


def patch_utils(path: Path) -> bool:
    """Patch the hub download boundary."""
    source = path.read_text(encoding="utf-8")
    if "def use_modelscope() -> bool:" in source:
        if ALIASED_HF_IMPORT not in source:
            raise RuntimeError(f"partial ModelScope patch found at {path}")
        return False

    source = replace_once(source, HF_IMPORT, ALIASED_HF_IMPORT, path)
    source = replace_once(source, OLD_DOWNLOAD_BLOCK, NEW_DOWNLOAD_BLOCK, path)
    path.write_text(source, encoding="utf-8")
    return True


def patch_llm(path: Path) -> bool:
    """Load tokenizer and configuration from the downloaded snapshot."""
    source = path.read_text(encoding="utf-8")
    tokenizer_marker = "        model_path = self._hf_model_dir or self.args.model\n"
    generation_marker = "        model_dir = self._hf_model_dir or self.args.model\n"
    if tokenizer_marker in source:
        if source.count(generation_marker) < 3:
            raise RuntimeError(f"partial ModelScope path patch found at {path}")
        return False

    tokenizer_start = source.index("    def _try_load_tokenizer(")
    tokenizer_end = source.index("\n    @property\n    def tokenizer", tokenizer_start)
    tokenizer_source = source[tokenizer_start:tokenizer_end]
    anchor = (
        "        if self.args.tokenizer is not None:\n"
        "            assert isinstance(self.args.tokenizer, TokenizerBase)\n"
        "            return self.args.tokenizer\n\n"
    )
    if tokenizer_source.count("self.args.model") != 4:
        raise RuntimeError(
            f"unsupported TensorRT-LLM tokenizer loader at {path}: "
            "unexpected model-path reference count"
        )
    tokenizer_source = tokenizer_source.replace("self.args.model", "model_path")
    tokenizer_source = replace_once(
        tokenizer_source, anchor, anchor + tokenizer_marker + "\n", path
    )
    source = source[:tokenizer_start] + tokenizer_source + source[tokenizer_end:]

    source = replace_once(
        source,
        "        return ModelLoader.load_hf_generation_config(self.args.model)\n",
        generation_marker
        + "        return ModelLoader.load_hf_generation_config(model_dir)\n",
        path,
    )
    source = replace_once(
        source,
        "        return ModelLoader.load_hf_model_config(\n"
        "            self.args.model, trust_remote_code=self.args.trust_remote_code)\n",
        generation_marker + "        return ModelLoader.load_hf_model_config(\n"
        "            model_dir, trust_remote_code=self.args.trust_remote_code)\n",
        path,
    )
    path.write_text(source, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [TENSORRT_LLM_PACKAGE_ROOT]", file=sys.stderr)
        return 2

    try:
        package_root = (
            Path(argv[1]).resolve() if len(argv) == 2 else installed_package_root()
        )
        changed = [
            patch_utils(package_root / "llmapi/utils.py"),
            patch_llm(package_root / "llmapi/llm.py"),
        ]
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"ERROR: failed to patch TensorRT-LLM ModelScope support: {error}",
            file=sys.stderr,
        )
        return 1

    state = "Patched" if any(changed) else "Already patched"
    print(f"{state} TensorRT-LLM ModelScope support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
