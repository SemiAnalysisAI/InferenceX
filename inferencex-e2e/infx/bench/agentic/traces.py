"""Which AIPerf public-dataset loader an AgentX point replays, and its Hugging Face corpus."""

from __future__ import annotations

from infx.bench.env import InputError

UNCAPPED = "semianalysis_cc_traces_weka_062126"
CAPPED = "semianalysis_cc_traces_weka_062126_256k"
LOADERS = {
    UNCAPPED: "semianalysisai/cc-traces-weka-062126",
    CAPPED: "semianalysisai/cc-traces-weka-062126-256k",
}
CAPPED_FAMILIES = ("dsr1", "qwen3.5", "qwen3.8next")


def resolve(model_prefix: str, override: str | None) -> tuple[str, str]:
    """``(loader, hf_dataset)``; ``override`` (``WEKA_LOADER_OVERRIDE``) pins either corpus."""
    loader = override or (CAPPED if model_prefix.startswith(CAPPED_FAMILIES) else UNCAPPED)
    if loader not in LOADERS:
        raise InputError(f"unknown WEKA_LOADER_OVERRIDE={loader!r}; allowed: {', '.join(LOADERS)}")
    return loader, LOADERS[loader]
