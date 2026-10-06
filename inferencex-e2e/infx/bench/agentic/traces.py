"""Which AIPerf public-dataset loader an AgentX point replays, and its Hugging Face corpus."""

from __future__ import annotations

from infx.bench.env import InputError

UNCAPPED = "semianalysis_cc_traces_weka_062126"
CAPPED = "semianalysis_cc_traces_weka_062126_256k"
# Each loader's corpus is the ``hf_dataset_name`` AIPerf registers for it in
# utils/aiperf/src/aiperf/plugin/plugins.yaml; the run pre-downloads exactly that dataset.
LOADERS = {
    UNCAPPED: "semianalysisai/cc-traces-weka-062126",
    CAPPED: "semianalysisai/cc-traces-weka-062126-256k",
}
# Families that serve 1M context replay the unfiltered corpus; others take the 256k cap.
# Matching is by prefix, so ``dsv4`` also selects ``dsv41flash``.
UNCAPPED_FAMILIES = ("dsv4", "glm5.2", "glm5.3", "minimaxm3", "kimik3")


def resolve(model_prefix: str, override: str | None) -> tuple[str, str]:
    """``(loader, hf_dataset)``; ``override`` (``WEKA_LOADER_OVERRIDE``) pins either corpus."""
    loader = override or (UNCAPPED if model_prefix.startswith(UNCAPPED_FAMILIES) else CAPPED)
    if loader not in LOADERS:
        raise InputError(f"unknown WEKA_LOADER_OVERRIDE={loader!r}; allowed: {', '.join(LOADERS)}")
    return loader, LOADERS[loader]
