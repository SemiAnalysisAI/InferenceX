"""Which AIPerf public-dataset loader an AgentX point replays, and its Hugging Face corpus."""

from __future__ import annotations

from infx.bench.env import InputError

# Each loader's corpus is the ``hf_dataset_name`` AIPerf registers for it in
# utils/aiperf/src/aiperf/plugin/plugins.yaml; the run pre-downloads exactly that dataset.
LOADERS = {
    "semianalysis_cc_traces_weka_062126": "semianalysisai/cc-traces-weka-062126",
    "semianalysis_cc_traces_weka_062126_256k": "semianalysisai/cc-traces-weka-062126-256k",
    # Rolling aliases; AIPerf currently points them at the 062126 corpora.
    "semianalysis_cc_traces_weka_with_subagents": "semianalysisai/cc-traces-weka-062126",
    "semianalysis_cc_traces_weka_with_subagents_256k": "semianalysisai/cc-traces-weka-062126-256k",
}
UNCAPPED = "semianalysis_cc_traces_weka_062126"
CAPPED = "semianalysis_cc_traces_weka_062126_256k"
# Families that serve 1M context replay the unfiltered corpus; others take the 256k cap.
# Matching is by prefix, so ``dsv4`` also selects ``dsv41flash``.
UNCAPPED_FAMILIES = ("dsv4", "glm5.2", "glm5.3", "minimaxm3", "kimik3")


def resolve(model_prefix: str, override: str | None) -> tuple[str, str]:
    """``(loader, hf_dataset)``; ``override`` (``WEKA_LOADER_OVERRIDE``) pins another corpus."""
    loader = override or (UNCAPPED if model_prefix.startswith(UNCAPPED_FAMILIES) else CAPPED)
    if loader not in LOADERS:
        raise InputError(f"unknown WEKA_LOADER_OVERRIDE={loader!r}; allowed: {', '.join(LOADERS)}")
    return loader, LOADERS[loader]
