# PRINCIPLES draft

1. InferenceX is the open platform for comparing ai accelerators to accelerators
2. Equitablenes towards AI accelerators
3. Positive Industry Wide Impact while *"doing as much as needed, as little as possible"*
4. move fast, break things, fix fast


## Anti-Goal
- Due to having limited GPU time and limited upstream maintainer bandwidth, we are unable to support comparing many inference engines.
  - The core focus of InferenceX is to compare between different AI accelerators.
  - We should add another inference engine only when its benefits clearly outweigh the core maintainer time it would take away from supporting an AI accelerator that InferenceX does not yet cover. **If you have an engine that supports Trainium, Ascend, Alibaba T-Head (PPU), Intel GPU, MI455X, SambaNova, Etched Sohu, or Cerebras, etc, we would be happy to collaborate**
  - For those that want to compare inference engines, we recommend forking the repo, hooking up self hosted GPU resources to your fork repo and potentially writing your own blog on it similar to [ParaSail Engine](https://www.parasail.io/blog/minimax-m3-1-flash-day-0)
