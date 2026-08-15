#!/usr/bin/env python3
"""Make AITER a16w16 split-K selection safe for vLLM FULL graph replay.

This reproduces the two-file local AITER delta present in the measured
unified-v2 runtime. The get_block_n_fp8 change is applied by PR #2585's
apply_dspark_fp8asm.sh; this script applies only the remaining C++ split-K
guard.
"""

import os
from pathlib import Path


LOCAL_AITER = Path(os.environ.get("LOCAL_AITER", "/opt/aiter-local"))
path = LOCAL_AITER / "csrc/py_itfs_cu/asm_gemm_a16w16.cu"
text = path.read_text()

include_old = '#include <cmath>\n#include <memory>\n'
include_new = '#include <cmath>\n#include <cstdlib>\n#include <memory>\n'

setup_old = '''    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
'''
setup_new = '''    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    // PATCH(splitk-cudagraph): the split-K a16w16 ASM kernels ("*_splitk_clean",
    // cfg.splitK==1) reduce partial-K results through a per-(device,stream) atomic
    // semaphore whose "last workgroup does the reduction" protocol relies on the
    // counter being zero at launch. Under vLLM CUDA-graph replay + multi-stream
    // DSpark drafting that invariant is violated (stale counters are never reset
    // on replay / counts mix across streams) and the reduction phase never fires
    // -> all waves spin forever (GPU 100%, ~310W) at seqs=64 warmup. Eager single
    // launches are fine, which is why this only bites in-serve. Disable auto
    // split-K by default; AITER_ALLOW_SPLITK=1 re-enables it for eager-only runs.
    static const bool kAllowSplitK = [] {
        const char* e = getenv("AITER_ALLOW_SPLITK");
        return e && atoi(e) != 0;
    }();

    // PATCH(splitk-cudagraph-csv): a tuned CSV can pin a *_splitk_clean kernel
    // (cfg.splitK==1). Those REQUIRE the split-K semaphore path and crash with a
    // null-pointer memory-access-fault if forced to split=1. When split-K is
    // disabled, drop such a pin so the heuristic below falls back to a non-split
    // (cfg.splitK==0) kernel. Non-clean tuned pins are left untouched.
    if(kernelName && !kAllowSplitK)
    {
        for(const auto& el : *cfgs)
        {
            if(el.first == (arch_id + kernelName))
            {
                if(el.second.splitK == 1)
                    kernelName = nullptr;
                break;
            }
        }
    }
    for(const auto& el : *cfgs)
'''

candidate_old = '''        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1))
'''
candidate_new = '''        // PATCH(splitk-cudagraph): when split-K is disabled, exclude *_splitk_clean
        // kernels (cfg.splitK==1) from selection -- they cannot run without the
        // semaphore-reduction path. A non-split (cfg.splitK==0) kernel with matching
        // tileN/bPreshuffle always exists (pf3/bshuffle variants).
        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1) && (kAllowSplitK || cfg.splitK == 0))
'''

split_old = '''            if(cfg.splitK == 1 && K / cfg.subK >= 2) // kernel and Kdim support splitk
'''
split_new = '''            if(kAllowSplitK && cfg.splitK == 1 && K / cfg.subK >= 2 &&
               pure_tg_num <= 1024) // PATCH(splitk-grid-guard): semaphore workspace is 16*64=1024 tiles, so never SplitK a larger grid (deadlock) — and SplitK is useless once the grid saturates all CUs
'''

if "PATCH(splitk-cudagraph)" not in text:
    anchors = {
        "include": (include_old, 1),
        "setup": (setup_old, 1),
        "candidate": (candidate_old, 1),
        "split": (split_old, 1),
    }
    missing = [name for name, (anchor, count) in anchors.items() if text.count(anchor) != count]
    if missing:
        raise SystemExit(f"ERROR: AITER split-K anchors missing or duplicated: {missing}")
    text = text.replace(include_old, include_new, 1)
    text = text.replace(setup_old, setup_new, 1)
    text = text.replace(candidate_old, candidate_new, 1)
    text = text.replace(split_old, split_new, 1)
    path.write_text(text)

required = (
    "PATCH(splitk-cudagraph)",
    "PATCH(splitk-cudagraph-csv)",
    "PATCH(splitk-grid-guard)",
    "AITER_ALLOW_SPLITK",
)
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(f"ERROR: AITER split-K markers missing: {missing}")
print(f"AITER split-K graph-safety patch OK: {path}")
