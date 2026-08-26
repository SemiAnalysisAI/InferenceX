"""Restore vllm.v1.attention.backends.utils.get_kv_cache_layout for LMCache.

vllm-project/vllm#51718 restructured the KV cache layouts and removed
``get_kv_cache_layout``; the layout now lives on
``cache_config.kv_cache_layout`` and is read via
``cache_config.get_resolved_kv_cache_layout()``. LMCache still calls the old
function (integration/vllm/utils.py::try_get_vllm_kv_cache_layout), swallows
the ImportError, and returns None -- so its layout hint arrives empty and
_MambaUnifiedViewEdit dies with:

    ValueError: Unsupported kv_layout: none. Only NHD and HND are supported.

Confirmed against lmcache 0.5.5.dev24+rocm7.2 (the newest ROCm nightly wheel
as of 2026-08-26); the old import is still there, so this is not fixable by
upgrading LMCache today.

FAIL-CLOSED: only LBNHC/LBHNC are translated, since those are the two layouts
#51718 keeps NHD/HND as aliases for. Any other resolved layout returns None
and lets LMCache raise its own error, because handing it a wrong legacy name
would silently mis-view the KV tensor rather than fail.

Delete this once LMCache reads the new API.
"""

import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/utils.py"

SHIM = '''

# --- BEGIN LMCACHE-KV-LAYOUT-SHIM (vllm#51718 compat) ---
def get_kv_cache_layout():
    """Legacy accessor removed by vllm#51718, still called by LMCache.

    Returns the legacy NHD/HND alias when the resolved layout has one, else
    None so the caller fails loudly instead of mis-viewing the KV tensor.
    """
    try:
        from vllm.config import get_current_vllm_config

        name = get_current_vllm_config().cache_config.kv_cache_layout
    except Exception:
        return None
    if name in ("NHD", "HND"):
        return name
    return {"LBNHC": "NHD", "LBHNC": "HND"}.get(name)
# --- END LMCACHE-KV-LAYOUT-SHIM ---
'''

src = open(TARGET).read()
if "LMCACHE-KV-LAYOUT-SHIM" in src:
    print("[lmcache-layout-shim] already present")
    sys.exit(0)
if "def get_kv_cache_layout" in src:
    print("[lmcache-layout-shim] vLLM already provides get_kv_cache_layout; nothing to do")
    sys.exit(0)
open(TARGET, "a").write(SHIM)
print("[lmcache-layout-shim] appended to", TARGET)
