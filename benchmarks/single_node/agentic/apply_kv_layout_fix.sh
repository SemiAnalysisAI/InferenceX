#!/usr/bin/env bash
# apply_kv_layout_fix.sh
# -----------------------------------------------------------------------------
# Fix "ValueError: Unsupported kv_layout: none. Only NHD and HND are supported."
# for the Kimi-K3 LMCache DRAM-offload arm on vLLM 0.26.x (the nightly-a3561ef8
# image).
#
# Root cause: vLLM 0.26.x removed the module-level get_kv_cache_layout() from
# vllm.v1.attention.backends.utils (the layout is now resolved per-model onto
# CacheConfig.kv_cache_layout). LMCache's LMCacheMPConnector still imports that
# removed symbol, catches the ImportError as "vLLM is not available", and falls
# back to kv_layout="none", which kv_cache_group_edits.apply() rejects.
#
# This patch (idempotent, container-local, backs up each file to .orig_kvlayout):
#   [vLLM]    re-add get_kv_cache_layout() -> legacy NHD/HND name from the
#             current config.
#   [LMCache] make register_kv_caches read the resolved layout off the
#             connector's own vllm_config (reliable; the global config context
#             is not set during KV-cache init) and normalize LBNHC/LBHNC/... to
#             NHD/HND.
#
# Safe to `source` OR `bash`. Set SKIP_KV_LAYOUT_FIX=1 to run stock.
# -----------------------------------------------------------------------------

apply_kv_layout_fix() {
    [ "${SKIP_KV_LAYOUT_FIX:-0}" = "1" ] && { echo "[kvlayout] skipped"; return 0; }

    python3 - <<'PY'
import importlib.util, os, re, py_compile, shutil, sys

MARK = "LMCACHE_KV_LAYOUT_FIX"


def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def backup(path):
    b = path + ".orig_kvlayout"
    if not os.path.exists(b):
        shutil.copy2(path, b)


def write(path, text):
    # Validate syntax BEFORE touching the file so a bad patch can never
    # corrupt the installed module (compile() writes nothing).
    compile(text, path, "exec")
    backup(path)
    with open(path, "w") as f:
        f.write(text)


# ---- 1. vLLM: restore get_kv_cache_layout() -------------------------------
vllm = pkg_dir("vllm")
if vllm:
    p = os.path.join(vllm, "v1/attention/backends/utils.py")
    if os.path.exists(p):
        src = open(p).read()
        if "def get_kv_cache_layout" not in src:
            src += (
                f"\n\n# {MARK}\n"
                "def get_kv_cache_layout():\n"
                '    """Legacy accessor: resolved KV cache layout as NHD/HND."""\n'
                "    _m = {'LBNHC': 'NHD', 'BLNHC': 'NHD', 'LBHNC': 'HND',\n"
                "          'BLHNC': 'HND', 'BHLNC': 'HND', 'NHD': 'NHD', 'HND': 'HND'}\n"
                "    from vllm.config import get_current_vllm_config_or_none\n"
                "    cfg = get_current_vllm_config_or_none()\n"
                "    if cfg is None:\n"
                "        return None\n"
                "    name = getattr(cfg.cache_config, 'kv_cache_layout', None)\n"
                "    if name is None:\n"
                "        return None\n"
                "    return _m.get(name)\n"
            )
            write(p, src)
            print(f"[kvlayout] vLLM: added get_kv_cache_layout in {p}")
        else:
            print("[kvlayout] vLLM: get_kv_cache_layout already present; skip")

# ---- 2. LMCache: read layout off the connector config + normalize ---------
lm = pkg_dir("lmcache")
if not lm:
    print("[kvlayout] LMCache not installed; nothing to patch")
    sys.exit(0)

utils = os.path.join(lm, "integration/vllm/utils.py")
src = open(utils).read()
if MARK not in src:
    block = f'''# {MARK}
_VLLM_KV_CACHE_LAYOUT_ALIASES = {{
    "NHD": "NHD", "HND": "HND",
    "LBNHC": "NHD", "BLNHC": "NHD",
    "LBHNC": "HND", "BLHNC": "HND", "BHLNC": "HND",
}}


def _normalize_vllm_kv_cache_layout(layout):
    if layout is None:
        return None
    norm = _VLLM_KV_CACHE_LAYOUT_ALIASES.get(layout)
    if norm is None:
        logger.error(
            "vLLM reported KV cache layout %r with no NHD/HND equivalent", layout
        )
    return norm


def vllm_layout_hints(vllm_config=None):
    hints = {{}}
    kv_layout = try_get_vllm_kv_cache_layout(vllm_config)
    if kv_layout is not None:
        hints["kv_layout"] = kv_layout
    return hints


def try_get_vllm_kv_cache_layout(vllm_config=None):
    if vllm_config is not None:
        norm = _normalize_vllm_kv_cache_layout(
            getattr(vllm_config.cache_config, "kv_cache_layout", None)
        )
        if norm is not None:
            return norm
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        norm = _normalize_vllm_kv_cache_layout(get_kv_cache_layout())
        if norm is not None:
            return norm
    except ImportError:
        pass
    except Exception:
        logger.error("cannot get KV cache layout from vLLM")
        return None
    try:
        from vllm.config import get_current_vllm_config_or_none
        cur = get_current_vllm_config_or_none()
        if cur is not None:
            return _normalize_vllm_kv_cache_layout(
                getattr(cur.cache_config, "kv_cache_layout", None)
            )
    except Exception:
        pass
    logger.error("cannot get KV cache layout from vLLM")
    return None
'''
    # Replace the two original functions (vllm_layout_hints ..
    # try_get_vllm_kv_cache_layout) up to the next top-level def.
    pat = re.compile(
        r"def vllm_layout_hints\(.*?(?=\ndef lmcache_get_or_create_config)",
        re.S,
    )
    if pat.search(src):
        src = pat.sub(lambda _m: block + "\n", src)
        write(utils, src)
        print(f"[kvlayout] LMCache: patched {utils}")
    else:
        print(f"[kvlayout] WARN: anchor not found in {utils}; not patched")
else:
    print("[kvlayout] LMCache utils already patched; skip")

# ---- 3. LMCache: pass the connector's config into vllm_layout_hints -------
conn = os.path.join(lm, "integration/vllm/lmcache_mp_connector.py")
csrc = open(conn).read()
old = "layout_hints = vllm_layout_hints()"
new = 'layout_hints = vllm_layout_hints(getattr(self, "_vllm_config", None))'
if old in csrc:
    csrc = csrc.replace(old, new)
    write(conn, csrc)
    print(f"[kvlayout] LMCache: connector passes _vllm_config in {conn}")
elif new in csrc:
    print("[kvlayout] LMCache connector already patched; skip")
else:
    print(f"[kvlayout] WARN: vllm_layout_hints() call not found in {conn}")

print("[kvlayout] done -- restart the vLLM server for it to take effect.")
PY
}

apply_kv_layout_fix
