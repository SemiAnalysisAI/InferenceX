#!/usr/bin/env python3
# Inject the MoriChannel into the *installed* LMCache and register it in the
# transfer-channel factory. Idempotent. Run inside the engine container at
# startup (before `vllm serve`), e.g.:
#
#   MORI_CHANNEL_SRC=/deploy/mori_channel.py python3 /deploy/patch_lmcache.py
#
# Finds lmcache via importlib, copies mori_channel.py next to nixl_channel.py,
# and adds the "mori" branch to CreateTransferChannel.
import importlib.util as _u
import os
import shutil
import sys


def _find_transfer_channel_dir():
    spec = _u.find_spec("lmcache.v1.transfer_channel")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("lmcache.v1.transfer_channel not importable")
    return spec.submodule_search_locations[0]


def main():
    tdir = _find_transfer_channel_dir()
    src = os.environ.get("MORI_CHANNEL_SRC", "/deploy/mori_channel.py")
    dst = os.path.join(tdir, "mori_channel.py")

    if not os.path.isfile(src):
        print(f"[patch_lmcache] ERROR: MORI_CHANNEL_SRC not found: {src}")
        return 1
    shutil.copyfile(src, dst)
    print(f"[patch_lmcache] copied mori_channel.py -> {dst}")

    init_py = os.path.join(tdir, "__init__.py")
    s = open(init_py).read()

    if "channel_type == \"mori\"" in s:
        print("[patch_lmcache] factory already has mori branch")
        return 0

    # 1) widen the assert
    old_assert = 'assert channel_type in ["nixl", "mock_memory"]'
    new_assert = 'assert channel_type in ["nixl", "mock_memory", "mori"]'
    if old_assert in s:
        s = s.replace(old_assert, new_assert, 1)
    else:
        print("[patch_lmcache] WARN: assert pattern not found (may differ); "
              "trying to insert mori branch anyway")

    # 2) insert the mori branch before the nixl branch
    anchor = '    if channel_type == "nixl":'
    mori_branch = (
        '    if channel_type == "mori":\n'
        '        from lmcache.v1.transfer_channel.mori_channel import MoriChannel\n'
        '        return MoriChannel(\n'
        '            async_mode=async_mode, role=role, buffer_ptr=buffer_ptr,\n'
        '            buffer_size=buffer_size, align_bytes=align_bytes,\n'
        '            tp_rank=tp_rank, peer_init_url=peer_init_url, device=device,\n'
        '            **kwargs,\n'
        '        )\n\n'
    )
    if anchor not in s:
        print("[patch_lmcache] ERROR: nixl branch anchor not found; aborting")
        return 1
    s = s.replace(anchor, mori_branch + anchor, 1)

    open(init_py, "w").write(s)
    print(f"[patch_lmcache] patched factory {init_py} with mori branch")

    # sanity: byte-compile
    import py_compile
    py_compile.compile(dst, doraise=True)
    py_compile.compile(init_py, doraise=True)
    print("[patch_lmcache] OK (byte-compiled clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
