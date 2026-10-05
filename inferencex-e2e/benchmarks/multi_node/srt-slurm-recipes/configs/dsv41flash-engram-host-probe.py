"""Debug probe: can this node's GPUs read hipHostRegister'd host memory?

Mirrors SGLang's engram host table (anonymous mmap, MADV_HUGEPAGE, prefault,
cudaHostRegister) and reads it from a Triton kernel through a raw int64
address, once with hipHostGetDevicePointer's address and once with the host
address. Each (size, flags) case runs in its own process so a GPU fault only
ends that case.
"""

import ctypes
import glob
import mmap
import os
import resource
import subprocess
import sys

GIB = 1 << 30
PAGE = 4096
SAMPLES = 1 << 16


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        return f"<{e.__class__.__name__}>"


def node_info():
    print(f"uname: {' '.join(os.uname())}")
    print(f"cmdline: {read_text('/proc/cmdline')}")
    print(f"amdgpu module version: {read_text('/sys/module/amdgpu/version')}")
    print(f"thp enabled: {read_text('/sys/kernel/mm/transparent_hugepage/enabled')}")
    print(f"iommu devices: {sorted(os.listdir('/sys/class/iommu')) if os.path.isdir('/sys/class/iommu') else '<none>'}")
    print(f"iommu groups: {len(glob.glob('/sys/kernel/iommu_groups/*'))}")
    print(f"memlock rlimit: {resource.getrlimit(resource.RLIMIT_MEMLOCK)}")
    print(f"cpus: {len(os.sched_getaffinity(0))}")
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        print(f"{key}={os.environ.get(key)}")
    for line in read_text("/proc/meminfo").splitlines():
        if line.split(":")[0] in ("MemFree", "HugePages_Total", "HugePages_Surp", "AnonHugePages"):
            print(f"meminfo {line}")
    for node in sorted(glob.glob("/sys/devices/system/node/node*/meminfo")):
        free = [l for l in read_text(node).splitlines() if "MemFree" in l]
        print(f"{free[0].split(':')[0]}: {free[0].split(':')[1].strip()}" if free else node)
    sys.stdout.flush()


def run_case(nbytes, flags):
    import numpy as np
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def gather(base_addr, off_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        base = base_addr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        off = tl.load(off_ptr + i, mask=mask, other=0)
        tl.store(out_ptr + i, tl.load(base + off, mask=mask, other=0), mask=mask)

    tag = f"[{nbytes // GIB} GiB flags={flags}]"
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    mm.madvise(mmap.MADV_HUGEPAGE)
    pages = nbytes // PAGE
    np.frombuffer(mm, dtype=np.uint8)[::PAGE] = (np.arange(pages, dtype=np.int64) % 251).astype(np.uint8)
    host = torch.frombuffer(mm, dtype=torch.uint8)
    host_ptr = host.data_ptr()
    err = int(torch.cuda.cudart().cudaHostRegister(host_ptr, nbytes, flags))
    print(f"{tag} cudaHostRegister rc={err}", flush=True)
    if err:
        return
    hip = ctypes.CDLL("libamdhip64.so")
    dev = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(host_ptr), ctypes.c_uint(0))
    dev_ptr = dev.value or 0
    print(f"{tag} hipHostGetDevicePointer rc={rc} host={host_ptr:#x} dev={dev_ptr:#x} "
          f"same={dev_ptr == host_ptr}", flush=True)

    page_ids = np.linspace(0, pages - 1, SAMPLES).astype(np.int64)
    expected = torch.from_numpy((page_ids % 251).astype(np.uint8))
    offsets = page_ids * PAGE
    targets = [("device_ptr", dev_ptr)] if rc == 0 and dev_ptr else []
    if dev_ptr != host_ptr:
        targets.append(("host_ptr", host_ptr))
    for name, addr in targets:
        for d in range(torch.cuda.device_count()):
            print(f"{tag} gpu{d} read via {name} ...", flush=True)
            with torch.cuda.device(d):
                off = torch.from_numpy(offsets).to(f"cuda:{d}")
                out = torch.empty(SAMPLES, dtype=torch.uint8, device=f"cuda:{d}")
                gather[(triton.cdiv(SAMPLES, 1024),)](addr, off, out, SAMPLES, BLOCK=1024)
                torch.cuda.synchronize(d)
                bad = int((out.cpu() != expected).sum())
            print(f"{tag} gpu{d} read via {name}: {'OK' if bad == 0 else f'WRONG {bad}/{SAMPLES}'}",
                  flush=True)
    torch.cuda.cudart().cudaHostUnregister(host_ptr)


def main():
    if len(sys.argv) == 3:
        run_case(int(sys.argv[1]), int(sys.argv[2]))
        return
    print("=== dsv41flash engram host probe ===")
    node_info()
    for nbytes in (GIB, 48 * GIB):
        for flags in (0, 2):
            try:
                proc = subprocess.run([sys.executable, __file__, str(nbytes), str(flags)], timeout=900)
                print(f"[{nbytes // GIB} GiB flags={flags}] exit={proc.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[{nbytes // GIB} GiB flags={flags}] TIMEOUT", flush=True)
    print("=== probe done ===", flush=True)


if __name__ == "__main__":
    main()
