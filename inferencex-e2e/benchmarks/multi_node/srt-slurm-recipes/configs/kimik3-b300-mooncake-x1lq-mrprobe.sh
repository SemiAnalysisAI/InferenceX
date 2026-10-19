#!/usr/bin/env bash
# X1-LQ startup diagnostic v2 (experiment branch only, not a benchmark): measure what bounds RDMA memory
# registration on the rail the Mooncake recipe selects, then stop before the engine starts. User-space verbs calls
# only (ibv_query_device / ibv_reg_mr / ibv_dereg_mr through ctypes); no driver, firmware or system setting is
# changed. Writes /logs/x1lq_mr_probe.txt.
set -o pipefail
out=/logs/x1lq_mr_probe.txt
{
    echo "== host"; hostname; date -u +%FT%TZ; uname -r
    echo "== limits"; grep -E 'locked memory|address space' /proc/self/limits
    grep -E '^(MemTotal|MemAvailable|HugePages_Free|Hugepagesize):' /proc/meminfo
    for node in /sys/devices/system/node/node*; do echo "$node: $(grep -E 'MemTotal|MemFree' "$node/meminfo" | tr -s ' ' | tr '\n' ' ')"; done
    echo "== rdma devices (sysfs)"
    for d in /sys/class/infiniband/*; do
        n=$(basename "$d")
        echo "$n vendor=$(cat "$d/device/vendor" 2>/dev/null) device=$(cat "$d/device/device" 2>/dev/null) hca_type=$(cat "$d/hca_type" 2>/dev/null) fw=$(cat "$d/fw_ver" 2>/dev/null) board=$(cat "$d/board_id" 2>/dev/null) numa=$(cat "$d/device/numa_node" 2>/dev/null) state=$(cat "$d/ports/1/state" 2>/dev/null) link=$(cat "$d/ports/1/link_layer" 2>/dev/null) rate=$(cat "$d/ports/1/rate" 2>/dev/null) driver=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)")"
    done
    echo "== verbs probe"
    python3 - <<'PY'
import ctypes, ctypes.util, errno, mmap, os, time
lib = ctypes.CDLL(ctypes.util.find_library("ibverbs") or "libibverbs.so.1", use_errno=True)
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.ibv_get_device_name.restype = ctypes.c_char_p
lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
lib.ibv_open_device.restype = ctypes.c_void_p
lib.ibv_open_device.argtypes = [ctypes.c_void_p]
lib.ibv_query_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.ibv_alloc_pd.restype = ctypes.c_void_p
lib.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
lib.ibv_reg_mr.restype = ctypes.c_void_p
lib.ibv_reg_mr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.ibv_dereg_mr.argtypes = [ctypes.c_void_p]
ACCESS = 1 | 2 | 4  # local write, remote write, remote read (as Mooncake registers segments)
GiB, MiB = 1 << 30, 1 << 20
num = ctypes.c_int(0)
devs = lib.ibv_get_device_list(ctypes.byref(num))
names = [lib.ibv_get_device_name(devs[i]).decode() for i in range(num.value)]
attr = ctypes.create_string_buffer(512)
def query(i):
    ctx = lib.ibv_open_device(devs[i])
    if not ctx:
        return None, None
    if lib.ibv_query_device(ctx, attr):
        return ctx, None
    raw = attr.raw
    u64 = lambda o: int.from_bytes(raw[o:o + 8], "little")
    i32 = lambda o: int.from_bytes(raw[o:o + 4], "little", signed=True)
    return ctx, {"fw": raw[:64].split(b"\0")[0].decode(errors="replace"), "max_mr_size": u64(80), "page_size_cap": hex(u64(88)),
                 "vendor_id": hex(int.from_bytes(raw[96:100], "little")), "vendor_part_id": int.from_bytes(raw[100:104], "little"),
                 "max_qp": i32(108), "max_cq": i32(128), "max_mr": i32(136), "max_pd": i32(140)}
target = None
for i, n in enumerate(names):
    ctx, info = query(i)
    print(f"device {n}: {info}", flush=True)
    state = open(f"/sys/class/infiniband/{n}/ports/1/state").read().strip() if os.path.exists(f"/sys/class/infiniband/{n}/ports/1/state") else ""
    numa = open(f"/sys/class/infiniband/{n}/device/numa_node").read().strip() if os.path.exists(f"/sys/class/infiniband/{n}/device/numa_node") else ""
    if target is None and n.startswith("rdmap") and "ACTIVE" in state and numa == "0" and ctx:
        target = (n, ctx)
if target is None:
    print("no active rdmap* rail on NUMA 0 found", flush=True)
    raise SystemExit(0)
name, ctx = target
pd = lib.ibv_alloc_pd(ctx)
print(f"== target rail {name} (first ACTIVE rdmap* on NUMA 0, as the recipe's setup selects); pd={'ok' if pd else 'FAILED'}", flush=True)
def reg(size):
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
    t0 = time.time()
    mr = lib.ibv_reg_mr(pd, addr, size, ACCESS)
    err = ctypes.get_errno()
    return m, mr, err, round(time.time() - t0, 2)
print("-- single MR length test", flush=True)
for gib in (1, 16, 64, 100, 128, 256):
    m, mr, err, dt = reg(gib * GiB)
    print(f"single {gib} GiB: {'OK' if mr else 'FAIL errno=%d (%s)' % (err, errno.errorcode.get(err, '?'))} in {dt} s", flush=True)
    if mr:
        lib.ibv_dereg_mr(mr)
    m.close()
    if not mr:
        break
print("-- cumulative 16 GiB MRs (cap 1 TiB)", flush=True)
held = []
total = 0
while total < 1024 * GiB:
    m, mr, err, dt = reg(16 * GiB)
    if not mr:
        print(f"cumulative: FAIL at MR #{len(held) + 1} after {total // GiB} GiB registered, errno={err} ({errno.errorcode.get(err, '?')})", flush=True)
        m.close()
        break
    held.append((m, mr))
    total += 16 * GiB
else:
    print(f"cumulative: OK up to {total // GiB} GiB in {len(held)} MRs", flush=True)
for m, mr in held:
    lib.ibv_dereg_mr(mr)
    m.close()
print("-- MR count test (1 MiB each, cap 200000)", flush=True)
held = []
while len(held) < 200000:
    m, mr, err, dt = reg(MiB)
    if not mr:
        print(f"count: FAIL at MR #{len(held) + 1}, errno={err} ({errno.errorcode.get(err, '?')})", flush=True)
        m.close()
        break
    held.append((m, mr))
else:
    print(f"count: OK for {len(held)} MRs", flush=True)
for m, mr in held:
    lib.ibv_dereg_mr(mr)
    m.close()
print("verbs probe done", flush=True)
PY
    echo "== selected environment (prefix-filtered; no tokens)"
    env | grep -E '^(MC_|MOONCAKE_|TOTAL_CPU_DRAM|SLURM_JOB_ID=|CUDA_VISIBLE_DEVICES=)' | grep -viE 'token|secret|key' || true
} > "$out" 2>&1
cat "$out"
echo "x1lq diagnostic: stopping before the engine starts (startup diagnostic, not a benchmark result)" >&2
exit 97
