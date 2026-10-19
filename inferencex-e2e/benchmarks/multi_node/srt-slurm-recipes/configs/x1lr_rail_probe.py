#!/usr/bin/env python3
"""X1-LR B300 Mooncake rail probe (experiment branch only; not a benchmark).

Each phase starts one store client per TP rank (one process each), as vLLM's MooncakeStoreConnector does, with the
recipe's segment sizes on the devices the phase assigns. A client registers a GPU buffer on its own GPU, writes a
payload into the next rank's segment and reads the previous rank's payload back, from host and GPU memory.

Phases:
  efa_per_rank      one EFA device per rank, nearest to the rank's GPU (the per-rank mapping under test)
  ib_one            every rank on the first ACTIVE InfiniBand HCA
  ib_split          ranks split across the ACTIVE InfiniBand HCAs
  tcp               protocol tcp, small segments
  efa_same_device   two clients on one EFA device, 200 GiB each (per-device aggregate control)
  efa_small_per_rank / efa_small_one
                    8 GiB segments (one EFA device per rank / every rank on one EFA device): registration far below
                    the observed EFA limits, so the store round trips test the RDMA data path itself

Results go to --out (JSON) and stdout. Only user-space verbs and store calls; no system setting is changed.
"""
import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time

GiB, MiB = 1 << 30, 1 << 20
PAYLOAD = 64 * MiB
BARRIER_S = 600


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def payload_for(phase, rank, kind):
    return random.Random(f"{phase}/{rank}/{kind}").randbytes(PAYLOAD)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def wait_for(paths, abort_path, timeout=BARRIER_S):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(abort_path):
            return "aborted"
        if all(os.path.exists(p) for p in paths):
            return "ok"
        time.sleep(0.5)
    return "timeout"


def child(a):
    """One store client (one TP rank)."""
    d = os.path.join(a.work, a.phase)
    me = {"phase": a.phase, "rank": a.rank, "world": a.world, "pid": os.getpid(), "device": a.device,
          "protocol": a.protocol, "segment_bytes": a.seg_bytes, "local_buffer_bytes": a.local_bytes,
          "gpu_bytes": a.gpu_bytes, "gpu": a.rank, "steps": []}
    abort = os.path.join(d, "ABORT")

    def step(name, **kw):
        kw["step"], kw["utc"] = name, now()
        me["steps"].append(kw)
        print(f"[{a.phase} r{a.rank}] {name} {json.dumps(kw, sort_keys=True)}", flush=True)
        write_json(os.path.join(d, f"r{a.rank}.result.json"), me)

    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
    store = MooncakeDistributedStore()
    t0 = time.time()
    try:
        ret = store.setup(local_ip(), "P2PHANDSHAKE", a.seg_bytes, a.local_bytes, a.protocol, a.device, a.master)
    except Exception as e:  # noqa: BLE001 - record any binding error verbatim
        ret = f"exception: {e!r}"
    step("setup", ret=ret, seconds=round(time.time() - t0, 2))
    segment = store.get_hostname() if ret == 0 else None
    gbuf = None
    if ret == 0 and a.gpu_bytes:
        t0 = time.time()
        try:
            import torch
            gbuf = torch.empty(a.gpu_bytes, dtype=torch.uint8, device=f"cuda:{a.rank}")
            gret = store.register_buffer(gbuf.data_ptr(), a.gpu_bytes)
        except Exception as e:  # noqa: BLE001
            gret = f"exception: {e!r}"
        step("register_gpu_buffer", ret=gret, seconds=round(time.time() - t0, 2))
        if gret != 0:
            gbuf = None
    write_json(os.path.join(d, f"r{a.rank}.ready.json"), {"segment": segment, "setup_ret": ret, "gpu": gbuf is not None})
    peers = [os.path.join(d, f"r{r}.ready.json") for r in range(a.world)]
    state = wait_for(peers, abort)
    step("barrier_ready", state=state)
    if state != "ok" or ret != 0:
        return finish(store, me, d, a, ret == 0)
    ready = {r: json.load(open(os.path.join(d, f"r{r}.ready.json"))) for r in range(a.world)}
    if any(v["setup_ret"] != 0 for v in ready.values()):
        step("transfers_skipped", reason="a peer's setup failed",
             failed_ranks=sorted(r for r, v in ready.items() if v["setup_ret"] != 0))
        return finish(store, me, d, a, True)
    nxt, prv = (a.rank + 1) % a.world, (a.rank - 1) % a.world
    target = ready[nxt]["segment"]
    cfg = ReplicateConfig()
    cfg.replica_num = 1
    if target:
        cfg.preferred_segment = target

    host = payload_for(a.phase, a.rank, "host")
    t0 = time.time()
    try:
        pret = store.put(f"x1lr/{a.phase}/host/{a.rank}", host, cfg)
    except Exception as e:  # noqa: BLE001
        pret = f"exception: {e!r}"
    step("host_put_to_next_rank_segment", ret=pret, target_rank=nxt, target_segment=target,
         bytes=len(host), seconds=round(time.time() - t0, 3))
    gput = None
    if gbuf is not None:
        import torch
        gbuf[:PAYLOAD].copy_(torch.frombuffer(bytearray(payload_for(a.phase, a.rank, "gpu")), dtype=torch.uint8))
        torch.cuda.synchronize(a.rank)
        t0 = time.time()
        try:
            gput = store.put_from(f"x1lr/{a.phase}/gpu/{a.rank}", gbuf.data_ptr(), PAYLOAD, cfg)
        except Exception as e:  # noqa: BLE001
            gput = f"exception: {e!r}"
        step("gpu_put_from_to_next_rank_segment", ret=gput, target_rank=nxt, seconds=round(time.time() - t0, 3))
    write_json(os.path.join(d, f"r{a.rank}.put.json"), {"host": pret, "gpu": gput})
    state = wait_for([os.path.join(d, f"r{r}.put.json") for r in range(a.world)], abort)
    step("barrier_put", state=state)

    t0 = time.time()
    try:
        got = store.get(f"x1lr/{a.phase}/host/{prv}")
        got = bytes(got) if got is not None else b""
        want = hashlib.sha256(payload_for(a.phase, prv, "host")).hexdigest()
        step("host_get_previous_rank_payload", source_rank=prv, bytes=len(got),
             verified=hashlib.sha256(got).hexdigest() == want, seconds=round(time.time() - t0, 3))
    except Exception as e:  # noqa: BLE001
        step("host_get_previous_rank_payload", source_rank=prv, error=repr(e))
    if gbuf is not None:
        import torch
        dst = gbuf.data_ptr() + 2 * PAYLOAD
        t0 = time.time()
        try:
            gget = store.get_into(f"x1lr/{a.phase}/gpu/{prv}", dst, PAYLOAD)
            torch.cuda.synchronize(a.rank)
            data = gbuf[2 * PAYLOAD:3 * PAYLOAD].cpu().numpy().tobytes()
            want = hashlib.sha256(payload_for(a.phase, prv, "gpu")).hexdigest()
            step("gpu_get_into_previous_rank_payload", source_rank=prv, ret=gget,
                 verified=hashlib.sha256(data).hexdigest() == want, seconds=round(time.time() - t0, 3))
        except Exception as e:  # noqa: BLE001
            step("gpu_get_into_previous_rank_payload", source_rank=prv, error=repr(e))
    if a.bandwidth_bytes:
        blob = (host * (a.bandwidth_bytes // PAYLOAD + 1))[:a.bandwidth_bytes]
        t0 = time.time()
        try:
            bret = store.put(f"x1lr/{a.phase}/bw/{a.rank}", blob, cfg)
            dt = time.time() - t0
            step("host_put_bandwidth", ret=bret, bytes=len(blob), seconds=round(dt, 3),
                 gbytes_per_s=round(len(blob) / dt / 1e9, 3) if dt > 0 else None)
        except Exception as e:  # noqa: BLE001
            step("host_put_bandwidth", error=repr(e))
    return finish(store, me, d, a, True)


def finish(store, me, d, a, ok_setup):
    write_json(os.path.join(d, f"r{a.rank}.done.json"), {"ok_setup": ok_setup})
    wait_for([os.path.join(d, f"r{r}.done.json") for r in range(a.world)], os.path.join(d, "ABORT"), timeout=300)
    try:
        store.close()
        me["closed"] = True
    except Exception as e:  # noqa: BLE001
        me["closed"] = repr(e)
    write_json(os.path.join(d, f"r{a.rank}.result.json"), me)
    return 0


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e!r}"


def rdma_devices(base):
    out = {}
    for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        def rd(rel):
            try:
                return open(os.path.join(base, n, rel)).read().strip()
            except OSError:
                return ""
        drv = os.path.basename(os.path.realpath(os.path.join(base, n, "device", "driver")))
        out[n] = {"state": rd("ports/1/state"), "link_layer": rd("ports/1/link_layer"), "numa": rd("device/numa_node"),
                  "driver": drv, "pci": os.path.basename(os.path.realpath(os.path.join(base, n, "device")))}
    return out


def gpu_pci():
    rows = sh("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader")
    res = {}
    for line in rows.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            bus = parts[1].lower()
            if bus.count(":") == 2 and len(bus.split(":")[0]) == 8:
                bus = bus[4:]
            numa = ""
            try:
                numa = open(f"/sys/bus/pci/devices/{bus}/numa_node").read().strip()
            except OSError:
                pass
            res[int(parts[0])] = {"pci": bus, "numa": numa}
    return res


def efa_per_rank_map(world, devs, topo, gpus):
    efa = [n for n, v in devs.items() if v["driver"] == "efa" and "ACTIVE" in v["state"]]
    used, mapping = set(), {}
    for r in range(world):
        choice, how = None, None
        for cand in (topo.get(f"cuda:{r}") or [[]])[0]:
            if cand in efa and cand not in used:
                choice, how = cand, "mooncake_topology_preferred"
                break
        if choice is None:
            numa = gpus.get(r, {}).get("numa", "")
            for cand in efa:
                if cand not in used and devs[cand]["numa"] == numa:
                    choice, how = cand, "same_numa_unused"
                    break
        if choice is None:
            for cand in efa:
                if cand not in used:
                    choice, how = cand, "any_unused"
                    break
        mapping[r] = {"device": choice, "how": how, "gpu_numa": gpus.get(r, {}).get("numa", ""),
                      "device_numa": devs.get(choice, {}).get("numa", "") if choice else ""}
        if choice:
            used.add(choice)
    return mapping


def run_phase(a, phase, devices, protocol, seg, local, gpu_bytes, bw, world, sequential=False):
    d = os.path.join(a.work, phase)
    os.makedirs(d, exist_ok=True)
    print(f"== phase {phase} ({now()}): protocol={protocol} world={world} seg={seg // GiB} GiB "
          f"local={local // GiB} GiB gpu={gpu_bytes // GiB} GiB devices={devices}", flush=True)
    procs, t0 = [], time.time()
    for r in range(world):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--phase", phase, "--rank", str(r),
               "--world", str(world), "--device", devices[r], "--protocol", protocol, "--seg-bytes", str(seg),
               "--local-bytes", str(local), "--gpu-bytes", str(gpu_bytes), "--bandwidth-bytes", str(bw),
               "--master", a.master, "--work", a.work]
        log = open(os.path.join(d, f"r{r}.log"), "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        if sequential:
            while not os.path.exists(os.path.join(d, f"r{r}.ready.json")) and procs[-1][0].poll() is None:
                time.sleep(0.5)
    deadline = time.time() + a.phase_timeout
    for p, log in procs:
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            open(os.path.join(d, "ABORT"), "w").close()
    for p, log in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        log.close()
    ranks = []
    for r in range(world):
        path = os.path.join(d, f"r{r}.result.json")
        res = json.load(open(path)) if os.path.exists(path) else {"rank": r, "steps": [], "missing_result": True}
        res["exit_code"] = procs[r][0].returncode
        tail = open(os.path.join(d, f"r{r}.log"), errors="replace").read().splitlines()
        res["log_tail"] = [x for x in tail if any(k in x for k in ("ERROR", "Failed", "failed", "WARNING", "rror"))][-12:]
        ranks.append(res)
    return {"phase": phase, "protocol": protocol, "devices": devices, "segment_bytes": seg, "local_buffer_bytes": local,
            "gpu_bytes": gpu_bytes, "world": world, "seconds": round(time.time() - t0, 1), "ranks": ranks,
            "summary": summarize(ranks)}


def summarize(ranks):
    def all_ok(name, key="ret", want=0):
        vals = [next((s.get(key) for s in r["steps"] if s["step"] == name), None) for r in ranks]
        return {"ok": sum(1 for v in vals if v == want), "of": len(ranks), "values": sorted({str(v) for v in vals})}
    return {"setup": all_ok("setup"), "register_gpu_buffer": all_ok("register_gpu_buffer"),
            "host_put": all_ok("host_put_to_next_rank_segment"),
            "host_get_verified": all_ok("host_get_previous_rank_payload", "verified", True),
            "gpu_put": all_ok("gpu_put_from_to_next_rank_segment"),
            "gpu_get_verified": all_ok("gpu_get_into_previous_rank_payload", "verified", True)}


def parent(a):
    os.makedirs(a.work, exist_ok=True)
    import mooncake
    mdir = os.path.dirname(os.path.abspath(mooncake.__file__))

    def sha(path):
        try:
            return hashlib.sha256(open(path, "rb").read()).hexdigest()
        except OSError:
            return None
    ident = {"mooncake_dir": mdir, "store_so_sha256": sha(os.path.join(mdir, "store.so")),
             "engine_so_sha256": sha(os.path.join(mdir, "engine.so")),
             "mooncake_version": sh(f"{sys.executable} -m pip show mooncake-transfer-engine-cuda13 | head -2")}
    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as w
        ident["vllm_store_worker_sha256"] = sha(w.__file__)
        ident["vllm_version"] = sh(f"{sys.executable} -c 'import vllm; print(vllm.__version__)'")
    except Exception as e:  # noqa: BLE001
        ident["vllm_store_worker_error"] = repr(e)
    ident["expected"] = json.loads(a.expected) if a.expected else {}
    devs = rdma_devices(a.rdma_sysfs)
    gpus = gpu_pci()
    topo_raw = sh(f"{sys.executable} -c \"import os; os.environ['MC_LOG_LEVEL']='ERROR'; "
                  "from mooncake.engine import TransferEngine; print(TransferEngine().get_local_topology(device_name=''))\"")
    try:
        topo = json.loads(topo_raw.splitlines()[-1])
    except Exception:  # noqa: BLE001
        topo = {}
    report = {"schema": "x1lr-b300-rail-probe/v1", "utc_start": now(), "host": socket.gethostname(),
              "identity": ident, "rdma_devices": devs, "gpus": gpus, "mooncake_topology": topo,
              "mooncake_topology_raw_tail": topo_raw[-4000:], "nvidia_smi_topo": sh("nvidia-smi topo -m"),
              "env": {k: v for k, v in os.environ.items() if k.startswith(("MC_", "WITH_NVIDIA", "MOONCAKE_"))},
              "meminfo": sh("grep -E '^(MemTotal|MemAvailable):' /proc/meminfo"), "phases": []}
    write_json(a.out, report)
    master = subprocess.Popen([os.path.join(mdir, "mooncake_master"), f"--rpc_port={a.master.split(':')[1]}",
                               f"--metrics_port={a.master_metrics_port}"],
                              stdout=open(os.path.join(a.work, "master.log"), "w"), stderr=subprocess.STDOUT)
    time.sleep(3)
    report["master"] = {"pid": master.pid, "alive": master.poll() is None, "addr": a.master}
    world = a.world
    ib = [n for n, v in devs.items() if v["driver"].startswith("mlx5") and "ACTIVE" in v["state"]
          and v["link_layer"] == "InfiniBand"]
    efa_map = efa_per_rank_map(world, devs, topo, gpus)
    report["efa_per_rank_map"] = efa_map
    report["ib_hcas"] = ib
    seg, local, gpu_b = a.seg_gib * GiB, a.local_gib * GiB, a.gpu_gib * GiB
    for phase in a.phases.split(","):
        if master.poll() is not None:
            report["phases"].append({"phase": phase, "skipped": "master exited"})
            continue
        if phase == "efa_per_rank":
            devices = [efa_map[r]["device"] or "" for r in range(world)]
            if not all(devices):
                report["phases"].append({"phase": phase, "skipped": "fewer active EFA devices than ranks"})
                continue
            res = run_phase(a, phase, devices, "rdma", seg, local, gpu_b, 0, world)
        elif phase == "ib_one":
            if not ib:
                report["phases"].append({"phase": phase, "skipped": "no ACTIVE InfiniBand HCA"})
                continue
            res = run_phase(a, phase, [ib[0]] * world, "rdma", seg, local, gpu_b, GiB, world)
        elif phase == "ib_split":
            if len(ib) < 2:
                report["phases"].append({"phase": phase, "skipped": "fewer than two ACTIVE InfiniBand HCAs"})
                continue
            res = run_phase(a, phase, [ib[r * len(ib) // world] for r in range(world)], "rdma", seg, local, gpu_b,
                            GiB, world)
        elif phase == "tcp":
            res = run_phase(a, phase, [""] * world, "tcp", 8 * GiB, local, 0, GiB, world)
        elif phase in ("efa_small_per_rank", "efa_small_one"):
            devices = [efa_map[r]["device"] or "" for r in range(world)]
            if phase == "efa_small_one":
                devices = [devices[0]] * world
            if not all(devices):
                report["phases"].append({"phase": phase, "skipped": "fewer active EFA devices than ranks"})
                continue
            res = run_phase(a, phase, devices, "rdma", 8 * GiB, local, 8 * GiB, GiB, world)
        elif phase == "efa_same_device":
            dev = efa_map[0]["device"]
            res = run_phase(a, phase, [dev, dev], "rdma", 200 * GiB, GiB, 0, 0, 2, sequential=True)
        else:
            report["phases"].append({"phase": phase, "skipped": "unknown phase"})
            continue
        print(f"== phase {phase} summary: {json.dumps(res['summary'], sort_keys=True)}", flush=True)
        report["phases"].append(res)
        write_json(a.out, report)
    master.terminate()
    try:
        master.wait(timeout=30)
    except subprocess.TimeoutExpired:
        master.kill()
    report["master_log_tail"] = open(os.path.join(a.work, "master.log"), errors="replace").read()[-3000:]
    report["utc_end"] = now()
    write_json(a.out, report)
    print(json.dumps({p["phase"]: p.get("summary", p.get("skipped")) for p in report["phases"]}, indent=1), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--phase", default="")
    p.add_argument("--phases", default="efa_per_rank,ib_one,ib_split,tcp,efa_same_device")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--protocol", default="rdma")
    p.add_argument("--seg-bytes", type=int, default=0)
    p.add_argument("--local-bytes", type=int, default=0)
    p.add_argument("--gpu-bytes", type=int, default=0)
    p.add_argument("--bandwidth-bytes", type=int, default=0)
    p.add_argument("--seg-gib", type=int, default=281)
    p.add_argument("--local-gib", type=int, default=4)
    p.add_argument("--gpu-gib", type=int, default=64)
    p.add_argument("--master", default="127.0.0.1:50151")
    p.add_argument("--master-metrics-port", type=int, default=19103)
    p.add_argument("--phase-timeout", type=int, default=900)
    p.add_argument("--work", default="/tmp/x1lr_rail_probe")
    p.add_argument("--out", default="/logs/x1lr_rail_probe.json")
    p.add_argument("--expected", default="")
    p.add_argument("--rdma-sysfs", default="/sys/class/infiniband")
    a = p.parse_args()
    return child(a) if a.child else parent(a)


if __name__ == "__main__":
    sys.exit(main())
