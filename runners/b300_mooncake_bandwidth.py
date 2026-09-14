"""Measure Mooncake loopback bandwidth on one node, the way the recipe uses it.

Sweep 34818543261 loaded KV from Mooncake at 34-47 MB/s per get on DSXE while
the B300 NV baseline did ~2 GB/s, so leases expired and vLLM livelocked. This
isolates the layers: raw transfer-engine RDMA (host->host, host->GPU) and the
store path vLLM takes (register_buffer + get_into a GPU buffer).
"""
import multiprocessing
import os
import socket
import subprocess
import sys
import time

MB = 1 << 20


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def gbps(nbytes, seconds):
    return nbytes / seconds / 1e9 if seconds > 0 else float("inf")


def te_server(device, size, queue, done):
    from mooncake.engine import TransferEngine
    engine = TransferEngine()
    port = free_port()
    rc = engine.initialize(f"127.0.0.1:{port}", "P2PHANDSHAKE", "rdma", device)
    buf = engine.allocate_managed_buffer(size) if rc == 0 else 0
    queue.put((rc, port, buf))
    done.wait()


def transfer_engine_test(device, size=256 * MB, iters=8):
    from mooncake.engine import TransferEngine
    queue, done = multiprocessing.Queue(), multiprocessing.Event()
    server = multiprocessing.Process(target=te_server, args=(device, size, queue, done))
    server.start()
    rc, port, peer = queue.get(timeout=120)
    print(f"te server init rc={rc} port={port} peer_buf={peer:#x}", flush=True)
    if rc != 0 or not peer:
        done.set(); server.join(); return
    target = f"127.0.0.1:{port}"
    engine = TransferEngine()
    rc = engine.initialize(f"127.0.0.1:{free_port()}", "P2PHANDSHAKE", "rdma", device)
    print(f"te client init rc={rc}", flush=True)
    host = engine.allocate_managed_buffer(size)
    for name, fn in (("host->host write", engine.transfer_sync_write), ("host<-host read", engine.transfer_sync_read)):
        t0 = time.perf_counter(); rcs = [fn(target, host, peer, size) for _ in range(iters)]
        dt = time.perf_counter() - t0
        print(f"te {name}: {size // MB} MB x{iters} rc={set(rcs)} {gbps(size * iters, dt):.2f} GB/s ({dt / iters * 1e3:.0f} ms each)", flush=True)
    try:
        import torch
        gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
        rc = engine.register_memory(gpu.data_ptr(), size)
        print(f"te register GPU memory rc={rc}", flush=True)
        if rc == 0:
            for name, fn in (("gpu<-host read", engine.transfer_sync_read), ("gpu->host write", engine.transfer_sync_write)):
                t0 = time.perf_counter(); rcs = [fn(target, gpu.data_ptr(), peer, size) for _ in range(iters)]
                dt = time.perf_counter() - t0
                print(f"te {name}: {size // MB} MB x{iters} rc={set(rcs)} {gbps(size * iters, dt):.2f} GB/s ({dt / iters * 1e3:.0f} ms each)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"te GPU path failed: {exc!r}", flush=True)
    done.set(); server.join()


def store_test(device, obj=32 * MB, count=16):
    from mooncake.store import MooncakeDistributedStore
    import torch
    port = free_port()
    master = subprocess.Popen(["mooncake_master", "--port", str(port),
                               "--eviction_high_watermark_ratio=0.95", "--eviction_ratio=0.10"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2)
    store = MooncakeDistributedStore()
    rc = store.setup(f"127.0.0.1:{free_port()}", "P2PHANDSHAKE", 8 << 30, 1 << 30, "rdma", device, f"127.0.0.1:{port}")
    print(f"store setup rc={rc}", flush=True)
    if rc != 0:
        master.kill(); return
    src = torch.full((obj,), 7, dtype=torch.uint8, device="cuda")
    gpu = torch.empty(obj, dtype=torch.uint8, device="cuda")
    host = torch.empty(obj, dtype=torch.uint8, pin_memory=True)
    for name, ptr in (("gpu", gpu.data_ptr()), ("host", host.data_ptr()), ("src", src.data_ptr())):
        print(f"store register_buffer {name} rc={store.register_buffer(ptr, obj)}", flush=True)
    t0 = time.perf_counter(); rcs = [store.put_from(f"bw-{i}", src.data_ptr(), obj) for i in range(count)]
    dt = time.perf_counter() - t0
    print(f"store put_from gpu: {obj // MB} MB x{count} rc={set(rcs)} {gbps(obj * count, dt):.2f} GB/s ({dt / count * 1e3:.0f} ms each)", flush=True)
    for name, ptr in (("gpu", gpu.data_ptr()), ("host", host.data_ptr())):
        t0 = time.perf_counter(); rcs = [store.get_into(f"bw-{i}", ptr, obj) for i in range(count)]
        dt = time.perf_counter() - t0
        print(f"store get_into {name}: {obj // MB} MB x{count} rc={set(rcs)} {gbps(obj * count, dt):.2f} GB/s ({dt / count * 1e3:.0f} ms each)", flush=True)
    keys = [f"bw-{i}" for i in range(8)]
    bufs = [torch.empty(obj, dtype=torch.uint8, device="cuda") for _ in keys]
    for b in bufs:
        store.register_buffer(b.data_ptr(), obj)
    t0 = time.perf_counter(); rcs = store.batch_get_into(keys, [b.data_ptr() for b in bufs], [obj] * len(keys))
    dt = time.perf_counter() - t0
    print(f"store batch_get_into gpu x{len(keys)}: rc={rcs} {gbps(obj * len(keys), dt):.2f} GB/s ({dt * 1e3:.0f} ms)", flush=True)
    print("store data check:", "ok" if bool((gpu == 7).all()) else "MISMATCH", flush=True)
    store.close(); master.kill()


if __name__ == "__main__":
    device = sys.argv[1]
    print("env", {k: v for k, v in os.environ.items() if k.startswith(("MC_", "WITH_NVIDIA", "RDMAV", "MOONCAKE"))}, flush=True)
    for label, test in (("transfer engine", transfer_engine_test), ("store", store_test)):
        print(f"== {label}", flush=True)
        try:
            test(device)
        except Exception as exc:  # noqa: BLE001
            print(f"{label} failed: {exc!r}", flush=True)
