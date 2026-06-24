#!/usr/bin/env python3
"""Read-only DeepEP capability probe (single process, no dist init needed for sigs).

Dumps the exact API surface CollectiveX needs to wire fp8 dispatch + low-latency:
constructor + dispatch/combine/low_latency_* signatures, the LL rdma size hint,
the fp8 per-token cast helpers, and the device. Drives the reject matrix + impl.
Run inside the SGLang container on one GPU; prints to stdout only.
"""
import inspect
import sys


def sig(obj, name):
    fn = getattr(obj, name, None)
    if fn is None:
        return f"  {name}: <ABSENT>"
    try:
        return f"  {name}{inspect.signature(fn)}"
    except (ValueError, TypeError):
        return f"  {name}: <builtin/no-signature>"


def main():
    import torch
    print("=== torch / device ===")
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"device={p.name} sms={p.multi_processor_count} "
              f"mem={p.total_memory/1e9:.0f}GB cc={p.major}.{p.minor}")
    print("fp8 dtypes:", [d for d in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2")
                          if hasattr(torch, d)])

    print("\n=== deep_ep ===")
    import deep_ep
    from deep_ep import Buffer
    print("deep_ep file:", getattr(deep_ep, "__file__", "?"))
    try:
        import importlib.metadata as md
        print("deep_ep version:", md.version("deep_ep"))
    except Exception as e:
        print("deep_ep version: <none>", repr(e))
    print("deep_ep dir:", [n for n in dir(deep_ep) if not n.startswith("_")])
    print("Buffer.num_sms (default):", getattr(Buffer, "num_sms", "<absent>"))

    print("\n=== Buffer signatures ===")
    print(sig(Buffer, "__init__"))
    for m in ("dispatch", "combine", "get_dispatch_layout",
              "low_latency_dispatch", "low_latency_combine",
              "clean_low_latency_buffer", "get_low_latency_rdma_size_hint",
              "get_dispatch_config", "get_combine_config", "set_num_sms",
              "get_buffer_size_hint", "internode_dispatch", "internode_combine"):
        print(sig(Buffer, m))

    print("\n=== fp8 cast helpers ===")
    # The canonical per-token fp8 cast in DeepEP's own tests/utils.
    for modname in ("deep_ep.utils", "deep_ep"):
        try:
            mod = __import__(modname, fromlist=["*"])
            cands = [n for n in dir(mod) if "fp8" in n.lower() or "cast" in n.lower()
                     or "quant" in n.lower()]
            print(f"{modname}: {cands}")
        except Exception as e:
            print(f"{modname}: <import failed> {e!r}")

    print("\n=== LL dispatch source (return shape / fp8 default) ===")
    for m in ("low_latency_dispatch", "low_latency_combine", "dispatch"):
        fn = getattr(Buffer, m, None)
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
            head = "\n".join(src.splitlines()[:45])
            print(f"--- {m} (first 45 lines) ---\n{head}\n")
        except (OSError, TypeError) as e:
            print(f"--- {m}: no source ({e!r}) ---")

    print("\nPROBE_OK")


if __name__ == "__main__":
    sys.exit(main())
