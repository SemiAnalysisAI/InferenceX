"""Compile one sustained-matmul kernel, or run it back to back on one NeuronCore (LNC=1) for a fixed time.

  python ubenchX/full_chip_matmul/run_point.py --target trn3 --kernel mm_bf16 --n 512 --mm 8192 \
    --reps 128 --out runs/x/compile --compile-only
  python ubenchX/full_chip_matmul/run_point.py --neff runs/x/compile/kernel.neff --core 3 --seconds 60 \
    --start-file runs/x/start --probe-neff runs/x/probe/kernel.neff --probe-last-s 12 --out runs/x/core3

A run loads the NEFF through libnrt, creates <out>/ready, waits for --start-file to appear and reads
the start time (epoch seconds) from it, so several processes start together, then calls nrt_execute
until --seconds have passed. It writes execs.csv (host wall-clock start and end of every execution)
and meta.json. With --probe-neff, the same kernel compiled with other --reps, it runs one profiled
execution of that NEFF --probe-last-s seconds before the end and leaves the trace in <out>/probe.ntff.
"""

import argparse
import ctypes
import inspect
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBNRT = "/opt/aws/neuron/lib/libnrt.so.1"
NRT_FRAMEWORK_TYPE_NO_FW = 1
NRT_TENSOR_PLACEMENT_DEVICE = 0
NRT_TENSOR_USAGE_INPUT = 0
NRT_TENSOR_NAME_MAX = 256


class TensorInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * NRT_TENSOR_NAME_MAX), ("usage", ctypes.c_int), ("size", ctypes.c_size_t),
                ("dtype", ctypes.c_int), ("shape", ctypes.POINTER(ctypes.c_uint32)), ("ndim", ctypes.c_uint32)]


def compile_neff(kernel, in_tensor, out_dir, target):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    build = os.path.join(out_dir, "build")
    shutil.rmtree(build, ignore_errors=True)
    os.makedirs(build)
    neff = os.path.join(out_dir, "kernel.neff")
    opts = CompileOptions(target=target, lnc=1, artifacts_dir=build, output_path=neff)
    param = list(inspect.signature(kernel.func).parameters)[0]
    bir, result = compile_to_bir(kernel, frontend=ParserFrontend(), inputs={param: in_tensor}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[in_tensor], argument_names=[param],
                        output_arg_names=getattr(result, "output_names", []))
    if not os.path.exists(neff):
        raise RuntimeError(f"compile produced no NEFF at {neff}")
    return neff


def check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with NRT status {status}")


class Model:
    """One NEFF loaded on the only visible NeuronCore, with device tensors for its inputs and outputs."""

    def __init__(self, nrt, neff, inp):
        self.nrt = nrt
        blob = open(neff, "rb").read()
        self.model = ctypes.c_void_p()
        check(nrt.nrt_load(blob, ctypes.c_size_t(len(blob)), 0, 1, ctypes.byref(self.model)), "nrt_load")
        info = ctypes.c_void_p()
        check(nrt.nrt_get_model_tensor_info(self.model, ctypes.byref(info)), "nrt_get_model_tensor_info")
        count = ctypes.cast(info, ctypes.POINTER(ctypes.c_uint64))[0]
        tensors = ctypes.cast(info.value + 8, ctypes.POINTER(TensorInfo * count))[0]
        self.inputs, self.outputs = ctypes.c_void_p(), ctypes.c_void_p()
        check(nrt.nrt_allocate_tensor_set(ctypes.byref(self.inputs)), "nrt_allocate_tensor_set")
        check(nrt.nrt_allocate_tensor_set(ctypes.byref(self.outputs)), "nrt_allocate_tensor_set")
        for t in tensors:
            tensor = ctypes.c_void_p()
            check(nrt.nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, ctypes.c_size_t(t.size), t.name,
                                          ctypes.byref(tensor)), "nrt_tensor_allocate")
            if t.usage == NRT_TENSOR_USAGE_INPUT:
                if inp.nbytes != t.size:
                    raise RuntimeError(f"input is {inp.nbytes} bytes, NEFF tensor {t.name!r} wants {t.size}")
                check(nrt.nrt_tensor_write(tensor, inp.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(0),
                                           ctypes.c_size_t(t.size)), "nrt_tensor_write")
            tensor_set = self.inputs if t.usage == NRT_TENSOR_USAGE_INPUT else self.outputs
            check(nrt.nrt_add_tensor_to_tensor_set(tensor_set, t.name, tensor), "nrt_add_tensor_to_tensor_set")

    def execute(self):
        check(self.nrt.nrt_execute(self.model, self.inputs, self.outputs), "nrt_execute")


def do_compile(a, out):
    sys.path.insert(0, HERE)
    from kernels import K_PER_PARTITION, get_kernel, make_input

    K = 128 * K_PER_PARTITION[a.kernel]
    inp = os.path.join(out, "input.npy")
    np.save(inp, make_input(a.kernel, a.m, a.n))
    t0 = time.time()
    compile_neff(get_kernel(a.kernel, a.mm, a.reps, a.m), np.load(inp), out, a.target)
    meta = {"kernel": a.kernel, "K": K, "M": a.m, "N": a.n, "num_mm": a.mm, "reps": a.reps,
            "flops_per_mm": 2 * K * a.m * a.n, "mm_per_exec": a.mm * a.reps, "lnc": 1, "target": a.target,
            "compile_s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"compiled {a.kernel} M={a.m} N={a.n} mm={a.mm} reps={a.reps} in {meta['compile_s']} s")


def do_run(a, out):
    neff = os.path.abspath(a.neff)
    src = os.path.dirname(neff)
    meta = json.load(open(os.path.join(src, "meta.json")))
    inp = np.load(os.path.join(src, "input.npy"))
    # libnrt reads both when nrt_init runs; the process then sees only this core, as vnc 0.
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(a.core)
    nrt = ctypes.CDLL(LIBNRT)
    check(nrt.nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, b"", b""), "nrt_init")
    model = Model(nrt, neff, inp)
    probe = Model(nrt, os.path.abspath(a.probe_neff), inp) if a.probe_neff else None
    ready = time.time()
    open(os.path.join(out, "ready"), "w").close()
    while not os.path.exists(a.start_file):
        time.sleep(0.05)
    start_at = float(open(a.start_file).read())
    if time.time() > start_at:
        raise RuntimeError(f"start time {start_at} had passed when {a.start_file} was read")
    time.sleep(start_at - time.time())

    rows, probes = [], []
    end_at = start_at + a.seconds
    k = 0
    while time.time() < end_at:
        k += 1
        t0 = time.time_ns()
        model.execute()
        rows.append((k, t0, time.time_ns()))
        if probe is not None and not probes and time.time() >= end_at - a.probe_last_s:
            ntff = os.path.join(out, "probe.ntff").encode()
            check(nrt.nrt_profile_start(probe.model, ntff), "nrt_profile_start")
            t0 = time.time_ns()
            probe.execute()
            t1 = time.time_ns()
            check(nrt.nrt_profile_stop(ntff), "nrt_profile_stop")
            probes.append({"after_exec": k, "start_ns": t0, "end_ns": t1, "resume_ns": time.time_ns()})
    nrt.nrt_close()

    with open(os.path.join(out, "execs.csv"), "w") as f:
        f.write("exec,start_ns,end_ns\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    meta.update({"core": a.core, "seconds": a.seconds, "start_at": start_at, "ready_at": ready,
                 "probe_last_s": a.probe_last_s, "num_exec": len(rows), "probes": probes})
    if probe is not None:
        pm = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.probe_neff)), "meta.json")))
        meta["probe_reps"] = pm["reps"]
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"done {meta['kernel']} core={a.core} execs={len(rows)} probes={len(probes)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compile-only", action="store_true", help="build the NEFF and stop")
    ap.add_argument("--kernel", choices=["mm_bf16", "mm_fp8dr", "mm_mx", "mm_mxfp4"])
    # trn3 is Trainium3 silicon revision v2 and trn3pre is revision v1.
    ap.add_argument("--target", choices=["trn3", "trn3pre"])
    ap.add_argument("--n", type=int, help="moving free size N (columns streamed per matmul)")
    ap.add_argument("--m", type=int, default=128, help="stationary free size M (output partitions, max 128)")
    ap.add_argument("--mm", type=int, help="matmul instructions per loop iteration")
    ap.add_argument("--reps", type=int, help="on-device loop iterations per execution")
    ap.add_argument("--neff", help="kernel.neff from a --compile-only run (meta.json and input.npy beside it)")
    ap.add_argument("--core", type=int, help="NeuronCore to run on (NEURON_RT_VISIBLE_CORES)")
    ap.add_argument("--seconds", type=float, help="keep executing for this long after the start time")
    ap.add_argument("--start-file", help="file that will hold the start time in epoch seconds")
    ap.add_argument("--probe-neff", help="NEFF of the same kernel to profile once near the end of the run")
    # A probe idles the core for about a second while the profiler arms and stops, and profiling a core
    # stops its sysfs throttle statistic from updating, so there is one probe and it comes last.
    ap.add_argument("--probe-last-s", type=float, help="start the probe this many seconds before the end")
    a = ap.parse_args()

    need = ["kernel", "target", "n", "mm", "reps"] if a.compile_only else ["neff", "core", "seconds", "start_file"]
    if a.probe_neff:
        need.append("probe_last_s")
    missing = [f"--{n.replace('_', '-')}" for n in need if getattr(a, n) is None]
    if missing:
        ap.error("missing " + " ".join(missing))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    (do_compile if a.compile_only else do_run)(a, out)


if __name__ == "__main__":
    main()
